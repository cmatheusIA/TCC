# ============================================================================
# PTv3 + SCALAR BRANCH FUSION — Segmentação supervisionada multiclasse
# ============================================================================
# Arquitetura de dois ramos em paralelo:
#
#   Ramo A (Geométrico): PTv3CompatibleTeacher (LLRD)
#     feature_adapter(16→128) → lfa → blocks → proj → (N, 512)
#
#   Ramo B (Físico): ScalarBranch — 6 stats locais do scalar_field
#     [sf_raw, sf_iqr_norm, sf_mean_knn, sf_std_knn, sf_range_knn, sf_delta]
#     → MLP(6→64→128) com LayerNorm → (N, 128)
#
#   Fusão: concat(512, 128) = 640D → LayerNorm → MLP → logits (N, 5)
#
# Justificativa: scalar_field (col 9) é proxy da largura física que define
# as classes ABNT NBR 6118. Injetar via branch dedicada garante que o sinal
# chegue ao head sem depender do backbone PTv3 aprender a priorizá-lo.
#
# LLRD param groups:
#   scalar_branch : LR × 10  (sem pesos pré-treinados)
#   fusion_head   : LR × 1
#   proj          : LR × 0.7
#   blocks[2]     : LR × 0.7^2
#   blocks[1]     : LR × 0.7^3
#   blocks[0]+lfa : congelados
#
# Classes ABNT NBR 6118 (0-indexed):
#   0 Microfissura < 0.05mm  |  1 Fissura 0.05–0.5mm
#   2 Trinca 0.5–1.5mm       |  3 Rachadura > 1.5mm  |  4 Normal
#
# Referências:
#   Wu et al. (2024) — Point Transformer V3, CVPR Oral (arXiv:2312.10035)
#   Veličković et al. (2018) — Graph Attention Networks
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, time, copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.amp import autocast
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score

from utils.config import (BASE_PATH, DATA_TRAIN_ABNT, DATA_TEST_ABNT,
                           NUM_CLASSES, CLASS_NAMES, NORMAL_CLASS,
                           INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS,
                           setup_logging)
from utils.data import load_folder
from teacher_student_v1 import build_teacher
from ptv3_linear_probe import SegmentationHead, cloud_metrics, CLASS_COLORS
from scalar_gat import multiclass_loss, compute_class_weights

VERSION   = 'ptv3_sf_fusion'
RESULTS   = f'{BASE_PATH}/results_{VERSION}'
MODELS_O  = f'{BASE_PATH}/models_{VERSION}'
PLY_O     = f'{RESULTS}/ply'
CKPT_PATH = f'{MODELS_O}/fusion.pth'
log       = setup_logging(f'{BASE_PATH}/logs_{VERSION}')

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
LR_HEAD      = 1e-4
LR_DECAY     = 0.7
FREEZE_RATIO = 0.5
NUM_EPOCHS   = 120
PROBE_EPOCHS = 20
PATIENCE     = 20
VAL_SPLIT    = 0.20
SF_IDX       = 9      # índice do scalar_field no vetor 16D
K_NEIGHBORS  = 16     # kNN para ScalarBranch
SF_DIM       = 128    # saída da ScalarBranch
GEO_DIM      = 512    # saída do backbone PTv3

# Fatores de repetição para oversampling de classes raras no treino.
# Nuvens que contêm Trinca (2) ou Rachadura (3) são duplicadas.
RARE_FACTORS = {2: 4, 3: 6}  # classe → número de cópias adicionais


# ============================================================================
# SCALAR BRANCH — sinal físico de largura de fissura
# ============================================================================

class ScalarBranch(nn.Module):
    """
    Processa o scalar_field via 6 estatísticas locais kNN → 128D.

    6 features por ponto:
      sf_raw        — valor bruto (col SF_IDX)
      sf_iqr_norm   — (sf - Q25) / (Q75 - Q25)  invariante à calibração do scanner
      sf_mean_knn   — média dos k vizinhos
      sf_std_knn    — desvio padrão dos k vizinhos
      sf_range_knn  — max - min dos k vizinhos
      sf_delta      — sf_raw - sf_mean_knn  (destaque do ponto vs. fundo)
    """

    def __init__(self, k: int = K_NEIGHBORS, out_dim: int = SF_DIM):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _local_stats(self, xyz: torch.Tensor, sf_np: np.ndarray) -> torch.Tensor:
        """Computa stats kNN no CPU via cKDTree. Retorna (N, 4) float32 no device original."""
        k = min(self.k, len(xyz) - 1)
        tree = cKDTree(xyz.detach().cpu().numpy())
        _, idx = tree.query(xyz.detach().cpu().numpy(), k=k)
        nb = sf_np[idx]                           # (N, k)
        mean  = nb.mean(axis=1)
        std   = nb.std(axis=1)
        rng   = nb.max(axis=1) - nb.min(axis=1)
        delta = sf_np - mean
        return torch.tensor(
            np.stack([mean, std, rng, delta], axis=1),
            dtype=torch.float32, device=xyz.device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, INPUT_DIM) → (N, SF_DIM)"""
        xyz = x[:, :3]
        # float32: divisão por IQR pequeno causa overflow em float16 (max=65504) → NaN via LayerNorm
        sf    = x[:, SF_IDX].float()
        sf_np = sf.detach().cpu().numpy()

        q25    = torch.quantile(sf, 0.25)
        q75    = torch.quantile(sf, 0.75)
        iqr    = (q75 - q25).clamp(min=1e-4)
        sf_iqr = (sf - q25) / iqr                 # (N,) invariante ao scanner

        local = self._local_stats(xyz, sf_np)      # (N, 4) float32

        feats = torch.stack([sf, sf_iqr], dim=1)   # (N, 2) float32
        feats = torch.cat([feats, local], dim=1)   # (N, 6) float32
        return self.net(feats)                     # (N, SF_DIM)


# ============================================================================
# FUSION HEAD — concat(geo, sf) → logits
# ============================================================================

class FusionHead(nn.Module):
    """
    Recebe features geométricas (N, GEO_DIM) e físicas (N, SF_DIM),
    concatena e projeta para NUM_CLASSES.
    """

    def __init__(self, in_geo: int = GEO_DIM, in_sf: int = SF_DIM,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()
        in_dim = in_geo + in_sf   # 640
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, geo: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([geo, sf], dim=1))   # (N, NUM_CLASSES)


# ============================================================================
# MODELO COMPLETO
# ============================================================================

class ScalarFusionPTv3(nn.Module):
    """PTv3CompatibleTeacher + ScalarBranch + FusionHead."""

    def __init__(self, teacher: nn.Module, scalar_branch: ScalarBranch,
                 fusion_head: FusionHead):
        super().__init__()
        self.teacher       = teacher
        self.scalar_branch = scalar_branch
        self.fusion_head   = fusion_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        geo  = self.teacher(x)         # (N, 512) — PTv3CompatibleTeacher
        sf   = self.scalar_branch(x)   # (N, 128)
        return self.fusion_head(geo, sf)


# ============================================================================
# PARAM GROUPS — LLRD backbone + high LR para ramos novos
# ============================================================================

def setup_param_groups(model: ScalarFusionPTv3,
                       base_lr: float = LR_HEAD,
                       decay: float   = LR_DECAY,
                       freeze_ratio: float = FREEZE_RATIO) -> list:
    """
    Congela os primeiros freeze_ratio dos blocos do backbone.
    ScalarBranch e FusionHead sempre treináveis com LR alto.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    # ScalarBranch — do zero, LR máximo
    for p in model.scalar_branch.parameters():
        p.requires_grad_(True)

    # FusionHead — do zero
    for p in model.fusion_head.parameters():
        p.requires_grad_(True)

    teacher = model.teacher
    is_compat = hasattr(teacher, 'blocks') and hasattr(teacher, 'proj')

    if is_compat:
        blocks   = list(teacher.blocks)
        n_blocks = len(blocks)
        n_freeze = int(np.ceil(n_blocks * freeze_ratio))

        for i in range(n_freeze, n_blocks):
            for p in blocks[i].parameters():
                p.requires_grad_(True)
        for p in teacher.proj.parameters():
            p.requires_grad_(True)
    else:
        # PTv3Teacher ou S3DIS fallback: descongelar apenas proj_head
        if hasattr(teacher, 'proj_head'):
            for p in teacher.proj_head.parameters():
                p.requires_grad_(True)

    # Montar grupos com LLRD
    param_groups = [
        {'params': list(model.scalar_branch.parameters()),
         'lr': base_lr * 10, 'name': 'scalar_branch'},
        {'params': list(model.fusion_head.parameters()),
         'lr': base_lr, 'name': 'fusion_head'},
    ]

    if is_compat:
        param_groups.append({
            'params': list(teacher.proj.parameters()),
            'lr': base_lr * decay,
            'name': 'proj',
        })
        for offset, i in enumerate(range(n_blocks - 1, n_freeze - 1, -1)):
            param_groups.append({
                'params': list(blocks[i].parameters()),
                'lr': base_lr * (decay ** (offset + 2)),
                'name': f'block_{i}',
            })
    elif hasattr(teacher, 'proj_head'):
        param_groups.append({
            'params': list(teacher.proj_head.parameters()),
            'lr': base_lr * decay,
            'name': 'proj_head',
        })

    trainable = sum(p.numel() for g in param_groups for p in g['params'])
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    log.info(f"Treináveis: {trainable:,}  |  Congelados: {frozen:,}")
    for g in param_groups:
        n = sum(p.numel() for p in g['params'])
        log.info(f"  {g['name']:<20} lr={g['lr']:.2e}  params={n:,}")

    return param_groups


# ============================================================================
# OVERSAMPLING DE CLASSES RARAS
# ============================================================================

def oversample_rare_classes(data: list,
                            rare_factors: dict = RARE_FACTORS) -> list:
    """
    Duplica nuvens de treino que contêm classes raras (Trinca, Rachadura).
    rare_factors = {classe_idx: n_cópias_extras}
    Não modifica os dados originais — devolve nova lista.
    """
    result = list(data)
    for d in data:
        if d['labels'] is None:
            continue
        labels = d['labels']
        extra  = max(
            (rare_factors.get(int(c), 0)
             for c in np.unique(labels)
             if int(c) in rare_factors),
            default=0,
        )
        for _ in range(extra):
            result.append(d)

    counts = {}
    for d in result:
        if d['labels'] is None:
            continue
        for c in np.unique(d['labels']):
            counts[int(c)] = counts.get(int(c), 0) + 1

    log.info(f"Após oversampling: {len(result)} nuvens  "
             f"(clouds/classe: "
             + ", ".join(f"{CLASS_NAMES[c]}={counts.get(c,0)}"
                         for c in range(NUM_CLASSES)) + ")")
    return result


# ============================================================================
# AVALIAÇÃO
# ============================================================================

@torch.no_grad()
def evaluate(model: ScalarFusionPTv3,
             data: list,
             device: torch.device) -> tuple[float, list]:
    """Retorna (mean_macro_f1, lista de dicts por nuvem)."""
    model.eval()
    rows, macro_f1s = [], []

    for d in data:
        if d['labels'] is None:
            continue
        x = torch.tensor(d['features'], dtype=torch.float32).to(device)

        with autocast('cuda'):
            logits = model(x)
            probs  = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

        m = cloud_metrics(d['labels'], probs)
        macro_f1s.append(m['macro_f1'])
        rows.append({'filename': d['filename'],
                     **{k: round(float(v), 4) for k, v in m.items()}})

    return float(np.mean(macro_f1s)) if macro_f1s else 0.0, rows


# ============================================================================
# TREINO
# ============================================================================

def train(model: ScalarFusionPTv3,
          train_data: list, val_data: list,
          param_groups: list,
          weights: torch.Tensor,
          device: torch.device,
          max_epochs: int = NUM_EPOCHS,
          patience: int   = PATIENCE) -> ScalarFusionPTv3:

    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, max_epochs // 4), eta_min=1e-7)

    best_val_f1  = 0.0
    patience_cnt = 0
    best_state   = None

    log.info(f"  Treino: {len(train_data)} nuvens  |  Val: {len(val_data)} nuvens")
    log.info(f"  Epochs: {max_epochs}  Patience: {patience}")

    for epoch in range(max_epochs):
        model.train()
        ep_loss = []
        perm    = np.random.permutation(len(train_data))

        for i in perm:
            d = train_data[i]
            if d['labels'] is None:
                continue
            x = torch.tensor(d['features'], dtype=torch.float32).to(device)
            y = torch.tensor(d['labels'],   dtype=torch.long).to(device)

            optimizer.zero_grad(set_to_none=True)
            try:
                with autocast('cuda'):
                    logits = model(x)
                    loss   = multiclass_loss(logits, y, weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g['params']], 1.0)
                optimizer.step()
                ep_loss.append(loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    log.warning(f"OOM: {d['filename']} — pulando")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

        scheduler.step()

        val_f1, _ = evaluate(model, val_data, device)
        avg_loss  = float(np.mean(ep_loss)) if ep_loss else float('nan')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lrs = [g['lr'] for g in optimizer.param_groups]
            log.info(f"  Epoch {epoch+1:03d}/{max_epochs} | "
                     f"Loss={avg_loss:.5f} | Val macro-F1={val_f1:.4f} | "
                     f"LR_head={lrs[1]:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            best_state   = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch,
                        'model_state_dict': best_state,
                        'val_macro_f1': best_val_f1}, CKPT_PATH)
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            log.info(f"Early stop (epoch {epoch+1}): val macro-F1={val_f1:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        log.info(f"Melhor modelo carregado: val macro-F1={best_val_f1:.4f}")

    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print(f"  PTv3 + SCALAR BRANCH FUSION — ABNT NBR 6118 ({NUM_CLASSES} classes)")
    print(f"  LR_HEAD={LR_HEAD}  LLRD_DECAY={LR_DECAY}  freeze={int(FREEZE_RATIO*100)}%")
    print(f"  SF_IDX={SF_IDX}  K_NEIGHBORS={K_NEIGHBORS}  SF_DIM={SF_DIM}")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    for p in [RESULTS, MODELS_O, PLY_O]:
        os.makedirs(p, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device}")

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    log.info("Carregando dados...")
    all_train = load_folder(DATA_TRAIN_ABNT)
    all_test  = load_folder(DATA_TEST_ABNT)

    labeled = [d for d in all_train + all_test
               if d.get('has_crack') and d['labels'] is not None]
    normal_clouds = [d for d in all_train if not d.get('has_crack', False)]

    np.random.seed(42)
    idx_perm   = np.random.permutation(len(labeled))
    n_val      = max(1, int(len(labeled) * VAL_SPLIT))
    val_data   = [labeled[i] for i in idx_perm[:n_val]]
    train_base = [labeled[i] for i in idx_perm[n_val:]] + normal_clouds

    train_data = oversample_rare_classes(train_base)

    labeled_for_weights = [d for d in train_base if d['labels'] is not None]
    log.info(f"Labeled treino (base): {len(train_base) - len(normal_clouds)} avaria + "
             f"{len(normal_clouds)} normais  |  Val: {len(val_data)}")

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    log.info("Carregando backbone PTv3...")
    teacher      = build_teacher(INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS).to(device)
    scalar_br    = ScalarBranch(k=K_NEIGHBORS, out_dim=SF_DIM).to(device)
    fusion_head  = FusionHead(in_geo=GEO_DIM, in_sf=SF_DIM,
                              num_classes=NUM_CLASSES).to(device)
    model        = ScalarFusionPTv3(teacher, scalar_br, fusion_head)

    weights = compute_class_weights(labeled_for_weights, device)
    log.info(f"Class weights: {[round(float(w), 3) for w in weights]}")
    log.info(f"  " + " | ".join(f"{CLASS_NAMES[i]}={float(weights[i]):.2f}"
                                 for i in range(NUM_CLASSES)))

    t0 = time.time()

    # ── 3a. FASE 1: Linear Probe — backbone 100% congelado ────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"FASE 1 — Linear Probe ({PROBE_EPOCHS} epochs, backbone frozen)")
    log.info(f"{'='*60}")
    for p in model.teacher.parameters():
        p.requires_grad_(False)

    probe_groups = [
        {'params': list(scalar_br.parameters()),   'lr': LR_HEAD * 10, 'name': 'scalar_branch'},
        {'params': list(fusion_head.parameters()), 'lr': LR_HEAD,      'name': 'fusion_head'},
    ]
    model = train(model, train_data, val_data, probe_groups, weights, device,
                  max_epochs=PROBE_EPOCHS, patience=PROBE_EPOCHS)

    # ── 3b. FASE 2: Partial Fine-tuning — LLRD backbone ───────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"FASE 2 — Partial Fine-tuning (LLRD, freeze={int(FREEZE_RATIO*100)}%)")
    log.info(f"{'='*60}")
    param_groups = setup_param_groups(model, base_lr=LR_HEAD,
                                      decay=LR_DECAY, freeze_ratio=FREEZE_RATIO)
    model = train(model, train_data, val_data, param_groups, weights, device,
                  max_epochs=NUM_EPOCHS, patience=PATIENCE)

    log.info(f"Treino total: {(time.time()-t0)/60:.1f} min")

    # ── 4. Avaliação final ────────────────────────────────────────────────────
    log.info("\nAvaliação final (todas as nuvens avaria)...")
    mean_f1, rows = evaluate(model, labeled, device)

    rows_sorted = sorted(rows, key=lambda r: r['macro_f1'])
    log.info(f"\n{'Arquivo':<38} {'macF1':>6} {'mIoU':>6} {'AUROC':>7}")
    log.info("─" * 60)
    for r in rows_sorted:
        flag = ' ◄' if r['macro_f1'] < 0.50 else (' ★' if r['macro_f1'] >= 0.75 else '')
        log.info(f"{r['filename']:<38} {r['macro_f1']:>6.4f} "
                 f"{r['miou']:>6.4f} {r['auroc_ovr']:>7.4f}{flag}")

    macro_f1s = [r['macro_f1']  for r in rows]
    mious     = [r['miou']      for r in rows]
    aurocs    = [r['auroc_ovr'] for r in rows if not np.isnan(r['auroc_ovr'])]

    print("\n" + "="*70)
    print(f"  RESULTADO — {VERSION}")
    print("="*70)
    print(f"  macro-F1  média={np.mean(macro_f1s):.4f}  std={np.std(macro_f1s):.4f}")
    print(f"  mIoU      média={np.mean(mious):.4f}  std={np.std(mious):.4f}")
    print(f"  AUROC-OvR média={np.mean(aurocs):.4f}  std={np.std(aurocs):.4f}")
    # Per-class F1 summary
    for c, name in enumerate(CLASS_NAMES):
        key = f'f1_{name.lower()[:5]}'
        vals = [r[key] for r in rows if key in r]
        if vals:
            print(f"  F1 {name:<15} média={np.mean(vals):.4f}  std={np.std(vals):.4f}")
    print(f"  macro-F1 < 0.50: {sum(1 for f in macro_f1s if f < 0.50)} nuvens")
    print(f"  macro-F1 ≥ 0.75: {sum(1 for f in macro_f1s if f >= 0.75)} nuvens")
    print("="*70)

    # ── 5. CSVs ───────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS, f'per_cloud_{ts}.csv')
    if rows:
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)

    global_csv = os.path.join(RESULTS, f'global_{ts}.csv')
    with open(global_csv, 'w', newline='') as f:
        fieldnames = ['versao', 'macro_f1_mean', 'macro_f1_std',
                      'miou_mean', 'miou_std', 'auroc_ovr_mean']
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerow(dict(versao=VERSION,
                        macro_f1_mean =round(float(np.mean(macro_f1s)), 4),
                        macro_f1_std  =round(float(np.std(macro_f1s)),  4),
                        miou_mean     =round(float(np.mean(mious)),      4),
                        miou_std      =round(float(np.std(mious)),       4),
                        auroc_ovr_mean=round(float(np.mean(aurocs)),     4)))
    log.info(f"CSVs salvos: {RESULTS}/")

    # ── 6. PLY coloridos por classe ───────────────────────────────────────────
    log.info("Salvando PLY coloridos...")
    from plyfile import PlyData, PlyElement
    model.eval()
    for d in labeled:
        x = torch.tensor(d['features'], dtype=torch.float32).to(device)
        with torch.no_grad(), autocast('cuda'):
            preds = model(x).argmax(dim=-1).cpu().numpy()   # (N,)

        xyz     = d['features'][:, :3]
        rgb_out = (d['features'][:, 3:6] * 255).clip(0, 255).astype(np.uint8)
        for c, color in enumerate(CLASS_COLORS):
            if c == NORMAL_CLASS:
                continue
            rgb_out[preds == c] = color

        n      = len(xyz)
        vertex = np.zeros(n, dtype=[('x','f4'),('y','f4'),('z','f4'),
                                     ('red','u1'),('green','u1'),('blue','u1')])
        vertex['x'], vertex['y'], vertex['z'] = xyz[:,0], xyz[:,1], xyz[:,2]
        vertex['red'], vertex['green'], vertex['blue'] = (
            rgb_out[:,0], rgb_out[:,1], rgb_out[:,2])
        ply_path = os.path.join(PLY_O,
                                d['filename'].replace('.ply', f'_{VERSION}.ply'))
        PlyData([PlyElement.describe(vertex, 'vertex')]).write(ply_path)
    log.info(f"PLY em: {PLY_O}")


if __name__ == '__main__':
    main()
