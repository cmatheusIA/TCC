# ============================================================================
# PTv3 LINEAR PROBE — Fine-tuning supervisionado (Opção A)
# ============================================================================
# Backbone PTv3 completamente congelado.
# Só a cabeça de segmentação é treinável.
#
# Objetivo: validar se as features PTv3 (pré-treinadas em ScanNet200) já
# codificam sinal suficiente para segmentar 5 classes ABNT NBR 6118 com
# uma cabeça linear.
#
# Critério de interpretação:
#   macro-F1 > 0.60 → backbone já discrimina; cabeça linear suficiente
#   macro-F1 < 0.45 → precisa de fine-tuning (usar ptv3_finetune.py)
#
# Arquitetura:
#   PTv3 backbone (100% frozen) → bottleneck (N, 512)
#   SegmentationHead: Linear(512→128) → BN → GELU → Dropout(0.3) → Linear(128→NUM_CLASSES)
#
# Loss: CrossEntropyLoss com pesos de classe inverso-frequência
# Split: 80% nuvens avaria treino / 20% validação (por nuvem)
#
# Referências:
#   Wu et al. (2024) — Point Transformer V3, CVPR Oral.
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
from sklearn.metrics import f1_score, jaccard_score, roc_auc_score

from utils.config import (BASE_PATH, DATA_TRAIN_ABNT, DATA_TEST_ABNT,
                           NUM_CLASSES, CLASS_NAMES, NORMAL_CLASS,
                           INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS,
                           setup_logging)
from utils.data import load_folder
from teacher_student_v1 import build_teacher, setup_logging as _sl
from scalar_gat import multiclass_loss, compute_class_weights

VERSION  = 'ptv3_probe'
RESULTS  = f'{BASE_PATH}/results_{VERSION}'
MODELS_O = f'{BASE_PATH}/models_{VERSION}'
PLY_O    = f'{RESULTS}/ply'
HEAD_PATH = f'{MODELS_O}/probe_head.pth'
log      = setup_logging(f'{BASE_PATH}/logs_{VERSION}')

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
LR_HEAD    = 1e-3
NUM_EPOCHS = 150
PATIENCE   = 15
VAL_SPLIT  = 0.20

# Paleta de cores por classe para PLY (RGB 0-255)
CLASS_COLORS = [
    (220,  50,  50),   # 0 Microfissura — vermelho
    (255, 140,   0),   # 1 Fissura      — laranja
    (200, 200,   0),   # 2 Trinca       — amarelo
    (140,   0,   0),   # 3 Rachadura    — vermelho escuro
    (  0,   0,   0),   # 4 Normal       — ignorado (usa RGB original)
]


# ============================================================================
# CABEÇA DE SEGMENTAÇÃO
# ============================================================================

class SegmentationHead(nn.Module):
    """MLP(512→128→NUM_CLASSES) com BatchNorm e Dropout."""

    def __init__(self, in_dim: int = 512, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (N, NUM_CLASSES)


# ============================================================================
# MÉTRICAS POR NUVEM
# ============================================================================

def cloud_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """
    labels : (N,) int64 — ground truth 0-indexed
    probs  : (N, NUM_CLASSES) float32 — softmax probabilities
    Returns dict com macro_f1, weighted_f1, miou, auroc_ovr, per-class f1.
    """
    preds = probs.argmax(axis=1)

    # Só avaliar nas classes que aparecem nesta nuvem — evita F1=0 espúrio para
    # classes ausentes (e.g. Fissura não aparece nos arquivos de validação avaria).
    present = sorted(np.unique(labels).tolist())

    macro_f1    = float(f1_score(labels, preds, labels=present,
                                 average='macro',    zero_division=0))
    weighted_f1 = float(f1_score(labels, preds, labels=present,
                                 average='weighted', zero_division=0))
    miou        = float(jaccard_score(labels, preds, labels=present,
                                      average='macro', zero_division=0))
    per_class   = f1_score(labels, preds, labels=list(range(NUM_CLASSES)),
                           average=None, zero_division=0).tolist()
    try:
        auroc_ovr = float(roc_auc_score(
            labels, probs[:, present] if len(present) < NUM_CLASSES else probs,
            multi_class='ovr', average='macro',
            labels=present
        ))
    except Exception:
        auroc_ovr = float('nan')

    row = dict(macro_f1=macro_f1, weighted_f1=weighted_f1,
               miou=miou, auroc_ovr=auroc_ovr)
    for c in range(NUM_CLASSES):
        row[f'f1_{CLASS_NAMES[c].lower()[:5]}'] = per_class[c]
    return row


# ============================================================================
# TREINO E AVALIAÇÃO
# ============================================================================

@torch.no_grad()
def evaluate(teacher: nn.Module, head: SegmentationHead,
             data: list, device: torch.device) -> tuple[float, list]:
    """Retorna (mean_macro_f1, lista de dicts por nuvem)."""
    teacher.eval(); head.eval()
    rows, macro_f1s = [], []

    for d in data:
        if d['labels'] is None:
            continue
        x      = torch.tensor(d['features'], dtype=torch.float32).to(device)
        labels = d['labels']

        with autocast('cuda'):
            bottleneck = teacher(x)
            logits     = head(bottleneck)
            probs      = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

        m = cloud_metrics(labels, probs)
        macro_f1s.append(m['macro_f1'])
        rows.append({'filename': d['filename'],
                     **{k: round(float(v), 4) for k, v in m.items()}})

    return float(np.mean(macro_f1s)) if macro_f1s else 0.0, rows


def train(teacher: nn.Module, head: SegmentationHead,
          train_data: list, val_data: list,
          weights: torch.Tensor, device: torch.device) -> SegmentationHead:

    optimizer = AdamW(head.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, eta_min=1e-6)

    best_val_f1  = 0.0
    patience_cnt = 0
    best_state   = None

    log.info(f"\nLinear Probe — {len(train_data)} nuvens treino, {len(val_data)} validação")
    log.info(f"Epochs: {NUM_EPOCHS}  LR: {LR_HEAD}  Patience: {PATIENCE}")

    for epoch in range(NUM_EPOCHS):
        teacher.eval()   # backbone sempre frozen
        head.train()
        ep_loss = []
        perm    = np.random.permutation(len(train_data))

        for step, i in enumerate(perm):
            d = train_data[i]
            if d['labels'] is None:
                continue
            x = torch.tensor(d['features'], dtype=torch.float32).to(device)
            y = torch.tensor(d['labels'],   dtype=torch.long).to(device)

            optimizer.zero_grad(set_to_none=True)
            try:
                with torch.no_grad(), autocast('cuda'):
                    bottleneck = teacher(x)
                with autocast('cuda'):
                    logits = head(bottleneck)
                    loss   = multiclass_loss(logits, y, weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                ep_loss.append(loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

        scheduler.step()

        val_f1, _ = evaluate(teacher, head, val_data, device)
        avg_loss  = float(np.mean(ep_loss)) if ep_loss else float('nan')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch+1:03d}/{NUM_EPOCHS} | "
                     f"Loss={avg_loss:.5f} | Val macro-F1={val_f1:.4f} | "
                     f"LR={optimizer.param_groups[0]['lr']:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            best_state   = copy.deepcopy(head.state_dict())
            torch.save({'epoch': epoch, 'head': best_state, 'val_macro_f1': best_val_f1},
                       HEAD_PATH)
        else:
            patience_cnt += 1

        if patience_cnt >= PATIENCE:
            log.info(f"Early stop (epoch {epoch+1}): val macro-F1={val_f1:.4f}")
            break

    if best_state is not None:
        head.load_state_dict(best_state)
        log.info(f"Melhor head carregado: val macro-F1={best_val_f1:.4f}")

    return head


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print(f"  PTv3 LINEAR PROBE — ABNT NBR 6118 ({NUM_CLASSES} classes)")
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
    idx_perm = np.random.permutation(len(labeled))
    n_val    = max(1, int(len(labeled) * VAL_SPLIT))
    val_data   = [labeled[i] for i in idx_perm[:n_val]]
    train_data = [labeled[i] for i in idx_perm[n_val:]] + normal_clouds

    labeled_train = [d for d in train_data if d['labels'] is not None]
    log.info(f"Labeled treino: {len(train_data) - len(normal_clouds)} avaria + "
             f"{len(normal_clouds)} normais  |  Val: {len(val_data)} avaria")

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    log.info("Carregando backbone PTv3...")
    teacher = build_teacher(INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS).to(device)

    for param in teacher.parameters():
        param.requires_grad_(False)
    frozen = sum(p.numel() for p in teacher.parameters())
    log.info(f"Backbone congelado: {frozen:,} parâmetros")

    head = SegmentationHead(in_dim=512, num_classes=NUM_CLASSES).to(device)
    trainable = sum(p.numel() for p in head.parameters())
    log.info(f"Cabeça treinável : {trainable:,} parâmetros")

    weights = compute_class_weights(labeled_train, device)
    log.info(f"Class weights: {weights.tolist()}")

    # ── 3. Treino ─────────────────────────────────────────────────────────────
    t0   = time.time()
    head = train(teacher, head, train_data, val_data, weights, device)
    log.info(f"Treino: {(time.time()-t0)/60:.1f} min")

    # ── 4. Avaliação final ────────────────────────────────────────────────────
    log.info("\nAvaliação final (todas as nuvens avaria)...")
    mean_f1, rows = evaluate(teacher, head, labeled, device)

    rows_sorted = sorted(rows, key=lambda r: r['macro_f1'])
    log.info(f"\n{'Arquivo':<38} {'macF1':>6} {'mIoU':>6} {'AUROC':>7}")
    log.info("─" * 60)
    for r in rows_sorted:
        flag = ' ◄' if r['macro_f1'] < 0.50 else (' ★' if r['macro_f1'] >= 0.75 else '')
        log.info(f"{r['filename']:<38} {r['macro_f1']:>6.4f} {r['miou']:>6.4f} "
                 f"{r['auroc_ovr']:>7.4f}{flag}")

    macro_f1s = [r['macro_f1']  for r in rows]
    mious     = [r['miou']      for r in rows]
    aurocs    = [r['auroc_ovr'] for r in rows if not np.isnan(r['auroc_ovr'])]

    print("\n" + "="*70)
    print(f"  RESULTADO — {VERSION}")
    print("="*70)
    print(f"  macro-F1  média={np.mean(macro_f1s):.4f}  std={np.std(macro_f1s):.4f}")
    print(f"  mIoU      média={np.mean(mious):.4f}  std={np.std(mious):.4f}")
    print(f"  AUROC-OvR média={np.mean(aurocs):.4f}  std={np.std(aurocs):.4f}")
    print(f"  macro-F1 < 0.50: {sum(1 for f in macro_f1s if f < 0.50)} nuvens")
    print(f"  macro-F1 ≥ 0.75: {sum(1 for f in macro_f1s if f >= 0.75)} nuvens")
    print("="*70)

    # ── 5. CSV ────────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS, f'per_cloud_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    global_csv = os.path.join(RESULTS, f'global_{ts}.csv')
    with open(global_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['versao', 'macro_f1_mean', 'macro_f1_std',
                                           'miou_mean', 'miou_std', 'auroc_ovr_mean'])
        w.writeheader()
        w.writerow(dict(versao=VERSION,
                        macro_f1_mean=round(float(np.mean(macro_f1s)), 4),
                        macro_f1_std =round(float(np.std(macro_f1s)),  4),
                        miou_mean    =round(float(np.mean(mious)),      4),
                        miou_std     =round(float(np.std(mious)),       4),
                        auroc_ovr_mean=round(float(np.mean(aurocs)),    4)))
    log.info(f"CSVs salvos: {RESULTS}/")

    # ── 6. PLY colorido por classe ────────────────────────────────────────────
    log.info("Salvando PLY coloridos...")
    from plyfile import PlyData, PlyElement
    teacher.eval(); head.eval()
    for d in labeled:
        x = torch.tensor(d['features'], dtype=torch.float32).to(device)
        with torch.no_grad(), autocast('cuda'):
            logits = head(teacher(x))
            preds  = logits.argmax(dim=-1).cpu().numpy()   # (N,)

        xyz     = d['features'][:, :3]
        rgb_out = (d['features'][:, 3:6] * 255).clip(0, 255).astype(np.uint8)
        for c, color in enumerate(CLASS_COLORS):
            if c == NORMAL_CLASS:
                continue
            mask = (preds == c)
            rgb_out[mask] = color

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
