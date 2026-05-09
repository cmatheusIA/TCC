# ============================================================================
# PTv3 PARTIAL FINE-TUNING — Fine-tuning supervisionado (Opção B)
# ============================================================================
# Primeiros 50% dos blocos congelados. Últimos blocos + proj + cabeça treináveis.
# Layer-wise LR Decay (LLRD): lr decai × 0.7 por bloco afastado da cabeça.
#
# Estrutura do backbone PTv3CompatibleTeacher:
#   feature_adapter (frozen)  — 15/16D → 128D
#   lfa            (frozen)   — PTv3CompatibleBlock(128)
#   blocks[0]      (frozen)   — PTv3CompatibleBlock(128)
#   blocks[1]      (trainable, lr × 0.7²)
#   blocks[2]      (trainable, lr × 0.7¹)
#   proj           (trainable, lr × 0.7⁰ = base_lr × 0.7)
#   SegmentationHead (trainable, lr = base_lr)
#
# Loss: CrossEntropyLoss com pesos de classe inverso-frequência
# Protocolo: PTv3 paper (Wu et al., CVPR 2024) para downstream tasks.
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
from ptv3_linear_probe import SegmentationHead, cloud_metrics, CLASS_COLORS
from scalar_gat import multiclass_loss, compute_class_weights

VERSION   = 'ptv3_finetune'
RESULTS   = f'{BASE_PATH}/results_{VERSION}'
MODELS_O  = f'{BASE_PATH}/models_{VERSION}'
PLY_O     = f'{RESULTS}/ply'
CKPT_PATH = f'{MODELS_O}/finetune.pth'
log       = setup_logging(f'{BASE_PATH}/logs_{VERSION}')

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
LR_HEAD      = 1e-4     # cabeça e proj
LR_DECAY     = 0.7      # fator LLRD por bloco
FREEZE_RATIO = 0.5      # fração do backbone a congelar
NUM_EPOCHS   = 100
PROBE_EPOCHS = 20       # fase 1: linear probe com backbone completamente congelado
PATIENCE     = 20
VAL_SPLIT    = 0.20


# ============================================================================
# CONFIGURAÇÃO DE FREEZE + LLRD
# ============================================================================

def setup_partial_finetune(teacher: nn.Module, head: SegmentationHead,
                           freeze_ratio: float = FREEZE_RATIO,
                           base_lr: float = LR_HEAD,
                           decay: float = LR_DECAY) -> list:
    """
    Congela os primeiros freeze_ratio dos blocos do backbone.
    Retorna param_groups com LLRD para AdamW.

    PTv3Teacher  (proj_head): congela apenas o backbone PTv3; treina
                 feature_adapter, sf_branch, lfa, blocks, proj_head.
    PTv3Compatible (proj):    freeze_ratio + LLRD padrão nos blocos finais.

    IMPORTANTE: usa list() em todos os generators de parâmetros antes de
    adicioná-los a param_groups — evitar exaustão do generator ao logar
    tamanhos antes de AdamW materializar o iterador.
    """
    for param in teacher.parameters():
        param.requires_grad_(False)

    is_ptv3_sparse = hasattr(teacher, 'proj_head')  # PTv3Teacher

    if is_ptv3_sparse:
        # PTv3Teacher: backbone (torchsparse) congelado; custom heads treináveis.
        trainable_names = ['feature_adapter', 'sf_branch', 'lfa', 'blocks', 'proj_head']
        active = []
        for name in trainable_names:
            if hasattr(teacher, name):
                for p in getattr(teacher, name).parameters():
                    p.requires_grad_(True)
                active.append(name)

        log.info(f"PTv3Teacher: backbone congelado | treináveis: {active}")

        param_groups = [{'params': list(head.parameters()), 'lr': base_lr, 'name': 'head'}]
        for offset, name in enumerate(reversed(active)):
            param_groups.append({
                'params': list(getattr(teacher, name).parameters()),
                'lr': base_lr * (decay ** (offset + 1)),
                'name': name,
            })

    else:
        # PTv3CompatibleTeacher: freeze_ratio + LLRD
        blocks     = list(teacher.blocks) if hasattr(teacher, 'blocks') else []
        n_blocks   = len(blocks)
        n_freeze   = int(np.ceil(n_blocks * freeze_ratio))
        n_unfreeze = n_blocks - n_freeze

        log.info(f"Blocos totais: {n_blocks}  |  congelados: {n_freeze}  |  treináveis: {n_unfreeze}")

        for i in range(n_freeze, n_blocks):
            for param in blocks[i].parameters():
                param.requires_grad_(True)

        if hasattr(teacher, 'proj'):
            for param in teacher.proj.parameters():
                param.requires_grad_(True)

        param_groups = [{'params': list(head.parameters()), 'lr': base_lr, 'name': 'head'}]

        if hasattr(teacher, 'proj'):
            param_groups.append({
                'params': list(teacher.proj.parameters()),
                'lr': base_lr * decay,
                'name': 'proj',
            })

        for offset, i in enumerate(range(n_blocks - 1, n_freeze - 1, -1)):
            lr_i = base_lr * (decay ** (offset + 2))
            param_groups.append({
                'params': list(blocks[i].parameters()),
                'lr': lr_i,
                'name': f'block_{i}',
            })

    trainable = sum(p.numel() for g in param_groups for p in g['params'])
    frozen    = sum(p.numel() for p in teacher.parameters() if not p.requires_grad)
    log.info(f"Parâmetros treináveis: {trainable:,}  |  congelados: {frozen:,}")
    for g in param_groups:
        n = sum(p.numel() for p in g['params'])
        log.info(f"  {g['name']:<15} lr={g['lr']:.2e}  params={n:,}")

    return param_groups


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

        bottleneck = teacher(x)
        with autocast('cuda'):
            logits = head(bottleneck)
            probs  = F.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)

        m = cloud_metrics(labels, probs)
        macro_f1s.append(m['macro_f1'])
        rows.append({'filename': d['filename'],
                     **{k: round(float(v), 4) for k, v in m.items()}})

    return float(np.mean(macro_f1s)) if macro_f1s else 0.0, rows


def train(teacher: nn.Module, head: SegmentationHead,
          train_data: list, val_data: list,
          param_groups: list, weights: torch.Tensor, device: torch.device,
          max_epochs: int = NUM_EPOCHS, patience: int = PATIENCE):

    optimizer = AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(10, max_epochs // 4), eta_min=1e-7)

    best_val_f1  = 0.0
    patience_cnt = 0
    best_state   = None

    log.info(f"\n  Treino: {len(train_data)} nuvens  |  Val: {len(val_data)} nuvens")
    log.info(f"  Epochs: {max_epochs}  Patience: {patience}")

    for epoch in range(max_epochs):
        teacher.train()
        head.train()
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
                # torchsparse (PTv3) não suporta FP16 — teacher sempre em FP32
                bottleneck = teacher(x)
                with autocast('cuda'):
                    logits = head(bottleneck)
                    loss   = multiclass_loss(logits, y, weights)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g['params']], 1.0)
                optimizer.step()
                ep_loss.append(loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    log.warning(f"OOM na nuvem {d['filename']} — pulando")
                    optimizer.zero_grad(set_to_none=True)
                    continue
                raise

        scheduler.step()

        val_f1, _ = evaluate(teacher, head, val_data, device)
        avg_loss  = float(np.mean(ep_loss)) if ep_loss else float('nan')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            lrs = [g['lr'] for g in optimizer.param_groups]
            log.info(f"  Epoch {epoch+1:03d}/{max_epochs} | "
                     f"Loss={avg_loss:.5f} | Val macro-F1={val_f1:.4f} | "
                     f"LR_head={lrs[0]:.2e}")

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            patience_cnt = 0
            best_state   = {
                'teacher': copy.deepcopy(teacher.state_dict()),
                'head'   : copy.deepcopy(head.state_dict()),
            }
            torch.save({'epoch': epoch, **best_state, 'val_macro_f1': best_val_f1},
                       CKPT_PATH)
        else:
            patience_cnt += 1

        if patience_cnt >= patience:
            log.info(f"Early stop (epoch {epoch+1}): val macro-F1={val_f1:.4f}")
            break

    if best_state is not None:
        teacher.load_state_dict(best_state['teacher'])
        head.load_state_dict(best_state['head'])
        log.info(f"Melhor modelo carregado: val macro-F1={best_val_f1:.4f}")

    return teacher, head


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print(f"  PTv3 PARTIAL FINE-TUNING — ABNT NBR 6118 ({NUM_CLASSES} classes)")
    print(f"  LLRD: base_lr={LR_HEAD}  decay={LR_DECAY}  freeze={int(FREEZE_RATIO*100)}%")
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
    train_data = [labeled[i] for i in idx_perm[n_val:]] + normal_clouds

    labeled_train = [d for d in train_data if d['labels'] is not None]
    log.info(f"Labeled treino: {len(train_data) - len(normal_clouds)} avaria + "
             f"{len(normal_clouds)} normais  |  Val: {len(val_data)} avaria")

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    log.info("Carregando backbone PTv3...")
    teacher = build_teacher(INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS).to(device)
    head    = SegmentationHead(in_dim=512, num_classes=NUM_CLASSES).to(device)

    weights = compute_class_weights(labeled_train, device)
    log.info(f"Class weights: {weights.tolist()}")

    t0 = time.time()

    # ── 3a. FASE 1: Linear Probe — backbone 100% congelado ────────────────────
    log.info(f"\n{'='*60}")
    log.info(f"FASE 1 — Linear Probe ({PROBE_EPOCHS} epochs, backbone frozen)")
    log.info(f"{'='*60}")
    for p in teacher.parameters():
        p.requires_grad_(False)
    probe_groups = [{'params': list(head.parameters()), 'lr': LR_HEAD * 10, 'name': 'head'}]
    teacher, head = train(
        teacher, head, train_data, val_data,
        probe_groups, weights, device,
        max_epochs=PROBE_EPOCHS, patience=PROBE_EPOCHS,
    )

    # ── 3b. FASE 2: Partial Fine-tuning — LLRD nos blocos finais ─────────────
    log.info(f"\n{'='*60}")
    log.info(f"FASE 2 — Partial Fine-tuning (LLRD, freeze_ratio={FREEZE_RATIO})")
    log.info(f"{'='*60}")
    param_groups = setup_partial_finetune(teacher, head,
                                          freeze_ratio=FREEZE_RATIO,
                                          base_lr=LR_HEAD,
                                          decay=LR_DECAY)
    teacher, head = train(
        teacher, head, train_data, val_data,
        param_groups, weights, device,
        max_epochs=NUM_EPOCHS, patience=PATIENCE,
    )

    log.info(f"Treino total: {(time.time()-t0)/60:.1f} min")

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
                        macro_f1_mean =round(float(np.mean(macro_f1s)), 4),
                        macro_f1_std  =round(float(np.std(macro_f1s)),  4),
                        miou_mean     =round(float(np.mean(mious)),      4),
                        miou_std      =round(float(np.std(mious)),       4),
                        auroc_ovr_mean=round(float(np.mean(aurocs)),     4)))
    log.info(f"CSVs salvos: {RESULTS}/")

    # ── 6. PLY colorido por classe ────────────────────────────────────────────
    log.info("Salvando PLY coloridos...")
    from plyfile import PlyData, PlyElement
    teacher.eval(); head.eval()
    for d in labeled:
        x = torch.tensor(d['features'], dtype=torch.float32).to(device)
        with torch.no_grad():
            bottleneck = teacher(x)
        with torch.no_grad(), autocast('cuda'):
            logits = head(bottleneck)
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
