"""
PTv3 Fine-tune Binário — supervisionado com backbone Point Transformer V3.

Estratégia: congela primeiros 50% dos blocos (FREEZE_RATIO=0.5),
treina últimos blocos + head binária com LLRD (lr × 0.7 por bloco).

Referências:
  Wu et al. CVPR 2024 arXiv:2312.10035 (PTv3)
  Jiang et al. Automation in Construction 2023 (crack detection DL)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, time, copy
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from utils.config import (BASE_PATH, DATA_TRAIN_BIN, INPUT_DIM,
                           PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS,
                           POS_WEIGHT_DEFAULT, setup_logging)
from utils.data import load_folder
from utils.augmentation import augment_cloud, CrackDatabase, crack_paste
from scalar_gat import binary_focal_loss, compute_pos_weight
from utils.visualization import save_crack_ply, plot_loco_metrics

VERSION   = 'ptv3_binary'
RESULTS   = f'{BASE_PATH}/results_{VERSION}'
PLY_DIR   = f'{RESULTS}/ply'
VIS_DIR   = f'{RESULTS}/vis'
_VIS_KEYS = {'xyz', 'rgb_orig', 'preds', 'gt_labels'}
log       = setup_logging(f'{BASE_PATH}/logs_{VERSION}')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR_HEAD      = 1e-4
LR_DECAY     = 0.7
FREEZE_RATIO = 0.5
NUM_EPOCHS   = 100
PATIENCE     = 20
VAL_SPLIT    = 0.20
RNG          = np.random.default_rng(42)


class BinarySegHead(nn.Module):
    """Cabeça de segmentação binária: D_MODEL → 1."""
    def __init__(self, d_model: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (N, 1)


def build_binary_model():
    """
    Carrega backbone PTv3 via build_teacher e adiciona cabeça binária.
    Usa mesma cadeia de fallback que ptv3_finetune.py.
    """
    from legacy.teacher_student_v1 import build_teacher
    teacher = build_teacher(
        input_dim=INPUT_DIM,
        ptv3_ckpt=PTRANSF_WEIGHTS,
        s3dis_ckpt=PTRANSF_WEIGHTS_S3DIS,
    ).to(DEVICE)
    head = BinarySegHead(d_model=128).to(DEVICE)
    return teacher, head


def setup_partial_finetune(teacher: nn.Module, head: BinarySegHead,
                           freeze_ratio: float = FREEZE_RATIO,
                           base_lr: float = LR_HEAD,
                           decay: float = LR_DECAY) -> list:
    """Congela primeiros freeze_ratio blocos. LLRD nos blocos restantes."""
    for p in teacher.parameters():
        p.requires_grad = False

    named_blocks = []
    for name, module in teacher.named_modules():
        if hasattr(module, 'weight') and 'blocks' in name:
            named_blocks.append((name, module))

    n_freeze = int(len(named_blocks) * freeze_ratio)
    for i, (name, module) in enumerate(named_blocks):
        if i >= n_freeze:
            for p in module.parameters():
                p.requires_grad = True

    param_groups = []
    trainable_blocks = [(n, m) for i, (n, m) in enumerate(named_blocks) if i >= n_freeze]
    for dist_from_head, (name, module) in enumerate(reversed(trainable_blocks)):
        lr = base_lr * (decay ** dist_from_head)
        params = list(module.parameters())
        if params:
            param_groups.append({'params': params, 'lr': lr})

    param_groups.append({'params': list(head.parameters()), 'lr': base_lr})
    return param_groups


def _loco_fold(i: int, test_cloud: dict, train_set: list) -> dict | None:
    teacher, head = build_binary_model()
    param_groups  = setup_partial_finetune(teacher, head)
    opt = AdamW(param_groups, weight_decay=1e-4)

    pos_weight = compute_pos_weight(train_set, DEVICE)
    best_val   = float('inf')
    patience   = 0
    best_head  = copy.deepcopy(head.state_dict())

    val_n = max(1, int(len(train_set) * VAL_SPLIT))
    val_set   = train_set[:val_n]
    tr_set    = train_set[val_n:]

    for epoch in range(NUM_EPOCHS):
        teacher.train(); head.train()
        RNG.shuffle(tr_set)
        for cloud in tr_set:
            feat = cloud['features']
            lbl  = cloud['labels']
            if lbl is None:
                continue
            feat_aug, lbl_aug = augment_cloud(feat, lbl, rng=RNG)
            x = torch.tensor(feat_aug, device=DEVICE)
            y = torch.tensor(lbl_aug, dtype=torch.float32, device=DEVICE)
            feats_enc = teacher(x)          # backbone features (N, D)
            logits    = head(feats_enc).squeeze(-1)   # (N,)
            loss = binary_focal_loss(logits, y, pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()

        # Validação
        teacher.eval(); head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cloud in val_set:
                if cloud['labels'] is None:
                    continue
                x   = torch.tensor(cloud['features'], device=DEVICE)
                y   = torch.tensor(cloud['labels'], dtype=torch.float32, device=DEVICE)
                log_ = head(teacher(x)).squeeze(-1)
                val_loss += binary_focal_loss(log_, y, pos_weight).item()
        if val_loss < best_val:
            best_val = val_loss
            best_head = copy.deepcopy(head.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    # Avaliação no fold de teste
    head.load_state_dict(best_head)
    teacher.eval(); head.eval()
    labels_np = test_cloud['labels']
    if labels_np is None or (labels_np == 1).sum() < 5:
        return None

    with torch.no_grad():
        x = torch.tensor(test_cloud['features'], device=DEVICE)
        probs = torch.sigmoid(head(teacher(x)).squeeze(-1)).cpu().numpy()

    auroc = float(roc_auc_score(labels_np, probs))
    ap    = float(average_precision_score(labels_np, probs))
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(labels_np, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])]) if len(thr) > 0 else 0.5
    preds = (probs >= best_thr).astype(np.int64)
    f1 = float(f1_score(labels_np, preds, zero_division=0))

    log.info(f"  Fold {i+1} {test_cloud.get('filename','?')} — "
             f"AUROC={auroc:.4f} F1={f1:.4f} AP={ap:.4f}")
    return {
        'filename' : test_cloud.get('filename', '?'),
        'auroc'    : round(auroc, 4),
        'f1'       : round(f1,   4),
        'ap'       : round(ap,   4),
        'n_crack'  : int((labels_np == 1).sum()),
        'n_normal' : int((labels_np == 0).sum()),
        # dados de visualização (filtrados antes do CSV)
        'xyz'      : test_cloud['features'][:, :3].astype(np.float32),
        'rgb_orig' : test_cloud['features'][:, 3:6].astype(np.float32),
        'preds'    : preds,
        'gt_labels': labels_np,
    }


def main():
    os.makedirs(RESULTS, exist_ok=True)
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')

    all_data = load_folder(DATA_TRAIN_BIN)
    labeled  = [d for d in all_data if d.get('has_crack') and d['labels'] is not None]
    normals  = [d for d in all_data if not d.get('has_crack', False)]

    # CrackPaste — augmentar conjunto de treino
    db = CrackDatabase(labeled)
    synthetic = []
    if len(db) > 0:
        rng = np.random.default_rng(42)
        for n_cloud in normals:
            for _ in range(2):
                synthetic.append(crack_paste(n_cloud, db.sample(rng), rng))
    log.info(f"Dataset: {len(labeled)} avaria | {len(normals)} normais | "
             f"{len(synthetic)} sintéticos (CrackPaste)")

    results = []
    for i, test_cloud in enumerate(labeled):
        train_set = [c for j, c in enumerate(labeled) if j != i] + normals + synthetic
        r = _loco_fold(i, test_cloud, train_set)
        if r:
            results.append(r)

    if not results:
        log.error("Sem resultados")
        return

    aurocs = [r['auroc'] for r in results]
    f1s    = [r['f1']    for r in results]
    print(f"\n{'='*55}")
    print(f"  PTv3 Binary LOCO | AUROC {np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
          f"| F1 {np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    print(f"{'='*55}")

    # PLY — avarias preditas em vermelho, restante em cor original
    os.makedirs(PLY_DIR, exist_ok=True)
    for r in results:
        if 'xyz' in r:
            save_crack_ply(
                xyz=r['xyz'], rgb_orig=r['rgb_orig'],
                pred_labels=r['preds'],
                out_path=os.path.join(PLY_DIR,
                                      r['filename'].replace('.ply', '_ptv3.ply')),
                gt_labels=r.get('gt_labels'),
            )
    log.info(f"PLY salvos em: {PLY_DIR}")

    # Gráficos 2D de métricas
    plot_loco_metrics(results, VIS_DIR, 'PTv3Binary', ts)

    # CSV — filtra arrays numpy antes de escrever
    csv_results = [{k: v for k, v in r.items() if k not in _VIS_KEYS}
                   for r in results]
    csv_path = f'{RESULTS}/loco_{ts}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    log.info(f"CSV: {csv_path}")


if __name__ == '__main__':
    import os
    main()
