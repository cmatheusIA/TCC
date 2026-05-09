# ============================================================================
# ScalarMAE — Semi-supervised MAE pre-training + Mean Teacher fine-tuning
# ============================================================================
# Phase 1: MAE on normal clouds (scalar-guided masking)
#   PCP-MAE (Pang et al., NeurIPS 2024): mask patches by scalar_field score
# Phase 2: Mean Teacher fine-tuning (labeled + consistency on unlabeled)
# ============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, copy, time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from utils.config import (BASE_PATH, DATA_TRAIN_BIN, setup_logging, POS_WEIGHT_DEFAULT)
from utils.data import load_folder
from scalar_gat import binary_focal_loss, compute_pos_weight
from utils.visualization import save_crack_ply, plot_loco_metrics

log = setup_logging(f'{BASE_PATH}/logs_scalar_mae')

RESULTS_DIR = f'{BASE_PATH}/results_scalar_mae'
PLY_DIR     = f'{RESULTS_DIR}/ply'
VIS_DIR     = f'{RESULTS_DIR}/vis'
_VIS_KEYS   = {'xyz', 'rgb_orig', 'preds', 'gt_labels'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

M_PATCHES        = 64     # FPS centers per cloud
K_PATCH          = 32     # points per patch
TOKEN_DIM        = 35     # 16D mean + 16D std + 3D center
D_MODEL          = 128
N_HEADS          = 4
N_ENC_LAYERS     = 4
N_DEC_LAYERS     = 2
EMA_DECAY        = 0.999
PRETRAIN_EPOCHS  = 50
FINETUNE_EPOCHS  = 80
PATIENCE         = 20
LR_PRETRAIN      = 2e-4
LR_FINETUNE      = 1e-4
LAMBDA_CONS_MAX  = 0.5
LAMBDA_WARMUP    = 20   # epochs to ramp lambda from 0 → 0.5


# ── Patch construction ─────────────────────────────────────────────────────

def fps_centroids(xyz: torch.Tensor, M: int) -> torch.Tensor:
    """Greedy FPS — returns (min(M,N), 3)."""
    N = xyz.size(0)
    M = min(M, N)
    idx   = [torch.randint(N, (1,), device=xyz.device).item()]
    dists = torch.full((N,), float('inf'), device=xyz.device)
    for _ in range(M - 1):
        c     = xyz[idx[-1]]
        d     = torch.norm(xyz - c, dim=1)
        dists = torch.minimum(dists, d)
        idx.append(int(dists.argmax()))
    return xyz[torch.tensor(idx, device=xyz.device)]


def build_patches(
    features: torch.Tensor,
    centers: torch.Tensor,
    k: int = K_PATCH,
) -> tuple:
    """
    For each center, find k nearest points → patch features.
    Returns (patches [M, k, 16], sf_means [M]).
    """
    xyz      = features[:, :3].cpu().numpy()
    tree     = cKDTree(xyz)
    M        = centers.size(0)
    k_actual = min(k, len(xyz))
    cents_np = centers.cpu().numpy()
    _, idx   = tree.query(cents_np, k=k_actual, workers=-1)   # (M, k)
    idx_t    = torch.tensor(idx, dtype=torch.long)
    patches  = features[idx_t]                # (M, k, 16)
    sf_means = patches[:, :, 9].mean(dim=1)  # (M,)
    return patches, sf_means


def tokenize_patches(patches: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    """(M, k, 16) + centers (M, 3) → tokens (M, 35)."""
    mean16 = patches.mean(dim=1)                # (M, 16)
    std16  = patches.std(dim=1).clamp(min=1e-8) # (M, 16)
    return torch.cat([mean16, std16, centers], dim=-1)  # (M, 35)


def scalar_guided_mask(
    sf_means: torch.Tensor,
    base_ratio: float = 0.4,
    extra: float = 0.3,
    T: float = 0.5,
) -> torch.Tensor:
    """Bernoulli mask with higher probability for low-sf patches. Returns bool (M,)."""
    weights = F.softmax(-sf_means / (T + 1e-8), dim=0)
    p_mask  = (base_ratio + extra * weights).clamp(0.0, 0.95)
    return torch.bernoulli(p_mask).bool()


# ── Transformer building block ─────────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.n1   = nn.LayerNorm(d_model)
        self.n2   = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x    = self.n1(x + self.drop(a))
        x    = self.n2(x + self.drop(self.ff(x)))
        return x


# ── Encoder & Decoder ──────────────────────────────────────────────────────

class ScalarMAEEncoder(nn.Module):
    def __init__(self, token_dim: int = TOKEN_DIM, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, n_layers: int = N_ENC_LAYERS):
        super().__init__()
        self.proj   = nn.Linear(token_dim, d_model)
        self.cls    = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads)
                                     for _ in range(n_layers)])
        self.norm   = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, N_vis, token_dim) → (B, N_vis+1, d_model)"""
        B   = tokens.size(0)
        x   = self.proj(tokens)
        cls = self.cls.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)   # (B, N_vis+1, d_model)


class ScalarMAEDecoder(nn.Module):
    def __init__(self, d_model: int = D_MODEL, dec_dim: int = 64,
                 n_heads: int = N_HEADS, n_layers: int = N_DEC_LAYERS,
                 token_dim: int = TOKEN_DIM):
        super().__init__()
        self.proj      = nn.Linear(d_model, dec_dim)
        self.mask_tok  = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.blocks    = nn.ModuleList([TransformerBlock(dec_dim, n_heads, dropout=0.0)
                                        for _ in range(n_layers)])
        self.norm      = nn.LayerNorm(dec_dim)
        self.pred_head = nn.Linear(dec_dim, token_dim)

    def forward(self, enc_out: torch.Tensor, n_mask: int) -> torch.Tensor:
        """
        enc_out: (B, N_vis+1, d_model)
        n_mask : number of masked patches to predict
        Returns: (B, n_mask, token_dim)
        """
        B           = enc_out.size(0)
        x           = self.proj(enc_out)
        mask_tokens = self.mask_tok.expand(B, n_mask, -1)
        x           = torch.cat([x, mask_tokens], dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.pred_head(x[:, -n_mask:, :])   # (B, n_mask, token_dim)


# ── ScalarMAEModel ─────────────────────────────────────────────────────────

class ScalarMAEModel(nn.Module):
    """
    Full MAE model.
    pretrain_step: single forward + MAE loss (no gradients tracked externally).
    segment      : full cloud → per-point logits (N, NUM_CLASSES).
    """
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS,
                 n_enc_layers: int = N_ENC_LAYERS, n_dec_layers: int = N_DEC_LAYERS):
        super().__init__()
        self.encoder  = ScalarMAEEncoder(TOKEN_DIM, d_model, n_heads, n_enc_layers)
        self.decoder  = ScalarMAEDecoder(d_model, 64, n_heads, n_dec_layers, TOKEN_DIM)
        self.seg_head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ELU(),
            nn.Linear(64, 1),
        )
        self._d_model = d_model

    def pretrain_step(self, features: torch.Tensor) -> torch.Tensor:
        """MAE reconstruction loss for pre-training on a single cloud."""
        centers          = fps_centroids(features[:, :3], M_PATCHES)
        patches, sf_means = build_patches(features, centers, K_PATCH)
        tokens           = tokenize_patches(patches, centers)   # (M, 35)
        mask             = scalar_guided_mask(sf_means)
        if mask.sum() == 0:
            mask[0] = True
        vis_tok = tokens[~mask]
        if vis_tok.size(0) == 0:
            vis_tok = tokens[:1]
            mask    = torch.zeros(tokens.size(0), dtype=torch.bool)
            mask[0] = True
        enc_out = self.encoder(vis_tok.unsqueeze(0))                  # (1, N_vis+1, d)
        recon   = self.decoder(enc_out, n_mask=int(mask.sum()))       # (1, n_mask, 35)
        target  = tokens[mask]                                        # (n_mask, 35)
        return F.mse_loss(recon.squeeze(0), target)

    def segment(self, features: torch.Tensor) -> torch.Tensor:
        """
        Full cloud segmentation via nearest-patch-center interpolation.
        features: (N, 16) → logits (N, NUM_CLASSES)
        """
        centers          = fps_centroids(features[:, :3], M_PATCHES)
        patches, sf_means = build_patches(features, centers, K_PATCH)
        tokens           = tokenize_patches(patches, centers)
        enc_out          = self.encoder(tokens.unsqueeze(0))          # (1, M+1, d)
        patch_feats      = enc_out[0, 1:, :]                          # (M, d)  skip CLS

        xyz_pts  = features[:, :3].cpu()
        xyz_ctr  = centers.cpu()
        dists    = torch.cdist(xyz_pts, xyz_ctr)
        nn_idx   = dists.argmin(dim=1)                                # (N,)
        if patch_feats.device != features.device:
            patch_feats = patch_feats.to(features.device)
        point_feats = patch_feats[nn_idx]                             # (N, d)
        return self.seg_head(point_feats)                             # (N, NUM_CLASSES)


# ── EMA update ─────────────────────────────────────────────────────────────

def ema_update(teacher: nn.Module, student: nn.Module, decay: float = EMA_DECAY):
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.mul_(decay).add_(s_p.data, alpha=1.0 - decay)


# ── FixMatch ────────────────────────────────────────────────────────────────

def fixmatch_loss(student: nn.Module, unlabeled_feats: torch.Tensor,
                  pos_weight: torch.Tensor,
                  tau_crack: float = 0.80,
                  tau_normal: float = 0.95) -> torch.Tensor:
    """
    FixMatch por ponto para nuvens não-rotuladas.
    tau_crack < tau_normal: threshold assimétrico compensa desbalanceamento.
    Inspirado em: Sohn et al. NeurIPS 2020 (FixMatch) +
                  Zhang et al. NeurIPS 2021 (FlexMatch — threshold por classe).
    """
    from utils.augmentation import augment_cloud as _aug
    student.eval()
    with torch.no_grad():
        logits_weak = student.segment(unlabeled_feats).squeeze(-1)   # (N,)
        probs_weak  = torch.sigmoid(logits_weak)                     # (N,) ∈ [0,1]

    mask_crack  = probs_weak > tau_crack
    mask_normal = probs_weak < (1.0 - tau_normal)
    mask = mask_crack | mask_normal

    if mask.sum() == 0:
        return torch.tensor(0.0, device=unlabeled_feats.device)

    pseudo_labels = mask_crack[mask].float()

    student.train()
    feats_np = unlabeled_feats.cpu().numpy()
    feats_aug_np, _ = _aug(feats_np, rng=np.random.default_rng())
    feats_strong = torch.tensor(feats_aug_np, device=unlabeled_feats.device)

    min_n = min(feats_strong.size(0), mask.size(0))
    mask_aligned = mask[:min_n]
    pseudo_aligned = pseudo_labels[:mask_aligned.sum()]

    if mask_aligned.sum() == 0:
        return torch.tensor(0.0, device=unlabeled_feats.device)

    logits_strong = student.segment(feats_strong[:min_n]).squeeze(-1)
    confident_logits = logits_strong[mask_aligned]

    n_conf = confident_logits.size(0)
    n_pseudo = pseudo_aligned.size(0)
    n = min(n_conf, n_pseudo)
    if n == 0:
        return torch.tensor(0.0, device=unlabeled_feats.device)

    return binary_focal_loss(
        confident_logits[:n],
        pseudo_aligned[:n].to(unlabeled_feats.device),
        pos_weight
    )


# ── Pre-training ────────────────────────────────────────────────────────────

def pretrain(normal_clouds: list) -> ScalarMAEModel:
    model = ScalarMAEModel().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR_PRETRAIN, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=PRETRAIN_EPOCHS)

    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        np.random.shuffle(normal_clouds)
        for d in normal_clouds:
            x = torch.tensor(d['features'], dtype=torch.float32).to(DEVICE)
            opt.zero_grad()
            loss = model.pretrain_step(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if (epoch + 1) % 10 == 0:
            log.info(f"  Pretrain epoch {epoch+1}/{PRETRAIN_EPOCHS}  "
                     f"loss={epoch_loss/max(len(normal_clouds),1):.4f}")

    return model


# ── Fine-tuning (FixMatch) ──────────────────────────────────────────────────

def finetune(
    pretrained_model: ScalarMAEModel,
    labeled: list,
    normal_clouds: list,
    val_labeled: list,
) -> ScalarMAEModel:
    student = pretrained_model

    pos_weight = compute_pos_weight(
        [d for d in labeled if d['labels'] is not None], DEVICE
    )

    opt   = torch.optim.AdamW(student.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FINETUNE_EPOCHS)

    best_val, patience_cnt, best_state = np.inf, 0, None
    unlabeled_pool = normal_clouds

    for epoch in range(FINETUNE_EPOCHS):
        student.train()
        lam = min(LAMBDA_CONS_MAX, LAMBDA_CONS_MAX * epoch / max(LAMBDA_WARMUP, 1))

        epoch_loss = 0.0
        for d in labeled:
            x = torch.tensor(d['features'], dtype=torch.float32).to(DEVICE)
            y = torch.tensor(d['labels'],   dtype=torch.float32).to(DEVICE)
            opt.zero_grad()
            logits   = student.segment(x).squeeze(-1)   # (N,)
            sup_loss = binary_focal_loss(logits, y, pos_weight)

            # FixMatch on a random unlabeled cloud
            if unlabeled_pool and lam > 0:
                u_d = unlabeled_pool[np.random.randint(len(unlabeled_pool))]
                u_x = torch.tensor(u_d['features'], dtype=torch.float32).to(DEVICE)
                cons_loss = fixmatch_loss(student, u_x, pos_weight,
                                         tau_crack=0.80, tau_normal=0.95)
            else:
                cons_loss = torch.tensor(0.0, device=DEVICE)

            loss = sup_loss + lam * cons_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for d in val_labeled:
                if d['labels'] is None:
                    continue
                x = torch.tensor(d['features'], dtype=torch.float32).to(DEVICE)
                y = torch.tensor(d['labels'],   dtype=torch.float32).to(DEVICE)
                val_loss += binary_focal_loss(
                    student.segment(x).squeeze(-1), y, pos_weight).item()

        if val_loss < best_val:
            best_val      = val_loss
            best_state    = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    if best_state is not None:
        student.load_state_dict(best_state)
    return student


# ── LOCO ────────────────────────────────────────────────────────────────────

def run_loco(labeled: list, normals: list) -> list:
    results = []
    for i, test_d in enumerate(labeled):
        fname  = test_d['filename']
        y_test = test_d['labels']
        if y_test is None or (y_test == 1).sum() < 5:
            continue

        train_labeled = [d for d in labeled if d['filename'] != fname]
        n_val         = max(1, int(0.2 * len(train_labeled)))
        val_idx       = np.random.default_rng(42).choice(
                            len(train_labeled), n_val, replace=False)
        val_clouds    = [train_labeled[j] for j in val_idx]
        tr_clouds     = [train_labeled[j] for j in range(len(train_labeled))
                         if j not in set(val_idx.tolist())]

        t0          = time.time()
        pretrained  = pretrain(normals)
        finetuned   = finetune(pretrained, tr_clouds, normals, val_clouds)
        elapsed     = time.time() - t0

        finetuned.eval()
        with torch.no_grad():
            x     = torch.tensor(test_d['features'], dtype=torch.float32).to(DEVICE)
            probs = torch.sigmoid(finetuned.segment(x).squeeze(-1)).cpu().numpy()  # (N,)

        n_crack  = int((y_test == 1).sum())
        n_normal = int((y_test == 0).sum())

        auroc = float(roc_auc_score(y_test, probs))
        ap    = float(average_precision_score(y_test, probs))
        from sklearn.metrics import precision_recall_curve
        prec, rec, thr = precision_recall_curve(y_test, probs)
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_thr = float(thr[np.argmax(f1s[:-1])]) if len(thr) > 0 else 0.5
        preds = (probs >= best_thr).astype(np.int64)
        f1 = float(f1_score(y_test, preds, zero_division=0))

        log.info(
            f"[{i+1:02d}/{len(labeled)}] {fname:<35} "
            f"AUROC={auroc:.4f}  F1={f1:.4f}  AP={ap:.4f}  ({elapsed:.0f}s)"
        )

        row = {
            'filename' : fname,
            'n_crack'  : n_crack,
            'n_normal' : n_normal,
            'auroc'    : round(auroc, 4),
            'f1'       : round(f1,   4),
            'ap'       : round(ap,   4),
            'elapsed_s': round(elapsed, 1),
            # dados de visualização (filtrados antes do CSV)
            'xyz'      : test_d['features'][:, :3].astype(np.float32),
            'rgb_orig' : test_d['features'][:, 3:6].astype(np.float32),
            'preds'    : preds,
            'gt_labels': y_test,
        }
        results.append(row)
    return results


def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n{'='*60}\n  ScalarMAE LOCO — {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*60}")
    log.info(f"Device: {DEVICE}")

    all_data = load_folder(DATA_TRAIN_BIN)
    labeled  = [d for d in all_data if d.get('has_crack') and d['labels'] is not None]
    normals  = [d for d in all_data if not d.get('has_crack', False)]
    log.info(f"Avaria: {len(labeled)}  Normais: {len(normals)}")

    results = run_loco(labeled, normals)
    if not results:
        log.error("Sem resultados.")
        return

    aurocs = [r['auroc'] for r in results if np.isfinite(r['auroc'])]
    f1s    = [r['f1']    for r in results if np.isfinite(r['f1'])]
    aps    = [r['ap']    for r in results if np.isfinite(r['ap'])]

    print(f"\n  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  F1   : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AP   : {np.mean(aps):.4f} ± {np.std(aps):.4f}")

    # PLY — avarias preditas em vermelho, restante em cor original
    os.makedirs(PLY_DIR, exist_ok=True)
    for r in results:
        if 'xyz' in r:
            save_crack_ply(
                xyz=r['xyz'], rgb_orig=r['rgb_orig'],
                pred_labels=r['preds'],
                out_path=os.path.join(PLY_DIR,
                                      r['filename'].replace('.ply', '_smae.ply')),
                gt_labels=r.get('gt_labels'),
            )
    log.info(f"PLY salvos em: {PLY_DIR}")

    # Gráficos 2D de métricas
    plot_loco_metrics(results, VIS_DIR, 'ScalarMAE', ts)

    # CSV — filtra arrays numpy antes de escrever
    csv_results = [{k: v for k, v in r.items() if k not in _VIS_KEYS}
                   for r in results]
    loco_csv = f'{RESULTS_DIR}/loco_{ts}.csv'
    with open(loco_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    log.info(f"CSV: {loco_csv}")


if __name__ == '__main__':
    main()
