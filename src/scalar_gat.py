# ============================================================================
# ScalarGAT — Supervised GAT with scalar-field attention bias
# ============================================================================
# Veličković et al. (2018) — Graph Attention Networks
# Scalar bias: e_ij += α * (-|sf_i - sf_j| / σ_sf)
# Pontos com SF similar recebem maior atenção → reforça clusters de crack
# ============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, time, multiprocessing as mp
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from torch_cluster import knn_graph
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (roc_auc_score, f1_score, average_precision_score,
                             jaccard_score, confusion_matrix)

from scipy.spatial import cKDTree

from utils.config import (BASE_PATH, DATA_TRAIN_BIN, setup_logging,
                          POS_WEIGHT_DEFAULT, INPUT_DIM)
from utils.data import load_folder
from utils.visualization import save_crack_ply, plot_loco_metrics

log = setup_logging(f'{BASE_PATH}/logs_scalar_gat')

RESULTS_DIR = f'{BASE_PATH}/results_scalar_gat'
PLY_DIR     = f'{RESULTS_DIR}/ply'
VIS_DIR     = f'{RESULTS_DIR}/vis'
_VIS_KEYS   = {'xyz', 'rgb_orig', 'preds', 'gt_labels'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Hyperparameters ────────────────────────────────────────────────────────
K_GRAPH    = 8     # kNN graph connectivity (16→8: 2× menos arestas → ~2× mais rápido)
N_WORKERS  = 16    # folds em paralelo (14GB free / ~0.6GB por fold ≈ 23 max)
HEADS_1    = 4     # GAT layer 1 heads
DIM_1      = 64    # GAT layer 1 out per head  → 256 total (concat)
HEADS_2    = 4     # GAT layer 2 heads
DIM_2      = 128   # GAT layer 2 out per head → 128 total (mean)
LR         = 1e-3
EPOCHS     = 100
PATIENCE   = 20
DROPOUT    = 0.3
JITTER_STD = 0.001


# ── Loss ──────────────────────────────────────────────────────────────────

def compute_pos_weight(clouds: list, device: torch.device) -> torch.Tensor:
    """pos_weight = N_neg / N_pos para BCEWithLogitsLoss."""
    n_pos = sum(int((d['labels'] == 1).sum()) for d in clouds if d['labels'] is not None)
    n_neg = sum(int((d['labels'] == 0).sum()) for d in clouds if d['labels'] is not None)
    n_pos = max(n_pos, 1)
    w = n_neg / n_pos
    return torch.tensor([w], dtype=torch.float32, device=device)


def binary_focal_loss(logits: torch.Tensor, targets: torch.Tensor,
                      pos_weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss binária. logits (N,), targets (N,) float32.
    FL = -alpha_t * (1-p_t)^gamma * log(p_t)
    pos_weight incorpora o desbalanceamento de classes.
    """
    logits  = logits.float().squeeze(-1)
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction='none')
    p_t = torch.exp(-bce)
    return ((1 - p_t) ** gamma * bce).mean()


# ── ScalarGATConv ──────────────────────────────────────────────────────────

class ScalarGATConv(MessagePassing):
    """
    Single GAT convolution layer with scalar-field attention bias.
    Attention: e_ij = LeakyReLU(att^T [Whi||Whj]) + alpha_bias * (-|sf_i-sf_j|/σ)
    """
    def __init__(self, in_channels: int, out_channels: int,
                 heads: int = 4, concat: bool = True, dropout: float = 0.0):
        super().__init__(aggr='add', node_dim=0)
        self.heads        = heads
        self.out_channels = out_channels
        self.concat       = concat
        self.dropout_p    = dropout

        self.lin = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_channels))
        self.alpha_bias = nn.Parameter(torch.ones(1))
        nn.init.xavier_uniform_(self.att.view(heads, 2 * out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                sf: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        H = self.lin(x).view(N, self.heads, self.out_channels)
        sigma_sf = sf.float().std().clamp(min=1e-8)
        out = self.propagate(edge_index, H=H, sf=sf.float(), sigma_sf=sigma_sf,
                             size=(N, N))   # (N, heads, out)
        if self.concat:
            return out.reshape(N, self.heads * self.out_channels)
        return out.mean(dim=1)   # (N, out)

    def message(self, H_i: torch.Tensor, H_j: torch.Tensor,
                sf_i: torch.Tensor, sf_j: torch.Tensor,
                sigma_sf: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        cat   = torch.cat([H_i, H_j], dim=-1)              # (E, heads, 2*out)
        e     = F.leaky_relu((cat * self.att).sum(-1), 0.2) # (E, heads)
        sim   = -torch.abs(sf_i - sf_j) / sigma_sf          # (E,)
        e     = e + self.alpha_bias * sim.unsqueeze(-1)      # (E, heads)
        alpha = pyg_softmax(e, index)                        # (E, heads) — softmax per dst
        alpha = F.dropout(alpha, p=self.dropout_p, training=self.training)
        return H_j * alpha.unsqueeze(-1)                     # (E, heads, out)


# ── ScalarGATModel ─────────────────────────────────────────────────────────

class ScalarGATModel(nn.Module):
    """Two ScalarGATConv layers + MLP head → per-point crack logit."""
    def __init__(self, in_channels: int = INPUT_DIM, k: int = K_GRAPH):
        super().__init__()
        self.k = k
        mid = HEADS_1 * DIM_1   # 256

        self.conv1 = ScalarGATConv(in_channels, DIM_1, heads=HEADS_1, concat=True,  dropout=0.0)
        self.bn1   = nn.BatchNorm1d(mid)
        self.conv2 = ScalarGATConv(mid, DIM_2,         heads=HEADS_2, concat=False, dropout=0.0)
        self.bn2   = nn.BatchNorm1d(DIM_2)
        self.head  = nn.Sequential(
            nn.Linear(DIM_2, 64), nn.ELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, sf_col: torch.Tensor,
                edge_index: torch.Tensor = None) -> torch.Tensor:
        """
        x          : (N, INPUT_DIM) full feature vector
        sf_col     : (N, 1) or (N,) scalar_field column (normalizado internamente)
        edge_index : (2, N*k) pré-computado por _build_edge_index — se None, computa on-the-fly
        """
        sf      = sf_col.squeeze(-1).float()
        sf_norm = (sf - sf.median()) / (sf.std().clamp(min=1e-8))

        if edge_index is None:
            xyz        = x[:, :3]
            batch      = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            edge_index = knn_graph(xyz, k=self.k, batch=batch, loop=False)

        h = F.elu(self.bn1(self.conv1(x, edge_index, sf_norm)))
        h = F.elu(self.bn2(self.conv2(h, edge_index, sf_norm)))
        return self.head(h)   # (N, NUM_CLASSES)


# ── kNN pré-computado ──────────────────────────────────────────────────────

def _build_edge_index(xyz: np.ndarray, k: int) -> torch.Tensor:
    """
    Constrói edge_index (2, N*k) no CPU usando scipy.
    Mesma semântica que knn_graph(loop=False): edge (vizinho→ponto).
    ~50× mais rápido que knn_graph em GPU para nuvens de 5k–25k pontos.
    """
    tree          = cKDTree(xyz)
    _, idx        = tree.query(xyz, k=k + 1, workers=1)   # workers=1: evita leak de semáforo em spawn pool
    idx           = idx[:, 1:]                              # remove self-loop → (N, k)
    N             = len(xyz)
    targets       = np.repeat(np.arange(N, dtype=np.int64), k)
    sources       = idx.reshape(-1).astype(np.int64)
    return torch.tensor(np.stack([sources, targets], axis=0), dtype=torch.long)


# ── Augmentation ───────────────────────────────────────────────────────────

def augment_cloud(features: torch.Tensor, sf_mask_p: float = 0.1) -> torch.Tensor:
    """Jitter xyz, random normal flip, scalar dropout."""
    feat = features.clone()
    feat[:, :3] += torch.randn_like(feat[:, :3]) * JITTER_STD
    if torch.rand(1).item() < 0.5:
        flip_dim = torch.randint(0, 3, (1,)).item()
        feat[:, 6 + flip_dim] *= -1
    if torch.rand(1).item() < sf_mask_p:
        feat[:, 9] = 0.0
    return feat


# ── Training loop ──────────────────────────────────────────────────────────

LOG_EVERY = 10   # imprime progresso a cada N epochs

def _cache_clouds(clouds: list) -> list:
    """
    Pré-computa edge_index (scipy CPU) e move tensores para DEVICE.
    edge_index é computado dos XYZ originais e reutilizado entre epochs.
    """
    cache = []
    for d in clouds:
        xyz_np = d['features'][:, :3].astype(np.float32)
        ei     = _build_edge_index(xyz_np, K_GRAPH).to(DEVICE)
        x      = torch.tensor(d['features'], dtype=torch.float32).to(DEVICE)
        y      = torch.tensor(d['labels'],   dtype=torch.long).to(DEVICE)
        sf     = x[:, 9:10].clone()
        cache.append((x, y, sf, ei))
    return cache


def train_one_fold(train_clouds: list, val_clouds: list,
                   fold_label: str = '', verbose: bool = True) -> ScalarGATModel:
    """Treina ScalarGATModel com early-stop. Imprime progresso a cada LOG_EVERY epochs."""
    model = ScalarGATModel().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val_loss, patience_cnt, best_state = np.inf, 0, None
    best_epoch = 0
    scaler = GradScaler()

    pos_w = compute_pos_weight(train_clouds, DEVICE)

    # Pré-computar edge_index e tensores — feito UMA vez, reutilizado em todos os epochs
    if verbose:
        log.info("    cache: pré-computando edge_index...")
    t_cache = time.time()
    train_cache = _cache_clouds(train_clouds)
    val_cache   = _cache_clouds(val_clouds)
    if verbose:
        log.info(f"    cache pronto ({time.time()-t_cache:.1f}s) — "
                 f"treino={len(train_cache)} nuvens  val={len(val_cache)} nuvens")

    for epoch in range(EPOCHS):
        # ── Treino ───────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        perm = np.random.permutation(len(train_cache))
        for i in perm:
            x, y, sf, ei = train_cache[i]
            x_aug = augment_cloud(x)
            opt.zero_grad()
            with autocast():
                logits = model(x_aug, sf, edge_index=ei)
                loss   = binary_focal_loss(logits, y.float(), pos_w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            train_losses.append(loss.item())

        sched.step()

        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y, sf, ei in val_cache:
                with autocast():
                    logits = model(x, sf, edge_index=ei)
                    val_losses.append(binary_focal_loss(logits, y.float(), pos_w).item())

        avg_train = float(np.mean(train_losses)) if train_losses else float('nan')
        avg_val   = float(np.mean(val_losses))   if val_losses   else float('nan')
        cur_lr    = opt.param_groups[0]['lr']

        if verbose and ((epoch + 1) % LOG_EVERY == 0 or epoch == 0):
            log.info(
                f"    ep {epoch+1:03d}/{EPOCHS} | "
                f"train={avg_train:.4f}  val={avg_val:.4f} | "
                f"lr={cur_lr:.1e}  pat={patience_cnt}/{PATIENCE}"
            )

        # ── Early stop ────────────────────────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch    = epoch + 1
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt  = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                if verbose:
                    log.info(
                        f"    early stop ep {epoch+1}  "
                        f"(melhor: ep {best_epoch}, val={best_val_loss:.4f})"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── Worker de fold (top-level para ser picklável pelo multiprocessing) ────────

def _fold_worker(args: tuple):
    """Executa um fold completo num processo independente."""
    i, n_labeled, test_d, train_clouds, val_clouds, seed = args

    np.random.seed(seed)
    torch.manual_seed(seed)

    fname  = test_d['filename']
    y_test = test_d['labels']   # int64, 0=normal 1=crack

    n_crack = int((y_test == 1).sum())
    if y_test is None or n_crack < 5:
        print(f"[{i+1:02d}/{n_labeled}] {fname} — pulado (< 5 pontos crack)", flush=True)
        return None

    n_normal  = int((y_test == 0).sum())
    crack_pct = 100.0 * n_crack / len(y_test)

    t0    = time.time()
    model = train_one_fold(train_clouds, val_clouds, verbose=False)
    elapsed = time.time() - t0

    model.eval()
    with torch.no_grad():
        x  = torch.tensor(test_d['features'], dtype=torch.float32).to(DEVICE)
        sf = x[:, 9:10]
        ei = _build_edge_index(test_d['features'][:, :3], K_GRAPH).to(DEVICE)
        with autocast():
            logits = model(x, sf, edge_index=ei)   # (N, 1)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()   # (N,)

    auroc = float(roc_auc_score(y_test, probs))
    ap    = float(average_precision_score(y_test, probs))

    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_test, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])]) if len(thr) > 0 else 0.5
    preds = (probs >= best_thr).astype(np.int64)
    f1    = float(f1_score(y_test, preds, zero_division=0))

    flag = '  ★' if auroc > 0.930 else ''
    print(
        f"[{i+1:02d}/{n_labeled}] {fname:<35} "
        f"AUROC={auroc:.4f}  F1={f1:.4f}  AP={ap:.4f}  "
        f"crack={crack_pct:.1f}%  ({elapsed:.0f}s){flag}",
        flush=True,
    )

    return {
        'filename' : fname,
        'auroc'    : round(auroc, 4),
        'f1'       : round(f1,   4),
        'ap'       : round(ap,   4),
        'n_crack'  : n_crack,
        'n_normal' : n_normal,
        'elapsed_s': round(elapsed, 1),
        # dados de visualização (filtrados antes do CSV)
        'xyz'      : test_d['features'][:, :3].astype(np.float32),
        'rgb_orig' : test_d['features'][:, 3:6].astype(np.float32),
        'preds'    : preds,
        'gt_labels': y_test,
    }


# ── LOCO evaluation ──────────────────────────────────────────────────────────

def run_loco(labeled: list, normals: list, n_workers: int = N_WORKERS) -> list:
    n_labeled = len(labeled)

    # Prepara args de cada fold
    fold_args = []
    for i, test_d in enumerate(labeled):
        train_labeled = [d for d in labeled if d['filename'] != test_d['filename']]
        n_val   = max(1, int(0.2 * len(train_labeled)))
        val_idx = np.random.default_rng(42).choice(len(train_labeled), n_val, replace=False)
        val_set   = [train_labeled[j] for j in val_idx]
        train_set = [train_labeled[j] for j in range(len(train_labeled))
                     if j not in set(val_idx.tolist())]
        train_set += normals
        fold_args.append((i, n_labeled, test_d, train_set, val_set, i * 42))

    log.info(f"\n{len(fold_args)} folds  |  {n_workers} workers em paralelo")

    ctx = mp.get_context('spawn')
    with ctx.Pool(processes=n_workers) as pool:
        raw = pool.map(_fold_worker, fold_args)

    results = [r for r in raw if r is not None]
    return results


def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  ScalarGAT — LOCO Evaluation")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}  |  device={DEVICE}")
    print(f"  k={K_GRAPH}  heads={HEADS_1}×{DIM_1}/{HEADS_2}×{DIM_2}  "
          f"epochs={EPOCHS}  patience={PATIENCE}  lr={LR}")
    print(f"  workers={N_WORKERS}  (paralelo)")
    print(f"{'='*65}")

    all_data = load_folder(DATA_TRAIN_BIN)
    labeled  = [d for d in all_data if d.get('has_crack') and d['labels'] is not None]
    normals  = [d for d in all_data if not d.get('has_crack', False)]

    log.info(f"Dataset: {len(labeled)} avaria | {len(normals)} normais")
    for d in labeled:
        y   = d['labels']
        pct = 100.0 * (y == 1).sum() / len(y)
        log.info(f"    {d['filename']:<40} crack={pct:.1f}%")

    if not labeled:
        log.error("Nenhuma nuvem com label encontrada.")
        return

    results = run_loco(labeled, normals, n_workers=N_WORKERS)
    if not results:
        log.error("Sem resultados.")
        return

    aurocs = [r['auroc'] for r in results if np.isfinite(r['auroc'])]
    f1s    = [r['f1']    for r in results if np.isfinite(r['f1'])]
    aps    = [r['ap']    for r in results if np.isfinite(r['ap'])]

    results_sorted = sorted(results, key=lambda r: r['auroc'])
    print(f"\n{'='*70}")
    print("  ScalarGAT — Binary LOCO Results")
    print(f"  {'Arquivo':<38} {'AUROC':>6} {'F1':>6} {'AP':>6} {'crack%':>7}")
    print(f"  {'─'*65}")
    for r in results_sorted:
        pct  = 100.0 * r['n_crack'] / (r['n_crack'] + r['n_normal'])
        flag = ' ★' if r['auroc'] > 0.930 else ''
        print(f"  {r['filename']:<38} {r['auroc']:>6.4f} {r['f1']:>6.4f} "
              f"{r['ap']:>6.4f} {pct:>6.1f}%{flag}")
    print(f"  {'─'*65}")
    print(f"  {'média':<38} {np.mean(aurocs):>6.4f} {np.mean(f1s):>6.4f} {np.mean(aps):>6.4f}")
    print(f"  {'std':<38} {np.std(aurocs):>6.4f} {np.std(f1s):>6.4f} {np.std(aps):>6.4f}")
    print(f"{'='*70}")

    # PLY — avarias preditas em vermelho, restante em cor original
    os.makedirs(PLY_DIR, exist_ok=True)
    for r in results:
        if 'xyz' in r:
            save_crack_ply(
                xyz=r['xyz'], rgb_orig=r['rgb_orig'],
                pred_labels=r['preds'],
                out_path=os.path.join(PLY_DIR,
                                      r['filename'].replace('.ply', '_sgat.ply')),
                gt_labels=r.get('gt_labels'),
            )
    log.info(f"PLY salvos em: {PLY_DIR}")

    # Gráficos 2D de métricas
    plot_loco_metrics(results, VIS_DIR, 'ScalarGAT', ts)

    # CSV — filtra arrays numpy antes de escrever
    csv_results = [{k: v for k, v in r.items() if k not in _VIS_KEYS}
                   for r in results]
    loco_csv = f'{RESULTS_DIR}/loco_{ts}.csv'
    with open(loco_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    log.info(f"\nCSV: {loco_csv}")


if __name__ == '__main__':
    main()
