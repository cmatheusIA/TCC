# ============================================================================
# ScalarMemory — Unsupervised crack detection via memory bank + Mahalanobis
# ============================================================================
# PointCore-inspired (Zhou et al., arxiv 2403.01804):
#   1. Build memory bank from normal cloud 8D local features (CoreSet sampling)
#   2. Global threshold = percentile_99 of normal cloud anomaly scores
#   3. At inference: Mahalanobis distance to m=3 nearest bank entries
# ============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, time
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.linalg import inv
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from utils.config import BASE_PATH, DATA_TRAIN, setup_logging
from utils.data import load_folder
from utils.scalar_features import extract_local_sf_features

log = setup_logging(f'{BASE_PATH}/logs_scalar_memory')

RESULTS_DIR  = f'{BASE_PATH}/results_scalar_memory'
K_NEIGHBORS  = 32    # local neighborhood size
BANK_SIZE    = 2048  # CoreSet representatives
M_NEAREST    = 3     # nearest bank entries for score
THR_PCTL     = 99    # percentile on normal cloud scores for global threshold


# ── CoreSet sampling ─────────────────────────────────────────────────────────

def coreset_sampling(features: np.ndarray, K: int) -> np.ndarray:
    """Greedy furthest-point sampling for representative coverage."""
    N = len(features)
    K = min(K, N)
    chosen = [np.random.default_rng(42).integers(N)]
    min_dists = np.full(N, np.inf, dtype=np.float64)
    feats_f64 = features.astype(np.float64)
    for _ in range(K - 1):
        c = feats_f64[chosen[-1]]
        dists = np.linalg.norm(feats_f64 - c, axis=1)
        min_dists = np.minimum(min_dists, dists)
        chosen.append(int(np.argmax(min_dists)))
    return np.array(chosen, dtype=np.int64)


# ── Memory bank construction ─────────────────────────────────────────────────

def build_memory_bank(
    normal_clouds: list,
    K: int = BANK_SIZE,
    k: int = K_NEIGHBORS,
) -> tuple:
    """
    Extract 8D local features from all normal clouds, CoreSet-sample K
    representatives, compute global inverse covariance.

    Returns (bank [K, 8], cov_inv [8, 8]).
    """
    all_feats = []
    for d in normal_clouds:
        feat8 = extract_local_sf_features(d['features'], k=k)
        all_feats.append(feat8)
    all_feats = np.concatenate(all_feats, axis=0)

    idx = coreset_sampling(all_feats, K)
    bank = all_feats[idx].astype(np.float64)

    # Regularized covariance (from all normal features, not just bank)
    cov = np.cov(all_feats.astype(np.float64).T) + np.eye(8) * 1e-6
    try:
        cov_inv = inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.eye(8)

    return bank.astype(np.float32), cov_inv.astype(np.float32)


# ── Anomaly scoring ──────────────────────────────────────────────────────────

def compute_anomaly_scores(
    cloud: dict,
    bank: np.ndarray,
    cov_inv: np.ndarray,
    m: int = M_NEAREST,
    k: int = K_NEIGHBORS,
) -> np.ndarray:
    """
    Mahalanobis distance from each point's 8D feature to the m nearest bank entries.
    score(p) = min distance over m nearest bank representatives.
    """
    feat8 = extract_local_sf_features(cloud['features'], k=k).astype(np.float64)
    bank64 = bank.astype(np.float64)
    cov64  = cov_inv.astype(np.float64)

    # Find m nearest bank entries per point (Euclidean pre-filter)
    tree = cKDTree(bank64)
    m_actual = min(m, len(bank64))
    _, nn_idx = tree.query(feat8, k=m_actual, workers=-1)   # (N, m)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, np.newaxis]

    N = len(feat8)
    scores = np.full(N, np.inf, dtype=np.float64)
    for j in range(m_actual):
        diff = feat8 - bank64[nn_idx[:, j]]                  # (N, 8)
        mah  = np.einsum('ni,ij,nj->n', diff, cov64, diff)   # (N,)
        scores = np.minimum(scores, mah)

    return np.sqrt(np.clip(scores, 0, None)).astype(np.float32)


# ── Global threshold calibration ─────────────────────────────────────────────

def calibrate_threshold(
    normal_clouds: list,
    bank: np.ndarray,
    cov_inv: np.ndarray,
    percentile: float = THR_PCTL,
    k: int = K_NEIGHBORS,
) -> float:
    """Compute global threshold as percentile_99 of all normal cloud scores."""
    all_scores = []
    for d in normal_clouds:
        scores = compute_anomaly_scores(d, bank, cov_inv, k=k)
        all_scores.append(scores)
    all_scores = np.concatenate(all_scores)
    return float(np.percentile(all_scores, percentile))


# ── LOCO evaluation ──────────────────────────────────────────────────────────

def run_loco(labeled: list, normals: list, K: int = BANK_SIZE) -> list:
    """
    Leave-One-Cloud-Out: bank is built ONCE from truly normal clouds only.
    Including avaria clouds in the bank contaminates the normal distribution
    (crack points in avaria ↔ crack features in bank → low Mahalanobis for crack
    points in test cloud → inverted scoring). Bank is fold-invariant so we
    build it once for efficiency.
    """
    log.info(f"Building memory bank from {len(normals)} normal clouds...")
    t_bank = time.time()
    bank, cov_inv = build_memory_bank(normals, K=K)
    threshold     = calibrate_threshold(normals, bank, cov_inv)
    log.info(f"Bank built in {time.time()-t_bank:.1f}s  threshold={threshold:.4f}")

    results = []
    for i, test_d in enumerate(labeled):
        fname = test_d['filename']
        y_test = test_d['labels']

        if y_test is None or y_test.sum() < 5:
            log.info(f"[{i+1}/{len(labeled)}] {fname}: sem crack — skip")
            continue

        t0 = time.time()
        scores  = compute_anomaly_scores(test_d, bank, cov_inv)
        elapsed = time.time() - t0

        try:
            auroc = float(roc_auc_score(y_test, scores))
            ap    = float(average_precision_score(y_test, scores))
        except Exception:
            auroc = ap = float('nan')

        # F1 at global threshold
        y_pred = (scores >= threshold).astype(np.int32)
        f1_global = float(f1_score(y_test, y_pred, zero_division=0))

        # F1 at optimal threshold (for analysis)
        best_f1, best_thr = 0.0, threshold
        for pct in range(50, 100):
            thr = float(np.percentile(scores, pct))
            f   = float(f1_score(y_test, (scores >= thr).astype(int), zero_division=0))
            if f > best_f1:
                best_f1, best_thr = f, thr

        n_crack  = int(y_test.sum())
        n_normal = int((y_test == 0).sum())
        log.info(
            f"[{i+1:02d}/{len(labeled)}] {fname:<35} "
            f"AUROC={auroc:.4f}  F1_global={f1_global:.4f}  F1_opt={best_f1:.4f}  "
            f"AP={ap:.4f}  ({elapsed:.1f}s)"
        )
        results.append({
            'filename': fname, 'n_crack': n_crack, 'n_normal': n_normal,
            'auroc': round(auroc, 4), 'f1': round(f1_global, 4),
            'f1_opt': round(best_f1, 4), 'ap': round(ap, 4),
            'thr': round(best_thr, 4),
        })

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"\n{'='*60}\n  ScalarMemory LOCO — {datetime.now():%Y-%m-%d %H:%M:%S}\n{'='*60}")

    all_data = load_folder(DATA_TRAIN)
    labeled  = [d for d in all_data if d.get('has_crack') and d['labels'] is not None]
    normals  = [d for d in all_data if not d.get('has_crack', False)]
    log.info(f"Avaria: {len(labeled)}  Normais: {len(normals)}")

    results = run_loco(labeled, normals)

    if not results:
        log.error("Sem resultados — abortando.")
        return

    aurocs = [r['auroc'] for r in results if np.isfinite(r['auroc'])]
    f1s    = [r['f1']    for r in results if np.isfinite(r['f1'])]
    f1opts = [r['f1_opt'] for r in results if np.isfinite(r['f1_opt'])]
    aps    = [r['ap']    for r in results if np.isfinite(r['ap'])]

    print(f"\n{'─'*60}")
    print(f"  AUROC  : {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  F1     : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}  (global thr)")
    print(f"  F1_opt : {np.mean(f1opts):.4f} ± {np.std(f1opts):.4f}  (oracle thr)")
    print(f"  AP     : {np.mean(aps):.4f} ± {np.std(aps):.4f}")

    per_csv = f'{RESULTS_DIR}/per_cloud_{ts}.csv'
    with open(per_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    global_csv = f'{RESULTS_DIR}/global_{ts}.csv'
    with open(global_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'auroc_mean', 'auroc_std', 'f1_mean', 'f1_std', 'ap_mean'])
        writer.writerow(['scalar_memory',
                         round(np.mean(aurocs), 4), round(np.std(aurocs), 4),
                         round(np.mean(f1s),    4), round(np.std(f1s),    4),
                         round(np.mean(aps),    4)])

    log.info(f"Resultados: {per_csv}")
    log.info(f"Global:     {global_csv}")


if __name__ == '__main__':
    main()
