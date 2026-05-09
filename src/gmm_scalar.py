"""
GMMScalar — Detector não-supervisionado de rachaduras via GMM bimodal.

Por nuvem: ajusta GMM(n=2) no scalar_field → componente de menor média = crack.
Score por ponto = responsabilidade posterior do componente crack.
Fallback Otsu quando GMM colapsa (componentes com variância < 1e-6).

Referências:
  Valença et al. (Automation in Construction, 2022) — bimodalidade TLS em rachaduras
  Laefer et al. (J. Bridge Eng., 2022)              — GMM em resíduos de intensidade
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, warnings
from datetime import datetime

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from utils.config import BASE_PATH, DATA_TRAIN_BIN, setup_logging
from utils.data import load_folder
from utils.visualization import save_crack_ply, plot_loco_metrics

log = setup_logging(f'{BASE_PATH}/logs_gmm_scalar')
RESULTS_DIR = f'{BASE_PATH}/results_gmm_scalar'
PLY_DIR     = f'{RESULTS_DIR}/ply'
VIS_DIR     = f'{RESULTS_DIR}/vis'
_VIS_KEYS   = {'xyz', 'rgb_orig', 'preds', 'gt_labels'}


# ── Core functions ────────────────────────────────────────────────────────────

def fit_gmm(scalar_field: np.ndarray, random_state: int = 42):
    """
    Ajusta GMM(n=2) no scalar_field de uma nuvem.
    Retorna (gmm, lower_component_idx).
    lower_component_idx: índice do componente com menor média (= componente crack).
    """
    sf = scalar_field.reshape(-1, 1).astype(np.float64)
    # IQR normalization para estabilidade numérica
    q1, q3 = np.percentile(sf, [25, 75])
    iqr = q3 - q1 + 1e-8
    sf_norm = (sf - q1) / iqr

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gmm = GaussianMixture(n_components=2, covariance_type='full',
                              max_iter=200, random_state=random_state)
        gmm.fit(sf_norm)

    lower_idx = int(np.argmin(gmm.means_.ravel()))
    return gmm, lower_idx


def crack_score_from_gmm(gmm, scalar_field: np.ndarray,
                         lower_idx: int) -> np.ndarray:
    """
    Retorna score de crack por ponto ∈ [0, 1].
    Score = responsabilidade posterior do componente de menor média.
    """
    sf = scalar_field.reshape(-1, 1).astype(np.float64)
    q1, q3 = np.percentile(sf, [25, 75])
    iqr = q3 - q1 + 1e-8
    sf_norm = (sf - q1) / iqr

    resp = gmm.predict_proba(sf_norm)  # (N, 2)
    return resp[:, lower_idx].astype(np.float32)


def otsu_score(scalar_field: np.ndarray) -> np.ndarray:
    """
    Fallback: score monotônico via limiar de Otsu.
    Pontos abaixo do limiar recebem score proporcional ao desvio.
    """
    from skimage.filters import threshold_otsu
    sf = scalar_field.astype(np.float64)
    try:
        thr = threshold_otsu(sf)
    except Exception:
        thr = float(np.median(sf))
    score = np.clip((thr - sf) / (np.abs(thr) + 1e-8), 0, 1)
    return score.astype(np.float32)


def evaluate_cloud(cloud: dict) -> dict | None:
    """Avalia uma nuvem: ajusta GMM no scalar_field e computa métricas binárias."""
    feat   = cloud['features']          # (N, 16)
    labels = cloud['labels']            # (N,) int64  0=normal 1=crack
    sf     = feat[:, 9]                 # scalar_field é col 9

    if labels is None or (labels == 1).sum() < 5:
        return None

    try:
        gmm, lower_idx = fit_gmm(sf)
        vars_ = gmm.covariances_.ravel()
        if vars_.min() < 1e-6:
            raise ValueError("GMM colapsou")
        scores = crack_score_from_gmm(gmm, sf, lower_idx)
    except Exception:
        scores = otsu_score(sf)

    auroc = float(roc_auc_score(labels, scores))
    ap    = float(average_precision_score(labels, scores))

    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(labels, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])]) if len(thr) > 0 else 0.5
    preds = (scores >= best_thr).astype(np.int64)
    f1    = float(f1_score(labels, preds, zero_division=0))

    return {
        'filename'  : cloud.get('filename', '?'),
        'auroc'     : round(auroc, 4),
        'f1'        : round(f1, 4),
        'ap'        : round(ap, 4),
        'n_crack'   : int((labels == 1).sum()),
        'n_normal'  : int((labels == 0).sum()),
        'threshold' : round(best_thr, 4),
        # dados de visualização (filtrados antes de escrever CSV)
        'xyz'       : feat[:, :3].astype(np.float32),
        'rgb_orig'  : feat[:, 3:6].astype(np.float32),
        'preds'     : preds,
        'gt_labels' : labels,
    }


def run_loco(labeled: list, normals: list) -> list:
    """LOCO sobre nuvens avaria. Normals não usados (unsupervised)."""
    results = []
    for i, cloud in enumerate(labeled):
        log.info(f"Fold {i+1}/{len(labeled)} — {cloud.get('filename','?')}")
        r = evaluate_cloud(cloud)
        if r:
            results.append(r)
            log.info(f"  AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}  AP={r['ap']:.4f}")
    return results


def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')

    all_data = load_folder(DATA_TRAIN_BIN)
    labeled  = [d for d in all_data if d.get('has_crack') and d['labels'] is not None]
    normals  = [d for d in all_data if not d.get('has_crack', False)]

    log.info(f"Dataset: {len(labeled)} avaria | {len(normals)} normais")

    results = run_loco(labeled, normals)
    if not results:
        log.error("Sem resultados")
        return

    aurocs = [r['auroc'] for r in results]
    f1s    = [r['f1']    for r in results]
    aps    = [r['ap']    for r in results]

    print(f"\n{'='*60}")
    print(f"  GMMScalar — LOCO Results")
    print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AP:    {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"{'='*60}")

    # PLY — avarias preditas em vermelho, restante em cor original
    os.makedirs(PLY_DIR, exist_ok=True)
    for r in results:
        if 'xyz' in r:
            save_crack_ply(
                xyz=r['xyz'], rgb_orig=r['rgb_orig'],
                pred_labels=r['preds'],
                out_path=os.path.join(PLY_DIR,
                                      r['filename'].replace('.ply', '_gmm.ply')),
                gt_labels=r.get('gt_labels'),
            )
    log.info(f"PLY salvos em: {PLY_DIR}")

    # Gráficos 2D de métricas
    plot_loco_metrics(results, VIS_DIR, 'GMMScalar', ts)

    # CSV — filtra arrays numpy antes de escrever
    csv_results = [{k: v for k, v in r.items() if k not in _VIS_KEYS}
                   for r in results]
    csv_path = f'{RESULTS_DIR}/loco_{ts}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    log.info(f"Resultados: {csv_path}")


if __name__ == '__main__':
    main()
