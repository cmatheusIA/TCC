"""
GMMAutoencoder — Detecção não-supervisionada de rachaduras via AE + GMM.

Por nuvem: treina MLP autoencoder nas 16D features → GMM(K=2) no latente 4D.
Score = responsabilidade da componente com menor média no scalar_field (col 9).
Fallback Otsu quando GMM colapsa (variância < 1e-6).

Referências:
  Zong et al. (ICLR 2018) — DAGMM: Deep Autoencoding GMM for Anomaly Detection
  Hinton & Salakhutdinov (Science 2006) — autoencoders para redução dimensional
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from utils.config import BASE_PATH, DATA_TRAIN_BIN, setup_logging
from utils.data import load_folder
from utils.visualization import save_crack_ply, plot_loco_metrics
from gmm_scalar import otsu_score

log = setup_logging(f'{BASE_PATH}/logs_gmm_ae')
RESULTS_DIR = f'{BASE_PATH}/results_gmm_ae'
PLY_DIR     = f'{RESULTS_DIR}/ply'
VIS_DIR     = f'{RESULTS_DIR}/vis'
_VIS_KEYS   = {'xyz', 'rgb_orig', 'preds', 'gt_labels'}


# ── Modelo ────────────────────────────────────────────────────────────────────

class CrackAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 16, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, 16),    nn.ReLU(),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, 32),         nn.ReLU(),
            nn.Linear(32, in_dim),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.decoder(z), z


# ── Core functions ─────────────────────────────────────────────────────────────

def _train_autoencoder(features_norm: np.ndarray, latent_dim: int = 4,
                        epochs: int = 100) -> CrackAutoencoder:
    x = torch.tensor(features_norm, dtype=torch.float32)
    model = CrackAutoencoder(in_dim=features_norm.shape[1], latent_dim=latent_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for _ in range(epochs):
        x_hat, _ = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def fit_ae_gmm(features: np.ndarray, latent_dim: int = 4,
               epochs: int = 100) -> np.ndarray:
    """
    Treina autoencoder nas 16D features, ajusta GMM(K=2) no latente.
    Retorna score de crack ∈ [0,1] por ponto.
    """
    # IQR normalization por feature (igual ao gmm_scalar, mas 16D)
    q1  = np.percentile(features, 25, axis=0)
    q3  = np.percentile(features, 75, axis=0)
    iqr = (q3 - q1) + 1e-8
    features_norm = ((features - q1) / iqr).astype(np.float32)

    model = _train_autoencoder(features_norm, latent_dim=latent_dim, epochs=epochs)

    model.eval()
    with torch.no_grad():
        _, z = model(torch.tensor(features_norm, dtype=torch.float32))
    z_np = z.numpy()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        gmm = GaussianMixture(n_components=2, covariance_type='full',
                              n_init=10, random_state=42)
        gmm.fit(z_np)

    # Verifica colapso
    vars_ = gmm.covariances_.reshape(2, -1).max(axis=1)
    if vars_.min() < 1e-6:
        raise ValueError("GMM colapsou no latente")

    # Identifica componente crack: menor média no scalar_field (col 9)
    cluster_assigns = gmm.predict(z_np)
    sf = features[:, 9]
    sf_means = np.array([
        sf[cluster_assigns == k].mean() if (cluster_assigns == k).any() else np.inf
        for k in range(2)
    ])
    crack_idx = int(np.argmin(sf_means))

    scores = gmm.predict_proba(z_np)[:, crack_idx]
    return scores.astype(np.float32)


def evaluate_cloud(cloud: dict, ae_epochs: int = 100) -> dict | None:
    """Avalia uma nuvem: treina AE+GMM e computa métricas binárias."""
    feat   = cloud['features']   # (N, 16)
    labels = cloud['labels']     # (N,) int64

    if labels is None or (labels == 1).sum() < 5:
        return None

    try:
        scores = fit_ae_gmm(feat, latent_dim=4, epochs=ae_epochs)
    except Exception:
        scores = otsu_score(feat[:, 9])

    auroc = float(roc_auc_score(labels, scores))
    ap    = float(average_precision_score(labels, scores))

    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(labels, scores)
    f1s     = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])]) if len(thr) > 0 else 0.5
    preds   = (scores >= best_thr).astype(np.int64)
    f1      = float(f1_score(labels, preds, zero_division=0))

    return {
        'filename' : cloud.get('filename', '?'),
        'auroc'    : round(auroc, 4),
        'f1'       : round(f1,   4),
        'ap'       : round(ap,   4),
        'n_crack'  : int((labels == 1).sum()),
        'n_normal' : int((labels == 0).sum()),
        'threshold': round(best_thr, 4),
        'xyz'      : feat[:, :3].astype(np.float32),
        'rgb_orig' : feat[:, 3:6].astype(np.float32),
        'preds'    : preds,
        'gt_labels': labels,
    }


def run_loco(labeled: list, normals: list) -> list:
    results = []
    for i, cloud in enumerate(labeled):
        log.info(f"Fold {i+1}/{len(labeled)} — {cloud.get('filename','?')}")
        r = evaluate_cloud(cloud)
        if r:
            results.append(r)
            log.info(f"  AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}  AP={r['ap']:.4f}")
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')

    all_data = load_folder(DATA_TRAIN_BIN)
    labeled  = [d for d in all_data if d.get('has_crack') and d['labels'] is not None]
    normals  = [d for d in all_data if not d.get('has_crack', False)]

    log.info(f"Dataset: {len(labeled)} avaria | {len(normals)} normais")

    results = run_loco(labeled, normals)
    if not results:
        log.error("Sem resultados"); return

    aurocs = [r['auroc'] for r in results]
    f1s    = [r['f1']    for r in results]
    aps    = [r['ap']    for r in results]

    print(f"\n{'='*60}")
    print(f"  GMMAutoencoder — LOCO Results")
    print(f"  AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  F1:    {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  AP:    {np.mean(aps):.4f} ± {np.std(aps):.4f}")
    print(f"{'='*60}")

    os.makedirs(PLY_DIR, exist_ok=True)
    for r in results:
        if 'xyz' in r:
            save_crack_ply(
                xyz=r['xyz'], rgb_orig=r['rgb_orig'],
                pred_labels=r['preds'],
                out_path=os.path.join(PLY_DIR,
                                      r['filename'].replace('.ply', '_gmm_ae.ply')),
                gt_labels=r.get('gt_labels'),
            )

    plot_loco_metrics(results, VIS_DIR, 'GMMAutoencoder', ts)

    csv_results = [{k: v for k, v in r.items() if k not in _VIS_KEYS} for r in results]
    csv_path = f'{RESULTS_DIR}/loco_{ts}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    log.info(f"Resultados: {csv_path}")


if __name__ == '__main__':
    main()
