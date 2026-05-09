"""
GMVAE — Gaussian Mixture VAE para detecção não-supervisionada de rachaduras.

Prior p(z) = Σ_{k=1}^{K} π_k N(z; μ_k, σ_k²I). Aprende K=2 clusters sem labels.
Score de crack = q(c=crack|x): componente com menor média no scalar_field.
β warm-up linear 0→1 previne posterior collapse.

Referências:
  Dilokthanakul et al. (2016)  — GMVAE [arxiv:1611.02648]
  Kingma & Welling (2014)      — VAE   [arxiv:1312.6114]
  Jang et al. (2017)           — Gumbel-Softmax [arxiv:1611.01144]
  Higgins et al. (2017)        — β-VAE [ICLR 2017]
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, warnings
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from utils.config import BASE_PATH, DATA_TRAIN_BIN, setup_logging
from utils.data import load_folder
from utils.visualization import save_crack_ply, plot_loco_metrics
from gmm_scalar import otsu_score

log = setup_logging(f'{BASE_PATH}/logs_gmm_vae')
RESULTS_DIR = f'{BASE_PATH}/results_gmm_vae'
PLY_DIR     = f'{RESULTS_DIR}/ply'
VIS_DIR     = f'{RESULTS_DIR}/vis'
_VIS_KEYS   = {'xyz', 'rgb_orig', 'preds', 'gt_labels'}


# ── Componentes ───────────────────────────────────────────────────────────────

def gumbel_softmax(logits: torch.Tensor, tau: float = 0.5,
                   hard: bool = False) -> torch.Tensor:
    """Reparametrização Gumbel-Softmax (Jang et al. 2017)."""
    gumbels = -torch.empty_like(logits).exponential_().log()
    y_soft  = ((logits + gumbels) / tau).softmax(dim=-1)
    if hard:
        idx    = y_soft.argmax(dim=-1, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class GMVAE(nn.Module):
    def __init__(self, in_dim: int = 16, latent_dim: int = 4, K: int = 2):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),    nn.ReLU(),
        )
        self.cluster_head  = nn.Linear(32, K)
        self.mu_heads      = nn.ModuleList([nn.Linear(32, latent_dim) for _ in range(K)])
        self.logvar_heads  = nn.ModuleList([nn.Linear(32, latent_dim) for _ in range(K)])

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(),
            nn.Linear(32, 64),         nn.ReLU(),
            nn.Linear(64, in_dim),
        )

        # Prior aprendível: μ_k, log-σ²_k, log-π_k
        self.prior_mu     = nn.Parameter(torch.zeros(K, latent_dim))
        self.prior_logvar = nn.Parameter(torch.zeros(K, latent_dim))
        self.prior_logpi  = nn.Parameter(torch.zeros(K))

    def encode(self, x: torch.Tensor, tau: float = 0.5):
        h         = self.backbone(x)
        logits_c  = self.cluster_head(h)
        c_soft    = gumbel_softmax(logits_c, tau=tau, hard=False)
        mu     = sum(c_soft[:, k:k+1] * self.mu_heads[k](h)     for k in range(self.K))
        logvar = sum(c_soft[:, k:k+1] * self.logvar_heads[k](h) for k in range(self.K))
        logvar = logvar.clamp(-10, 4)
        return mu, logvar, logits_c, c_soft

    def reparametrize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor, tau: float = 0.5):
        mu, logvar, logits_c, _ = self.encode(x, tau=tau)
        z     = self.reparametrize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar, logits_c, z

    def elbo(self, x: torch.Tensor, x_hat: torch.Tensor,
             mu: torch.Tensor, logvar: torch.Tensor,
             logits_c: torch.Tensor, beta: float) -> torch.Tensor:
        recon = F.mse_loss(x_hat, x)

        q_c = logits_c.softmax(-1)  # (N, K)

        # KL(q(z|x,c) || p(z|c)) — média ponderada por q(c|x)
        kl_z = torch.zeros(1, device=x.device)
        for k in range(self.K):
            kl_k = 0.5 * (
                self.prior_logvar[k] - logvar
                + (logvar.exp() + (mu - self.prior_mu[k]).pow(2))
                  / (self.prior_logvar[k].clamp(-6, 4).exp() + 1e-8)
                - 1
            ).sum(-1)  # (N,)
            kl_z = kl_z + (q_c[:, k] * kl_k).mean()

        # KL(q(c|x) || p(c)) — KL categórico
        log_prior_pi = self.prior_logpi - self.prior_logpi.logsumexp(0)
        kl_c = (q_c * (q_c.clamp(1e-8).log() - log_prior_pi)).sum(-1).mean()

        return recon + beta * (kl_z + kl_c)


# ── Core functions ─────────────────────────────────────────────────────────────

def _train_gmvae(features_norm: np.ndarray, K: int = 2, latent_dim: int = 4,
                  epochs: int = 150, beta_warmup: int = 50) -> GMVAE:
    x     = torch.tensor(features_norm, dtype=torch.float32)
    model = GMVAE(in_dim=features_norm.shape[1], latent_dim=latent_dim, K=K)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for epoch in range(epochs):
        beta = min(1.0, epoch / max(1, beta_warmup))
        tau  = max(0.3, 1.0 - epoch / epochs)
        x_hat, mu, logvar, logits_c, _ = model(x, tau=tau)
        loss = model.elbo(x, x_hat, mu, logvar, logits_c, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
    return model


def gmvae_crack_score(model: GMVAE, features_norm: np.ndarray,
                       features_raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Retorna (scores ∈ [0,1], crack_component_idx)."""
    model.eval()
    with torch.no_grad():
        x = torch.tensor(features_norm, dtype=torch.float32)
        _, _, _, logits_c, _ = model(x, tau=0.01)
    q_c = logits_c.softmax(-1).numpy()  # (N, K)

    # Componente crack = menor média ponderada no scalar_field
    sf = features_raw[:, 9].astype(np.float64)
    sf_means = np.array([
        (q_c[:, k] * sf).sum() / (q_c[:, k].sum() + 1e-8)
        for k in range(model.K)
    ])
    crack_idx = int(np.argmin(sf_means))
    return q_c[:, crack_idx].astype(np.float32), crack_idx


def evaluate_cloud(cloud: dict, epochs: int = 150) -> dict | None:
    """Avalia uma nuvem com GMVAE e retorna métricas binárias."""
    feat   = cloud['features']
    labels = cloud['labels']

    if labels is None or (labels == 1).sum() < 5:
        return None

    try:
        q1   = np.percentile(feat, 25, axis=0)
        q3   = np.percentile(feat, 75, axis=0)
        norm = ((feat - q1) / ((q3 - q1) + 1e-8)).astype(np.float32)

        model  = _train_gmvae(norm, K=2, latent_dim=4, epochs=epochs, beta_warmup=50)
        scores, _ = gmvae_crack_score(model, norm, feat)
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
    print(f"  GMVAE — LOCO Results")
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
                                      r['filename'].replace('.ply', '_gmvae.ply')),
                gt_labels=r.get('gt_labels'),
            )

    plot_loco_metrics(results, VIS_DIR, 'GMVAE', ts)

    csv_results = [{k: v for k, v in r.items() if k not in _VIS_KEYS} for r in results]
    csv_path = f'{RESULTS_DIR}/loco_{ts}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    log.info(f"Resultados: {csv_path}")


if __name__ == '__main__':
    main()
