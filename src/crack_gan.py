"""
DGCNN-BiGAN — Detector não-supervisionado de rachaduras.

Arquitetura:
  Discriminador D: 3× DynamicEdgeConv(k=20) → global_max_pool → MLP(spectral_norm)
  Gerador G:       z(128D) → MLP residual → (K=64, 16D)
  Encoder E:       3× DynamicEdgeConv(k=20) → global_max_pool → z_hat(128D)  [BiGAN]

Treinamento:
  WGAN-GP (Gulrajani et al. NeurIPS 2017) com gradient penalty λ=10
  BiGAN: D discrimina (x_real, E(x_real)) vs (G(z), z)
  StyleGAN-ADA: probabilidade de augmentation adaptativa para datasets pequenos

Referências:
  Wang et al. ACM TOG 2019 arXiv:1801.07829 (DGCNN / EdgeConv)
  Donahue et al. ICLR 2017 (BiGAN — encoder treinado conjuntamente)
  Gulrajani et al. NeurIPS 2017 (WGAN-GP)
  Karras et al. NeurIPS 2020 arXiv:2006.06676 (StyleGAN-ADA)
  Liu et al. 2024 arXiv:2408.16201 (Uni-3DAD)
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DynamicEdgeConv, global_max_pool
from torch.nn.utils import spectral_norm
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score

from utils.config import BASE_PATH, DATA_TRAIN_BIN, setup_logging
from utils.data import load_folder
from utils.augmentation import augment_cloud
from utils.visualization import save_crack_ply, plot_loco_metrics

log = setup_logging(f'{BASE_PATH}/logs_crack_gan')
RESULTS_DIR = f'{BASE_PATH}/results_crack_gan'
PLY_DIR     = f'{RESULTS_DIR}/ply'
VIS_DIR     = f'{RESULTS_DIR}/vis'
_VIS_KEYS   = {'xyz', 'rgb_orig', 'preds', 'gt_labels'}
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Hiperparâmetros ───────────────────────────────────────────────────────────
K_PATCH     = 64    # pontos por patch
K_EDGE      = 20    # k para DynamicEdgeConv
Z_DIM       = 128   # dimensão do espaço latente
D_IN        = 16    # dimensão das features de entrada
GP_LAMBDA   = 10    # WGAN-GP gradient penalty
LR_G        = 1e-4
LR_D        = 1e-4
N_CRITIC    = 5     # passos do discriminador por passo do gerador
PRETRAIN_EP = 200   # epochs de treino GAN
ADA_TARGET  = 0.6   # alvo para sinal do discriminador (StyleGAN-ADA)
PATCH_STRIDE = K_PATCH // 2   # sobreposição de patches na inferência


# ── Módulos ────────────────────────────────────────────────────────────────────

class _EdgeConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=K_EDGE):
        super().__init__()
        self.conv = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(2 * in_ch, out_ch), nn.BatchNorm1d(out_ch), nn.ReLU(),
                nn.Linear(out_ch, out_ch),    nn.BatchNorm1d(out_ch), nn.ReLU(),
            ), k=k, aggr='max')

    def forward(self, x):
        # x: (N, C) — batch implícito de 1 nuvem
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        return self.conv(x, batch)


class DGCNNEncoder(nn.Module):
    """EdgeConv encoder: patch (K, D_IN) → z (Z_DIM,)."""

    def __init__(self, in_channels=D_IN, z_dim=Z_DIM, k=K_EDGE):
        super().__init__()
        self.ec1 = _EdgeConvBlock(in_channels, 64,  k=k)
        self.ec2 = _EdgeConvBlock(64,          128, k=k)
        self.ec3 = _EdgeConvBlock(128,         256, k=k)
        self.mlp = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, z_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (K, D_IN) → z: (Z_DIM,)"""
        h = self.ec3(self.ec2(self.ec1(x)))           # (K, 256)
        batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        g = global_max_pool(h, batch)                  # (1, 256)
        return self.mlp(g).squeeze(0)                  # (Z_DIM,)

    def intermediate(self, x: torch.Tensor) -> tuple:
        """Retorna features intermediárias para feature matching."""
        h1 = self.ec1(x)
        h2 = self.ec2(h1)
        h3 = self.ec3(h2)
        batch = torch.zeros(h3.size(0), dtype=torch.long, device=h3.device)
        g = global_max_pool(h3, batch).squeeze(0)
        z = self.mlp(g)
        return z, g   # (Z_DIM,), (256,)


class Generator(nn.Module):
    """MLP residual: z (Z_DIM,) → patch (out_points, out_channels)."""

    def __init__(self, z_dim=Z_DIM, out_points=K_PATCH, out_channels=D_IN):
        super().__init__()
        self.out_points   = out_points
        self.out_channels = out_channels
        self.fc1 = nn.Linear(z_dim,  256)
        self.fc2 = nn.Linear(256,    512)
        self.fc3 = nn.Linear(512,    512)
        self.fc4 = nn.Linear(512,    out_points * out_channels)
        self.res = nn.Linear(z_dim, out_points * out_channels)  # residual direto
        self.bn1 = nn.LayerNorm(256)   # LayerNorm works for single-sample inference
        self.bn2 = nn.LayerNorm(512)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (Z_DIM,) → (out_points, out_channels)"""
        h = F.relu(self.bn1(self.fc1(z)))
        h = F.relu(self.bn2(self.fc2(h)))
        h = F.relu(self.fc3(h))
        out = self.fc4(h) + self.res(z)   # residual connection
        return out.view(self.out_points, self.out_channels)


class Discriminator(nn.Module):
    """
    BiGAN discriminator: recebe (x, z) como par e retorna escalar WGAN.
    D(x, z): x → DGCNNEncoder → g_x; concat(g_x, z) → MLP(spectral_norm) → ℝ
    """

    def __init__(self, in_channels=D_IN, z_dim=Z_DIM, k=K_EDGE):
        super().__init__()
        self.enc = DGCNNEncoder(in_channels, z_dim, k=k)
        self.mlp = nn.Sequential(
            spectral_norm(nn.Linear(256 + z_dim, 256)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 128)),          nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 1)),
        )

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """x: (K, D_IN), z: (Z_DIM,) → scalar (1,)"""
        z_hat, g_x = self.enc.intermediate(x)
        pair = torch.cat([g_x, z], dim=0)   # (256 + Z_DIM,)
        return self.mlp(pair)               # (1,)

    def features(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Activations intermediárias para feature matching."""
        _, g_x = self.enc.intermediate(x)
        pair = torch.cat([g_x, z], dim=0)
        h = self.mlp[1](self.mlp[0](pair))   # após 1ª camada
        return h


# ── Funções de treinamento ─────────────────────────────────────────────────────

def gradient_penalty(D: Discriminator, x_real: torch.Tensor, x_fake: torch.Tensor,
                     z_real: torch.Tensor, z_fake: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """WGAN-GP gradient penalty (Gulrajani et al. 2017)."""
    alpha = torch.rand(1, device=device)
    x_hat = (alpha * x_real + (1 - alpha) * x_fake).requires_grad_(True)
    z_hat = (alpha * z_real + (1 - alpha) * z_fake).requires_grad_(True)
    d_hat = D(x_hat, z_hat)
    grads = torch.autograd.grad(
        outputs=d_hat, inputs=[x_hat, z_hat],
        grad_outputs=torch.ones_like(d_hat),
        create_graph=True, retain_graph=True)[0]
    gp = ((grads.norm(2) - 1) ** 2).mean()
    return gp


def ada_update(p_aug: float, sign_real: float, target: float = ADA_TARGET,
               step: float = 0.01) -> float:
    """StyleGAN-ADA: ajusta p_aug para manter E[sign(D_real)] ≈ target."""
    if sign_real > target:
        return min(p_aug + step, 1.0)
    return max(p_aug - step, 0.0)


def apply_ada_augment(patch: torch.Tensor, p: float,
                      rng: np.random.Generator) -> torch.Tensor:
    """Augmentações diferenciáveis para ADA (NÃO perturba col 9 = scalar_field)."""
    if rng.random() > p:
        return patch
    feat = patch.cpu().numpy().copy()
    feat_aug, _ = augment_cloud(
        feat, labels=None,
        rotate_z=True, jitter_std=0.01, scale_range=(0.9, 1.1),
        flip_prob=0.3, dropout_max=0.0, feat_jitter_sf=0.0,
        rng=rng)
    if len(feat_aug) < K_PATCH:
        pad = np.tile(feat_aug[:1], (K_PATCH - len(feat_aug), 1))
        feat_aug = np.vstack([feat_aug, pad])
    elif len(feat_aug) > K_PATCH:
        feat_aug = feat_aug[:K_PATCH]
    return torch.tensor(feat_aug, dtype=torch.float32, device=patch.device)


# ── Extração de patches ────────────────────────────────────────────────────────

def extract_patches(features: np.ndarray, k: int = K_PATCH,
                    stride: int = PATCH_STRIDE) -> list:
    """
    Extrai patches de k pontos via FPS com passo stride.
    Retorna lista de arrays (k, 16).
    """
    from scipy.spatial import cKDTree
    N = len(features)
    if N < k:
        pad = np.tile(features[:1], (k - N, 1))
        return [np.vstack([features, pad]).astype(np.float32)]

    xyz = features[:, :3]
    tree = cKDTree(xyz)
    centers = []
    remaining = list(range(N))
    rng = np.random.default_rng(42)
    while len(remaining) >= stride:
        idx = rng.choice(remaining)
        centers.append(idx)
        _, nn_idx = tree.query(xyz[idx], k=stride)
        remaining = [i for i in remaining if i not in set(nn_idx.tolist())]

    patches = []
    for c in centers:
        _, nn_idx = tree.query(xyz[c], k=k)
        patches.append(features[nn_idx].astype(np.float32))
    return patches


# ── Anomaly score ──────────────────────────────────────────────────────────────

def anomaly_score(patch: torch.Tensor, E: DGCNNEncoder, G: Generator,
                  D: Discriminator, kappa: float = 0.1) -> float:
    """
    A(x) = ||x - G(E(x))||²_F  +  κ · ||f_D(x,E(x)) - f_D(G(E(x)),E(x))||²
    """
    E.eval(); G.eval(); D.eval()
    with torch.no_grad():
        z_hat = E(patch)
        x_rec = G(z_hat)
        rec_loss = ((patch - x_rec) ** 2).mean().item()
        feat_real = D.features(patch, z_hat)
        feat_rec  = D.features(x_rec, z_hat)
        feat_loss = ((feat_real - feat_rec) ** 2).mean().item()
    return rec_loss + kappa * feat_loss


# ── Training loop ──────────────────────────────────────────────────────────────

def train_bigan(normal_clouds: list, n_epochs: int = PRETRAIN_EP,
                device: torch.device = DEVICE) -> tuple:
    """
    Treina DGCNN-BiGAN em nuvens normais.
    Retorna (E, G, D) treinados.
    """
    E = DGCNNEncoder().to(device)
    G = Generator().to(device)
    D = Discriminator().to(device)

    opt_G = torch.optim.Adam(list(G.parameters()) + list(E.parameters()),
                              lr=LR_G, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=LR_D, betas=(0.0, 0.9))

    all_patches = []
    for cloud in normal_clouds:
        patches = extract_patches(cloud['features'])
        all_patches.extend(patches)

    log.info(f"  BiGAN treino: {len(all_patches)} patches de {len(normal_clouds)} nuvens normais")

    rng = np.random.default_rng(42)
    p_aug = 0.0

    for epoch in range(1, n_epochs + 1):
        rng.shuffle(all_patches)
        d_losses, g_losses = [], []

        for patch_np in all_patches:
            x_real = torch.tensor(patch_np, device=device)
            x_real = apply_ada_augment(x_real, p_aug, rng)

            # ── Discriminador (N_CRITIC passos) ──────────────────────────────
            for _ in range(N_CRITIC):
                z_noise = torch.randn(Z_DIM, device=device)
                with torch.no_grad():
                    x_fake = G(z_noise)
                    z_real = E(x_real)

                D.zero_grad()
                loss_real = -D(x_real, z_real).mean()
                loss_fake =  D(x_fake, z_noise).mean()
                gp = gradient_penalty(D, x_real, x_fake, z_real, z_noise, device)
                loss_D = loss_real + loss_fake + GP_LAMBDA * gp
                loss_D.backward()
                opt_D.step()
                d_losses.append(loss_D.item())

            # ── ADA update ────────────────────────────────────────────────────
            with torch.no_grad():
                sign_real = float(torch.sign(D(x_real, E(x_real))).mean().item())
            p_aug = ada_update(p_aug, sign_real)

            # ── Gerador + Encoder (BiGAN) ─────────────────────────────────────
            G.zero_grad(); E.zero_grad()
            z_noise = torch.randn(Z_DIM, device=device)
            x_fake  = G(z_noise)
            z_real  = E(x_real)
            loss_GE = -D(x_real, z_real).mean() + D(x_fake, z_noise).mean()
            loss_GE.backward()
            opt_G.step()
            g_losses.append(loss_GE.item())

        if epoch % 50 == 0:
            log.info(f"  Epoch {epoch}/{n_epochs} | D={np.mean(d_losses):.4f} "
                     f"G={np.mean(g_losses):.4f} | p_aug={p_aug:.3f}")

    return E, G, D


def evaluate_cloud_gan(cloud: dict, E: DGCNNEncoder, G: Generator,
                       D: Discriminator, device: torch.device) -> dict | None:
    """Avalia uma nuvem com DGCNN-BiGAN: score por patch → score por ponto → métricas."""
    features = cloud['features']   # (N, 16)
    labels   = cloud['labels']     # (N,) int64

    if labels is None or (labels == 1).sum() < 5:
        return None

    patches = extract_patches(features)
    N = len(features)
    scores_sum   = np.zeros(N, dtype=np.float32)
    scores_count = np.zeros(N, dtype=np.float32)

    from scipy.spatial import cKDTree
    tree = cKDTree(features[:, :3])

    for patch_np in patches:
        x = torch.tensor(patch_np, device=device)
        A = anomaly_score(x, E, G, D)
        center = patch_np[:, :3].mean(0)
        _, nn_idx = tree.query(center, k=K_PATCH)
        scores_sum[nn_idx]   += A
        scores_count[nn_idx] += 1

    scores_count = np.maximum(scores_count, 1)
    point_scores = scores_sum / scores_count

    auroc = float(roc_auc_score(labels, point_scores))
    ap    = float(average_precision_score(labels, point_scores))
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(labels, point_scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    best_thr = float(thr[np.argmax(f1s[:-1])]) if len(thr) > 0 else 0.5
    preds = (point_scores >= best_thr).astype(np.int64)
    f1 = float(f1_score(labels, preds, zero_division=0))

    return {
        'filename' : cloud.get('filename', '?'),
        'auroc'    : round(auroc, 4),
        'f1'       : round(f1,    4),
        'ap'       : round(ap,    4),
        'n_crack'  : int((labels == 1).sum()),
        'n_normal' : int((labels == 0).sum()),
        # dados de visualização (filtrados antes do CSV)
        'xyz'      : features[:, :3].astype(np.float32),
        'rgb_orig' : features[:, 3:6].astype(np.float32),
        'preds'    : preds,
        'gt_labels': labels,
    }


def run_loco(labeled: list, normals: list,
             device: torch.device = DEVICE) -> list:
    """LOCO: para cada fold, treina BiGAN nos normais, avalia na nuvem de teste."""
    results = []
    for i, test_cloud in enumerate(labeled):
        log.info(f"Fold {i+1}/{len(labeled)} — {test_cloud.get('filename','?')}")
        E, G, D = train_bigan(normals, n_epochs=PRETRAIN_EP, device=device)
        r = evaluate_cloud_gan(test_cloud, E, G, D, device)
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
    print(f"\n{'='*55}")
    print(f"  DGCNN-BiGAN LOCO | AUROC {np.mean(aurocs):.4f}±{np.std(aurocs):.4f} "
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
                                      r['filename'].replace('.ply', '_gan.ply')),
                gt_labels=r.get('gt_labels'),
            )
    log.info(f"PLY salvos em: {PLY_DIR}")

    # Gráficos 2D de métricas
    plot_loco_metrics(results, VIS_DIR, 'DGCNN-BiGAN', ts)

    # CSV — filtra arrays numpy antes de escrever
    csv_results = [{k: v for k, v in r.items() if k not in _VIS_KEYS}
                   for r in results]
    csv_path = f'{RESULTS_DIR}/loco_{ts}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_results[0].keys())
        writer.writeheader()
        writer.writerows(csv_results)
    log.info(f"CSV: {csv_path}")


if __name__ == '__main__':
    main()
