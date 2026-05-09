import numpy as np
from scipy.spatial import cKDTree


# ── Improved scalar features (v2) ─────────────────────────────────────────────

def is_damage_scalar(col_name: str) -> bool:
    """scalar_R is the RGB red channel — not a damage metric."""
    return col_name.lower() != 'scalar_r'


def _local_z_score(xyz: np.ndarray, scalar: np.ndarray, k: int) -> np.ndarray:
    k_q = min(k + 1, len(xyz))
    _, idx = cKDTree(xyz).query(xyz, k=k_q, workers=-1)
    nb = scalar[idx[:, 1:]]
    return ((scalar - nb.mean(1)) / (nb.std(1) + 1e-8)).astype(np.float32)


def _gradient_mag(xyz: np.ndarray, scalar: np.ndarray, k: int) -> np.ndarray:
    k_q = min(k + 1, len(xyz))
    _, idx = cKDTree(xyz).query(xyz, k=k_q, workers=-1)
    dx  = xyz[idx[:, 1:]] - xyz[:, None, :]      # (N, k, 3)
    ds  = scalar[idx[:, 1:]] - scalar[:, None]   # (N, k)
    ATA = np.einsum('nki,nkj->nij', dx, dx) + 1e-6 * np.eye(3)
    ATb = np.einsum('nki,nk->ni', dx, ds)
    try:
        g   = np.linalg.solve(ATA, ATb[:, :, None]).squeeze(-1)  # (N, 3)
        mag = np.linalg.norm(g, axis=1)
        return (mag / (mag.max() + 1e-8)).astype(np.float32)
    except np.linalg.LinAlgError:
        return np.zeros(len(xyz), dtype=np.float32)


def _crack_connectivity(xyz: np.ndarray, scalar: np.ndarray, k: int) -> np.ndarray:
    thr  = np.percentile(scalar, 75)
    high = (scalar >= thr).astype(np.float32)
    k_q  = min(k + 1, len(xyz))
    _, idx = cKDTree(xyz).query(xyz, k=k_q, workers=-1)
    return high[idx[:, 1:]].mean(axis=1).astype(np.float32)


def compute_improved_scalar_features(
    xyz: np.ndarray,
    scalar_norm: np.ndarray,
    col_name: str,
    k_zscore: int = 10,
    k_grad: int = 15,
) -> np.ndarray:
    """
    Retorna (N, 2): [z_score, gradient_mag].
    Zeros se col_name for scalar_R (canal RGB, não métrica de dano).

    z_score      — anomalia local vs vizinhança (k=10, cotovelo empírico AUC=0.945)
    gradient_mag — magnitude do gradiente 3D (k=15, mínimo para lstsq estável)

    connectivity foi removida: AUC=0.50 em todo sweep de k (percentil 75 cai na
    região normal pois cracks são ~7% do dataset — feature não discrimina).
    """
    if not is_damage_scalar(col_name):
        return np.zeros((len(xyz), 2), dtype=np.float32)

    return np.column_stack([
        _local_z_score(xyz, scalar_norm, k=k_zscore),
        _gradient_mag(xyz, scalar_norm, k=k_grad),
    ]).astype(np.float32)


# ── Funções originais ─────────────────────────────────────────────────────────

def iqr_normalize_sf(sf: np.ndarray) -> np.ndarray:
    """Robust per-cloud scalar_field normalization.
    Maps median → 0, IQR → 1. Handles constant sf gracefully."""
    sf = np.asarray(sf, dtype=np.float32)
    median = np.median(sf)
    q25, q75 = np.percentile(sf, [25, 75])
    iqr = q75 - q25
    return ((sf - median) / (iqr + 1e-8)).astype(np.float32)


def extract_local_sf_features(features: np.ndarray, k: int = 32) -> np.ndarray:
    """
    Extract 8D local feature vector per point for ScalarMemory.

    Input:  features (N, 16) — full 16D feature vector
            k                — neighborhood size

    Output: (N, 8) float32
      [0] mean(sf_local)
      [1] std(sf_local)
      [2] percentile_10(sf_local)
      [3] percentile_90(sf_local)
      [4] mean(curvature_neighbors)   col 10
      [5] std(normals_nx_neighbors)   col 6
      [6] std(normals_ny_neighbors)   col 7
      [7] std(normals_nz_neighbors)   col 8
    """
    N = len(features)
    k_actual = min(k, N)

    xyz = features[:, :3]
    sf  = iqr_normalize_sf(features[:, 9])
    nx, ny, nz = features[:, 6], features[:, 7], features[:, 8]
    curv = features[:, 10]

    tree = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k_actual, workers=-1)   # (N, k_actual)

    sf_nb   = sf[idx]       # (N, k)
    curv_nb = curv[idx]
    nx_nb   = nx[idx]
    ny_nb   = ny[idx]
    nz_nb   = nz[idx]

    out = np.column_stack([
        sf_nb.mean(axis=1),
        sf_nb.std(axis=1) + 1e-8,
        np.percentile(sf_nb, 10, axis=1),
        np.percentile(sf_nb, 90, axis=1),
        curv_nb.mean(axis=1),
        nx_nb.std(axis=1) + 1e-8,
        ny_nb.std(axis=1) + 1e-8,
        nz_nb.std(axis=1) + 1e-8,
    ]).astype(np.float32)

    return out
