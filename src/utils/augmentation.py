"""
Augmentation pipeline para detecção binária de rachaduras.

CrackPaste: extrai segmentos de rachadura de nuvens avaria via DBSCAN
            e os insere em nuvens normais para criar amostras sintéticas.
            Inspirado em Choi et al. (IROS 2021, arXiv:2007.13373).

augment_cloud: augmentações geométricas padrão (rotação Z, jitter, scale,
               flip, dropout de pontos, jitter de features RGB).
               Inspirado em PointNeXt (Qian et al., NeurIPS 2022).
"""
import numpy as np
from sklearn.cluster import DBSCAN


class CrackDatabase:
    """Banco de patches de rachadura extraídos de nuvens avaria."""

    def __init__(self, clouds: list, eps: float = 0.05, min_pts: int = 10,
                 min_patch_size: int = 20):
        self.patches = []
        for d in clouds:
            if not d.get('has_crack'):
                continue
            labels = d.get('labels')
            if labels is None:
                continue
            crack_idx = np.where(labels == 1)[0]
            if len(crack_idx) < min_pts:
                continue
            xyz_crack = d['features'][crack_idx, :3]
            cluster_labels = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(xyz_crack)
            for cid in np.unique(cluster_labels):
                if cid < 0:
                    continue
                mask = cluster_labels == cid
                if mask.sum() < min_patch_size:
                    continue
                patch_feat = d['features'][crack_idx[mask]].copy()
                patch_feat[:, :3] -= patch_feat[:, :3].mean(0)  # centrar em origem
                avg_normal = patch_feat[:, 6:9].mean(0)
                self.patches.append({'features': patch_feat, 'normal': avg_normal})

    def __len__(self) -> int:
        return len(self.patches)

    def sample(self, rng=None) -> dict:
        if rng is None:
            rng = np.random.default_rng()
        return self.patches[rng.integers(len(self.patches))]


def crack_paste(cloud: dict, patch: dict, rng=None) -> dict:
    """
    Insere um patch de rachadura em uma nuvem normal.
    Retorna novo dict de nuvem com pontos de crack adicionados.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = len(cloud['features'])
    anchor_xyz = cloud['features'][rng.integers(N), :3]
    patch_feat = patch['features'].copy()
    patch_feat[:, :3] += anchor_xyz
    new_features = np.vstack([cloud['features'], patch_feat]).astype(np.float32)
    new_labels = np.concatenate([
        np.zeros(N, dtype=np.int64),
        np.ones(len(patch_feat), dtype=np.int64)
    ])
    result = {k: v for k, v in cloud.items()}
    result['features'] = new_features
    result['labels'] = new_labels
    result['has_crack'] = True
    return result


def augment_cloud(features: np.ndarray, labels: np.ndarray = None,
                  rotate_z: bool = True,
                  jitter_std: float = 0.002,
                  scale_range: tuple = (0.8, 1.2),
                  flip_prob: float = 0.5,
                  dropout_max: float = 0.2,
                  feat_jitter_rgb: float = 0.01,
                  feat_jitter_sf: float = 0.0,
                  rng=None) -> tuple:
    """
    Augmentações geométricas para nuvem de pontos.
    feat_jitter_sf=0.0 por padrão: scalar_field não é perturbado (valor físico real).
    Retorna (features_aug, labels_aug).
    """
    if rng is None:
        rng = np.random.default_rng()
    feat = features.copy()

    if rotate_z:
        theta = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        feat[:, :3] = feat[:, :3] @ R.T
        if feat.shape[1] >= 9:
            feat[:, 6:9] = feat[:, 6:9] @ R.T

    feat[:, :3] += rng.normal(0, jitter_std, feat[:, :3].shape).astype(np.float32)

    scale = rng.uniform(*scale_range)
    feat[:, :3] = (feat[:, :3] * scale).astype(np.float32)

    if rng.random() < flip_prob:
        feat[:, 0] *= -1
        if feat.shape[1] >= 7:
            feat[:, 6] *= -1

    p_drop = rng.uniform(0, dropout_max)
    keep = rng.random(len(feat)) > p_drop
    if keep.sum() < 10:
        keep[:10] = True
    feat = feat[keep]

    feat[:, 3:6] = np.clip(
        feat[:, 3:6] + rng.normal(0, feat_jitter_rgb, feat[:, 3:6].shape),
        0, 1
    ).astype(np.float32)

    if feat_jitter_sf > 0 and feat.shape[1] > 9:
        feat[:, 9] += rng.normal(0, feat_jitter_sf, feat[:, 9].shape).astype(np.float32)

    lbl = labels[keep] if labels is not None else None
    return feat, lbl
