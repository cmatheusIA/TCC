# ============================================================================
# PIPELINE DE DADOS — pré-processamento, augmentation, dataset e loaders
# ============================================================================
# Compartilhado por gan_rachaduras_v5.py e teacher_student_v1.py.
# Referências:
#   Thomas et al. (2019) — KPConv pipeline
#   Zhou et al. (2024)   — R3D-AD preprocessing
# ============================================================================

from utils.config import *

import logging
import hashlib

log = setup_logging(LOG_PATH)


PREPROCESSOR = None   # inicializado em runtime após definir a classe

class PointCloudPreprocessor:
    """
    Pré-processamento completo: 7 etapas.
    Conforme Thomas et al. (2019) e o pipeline do notebook comparative-pure-torch.
    """

    def __init__(self, remove_outliers=True, nb_neighbors=20, std_ratio=2.0,
                 normalize_spatial=True, voxelize=True, voxel_size=VOXEL_SIZE,
                 normalize_features=True, estimate_normals=True, verbose=False):
        self.remove_outliers    = remove_outliers
        self.nb_neighbors       = nb_neighbors
        self.std_ratio          = std_ratio
        self.normalize_spatial  = normalize_spatial
        self.voxelize           = voxelize
        self.voxel_size         = voxel_size
        self.normalize_features = normalize_features
        self.estimate_normals   = estimate_normals
        self.verbose            = verbose

    def __call__(self, xyz, rgb=None, normals=None, scalar=None, labels=None):
        if self.remove_outliers:
            xyz, rgb, normals, scalar, labels = self._remove_outliers(
                xyz, rgb, normals, scalar, labels)

        if self.normalize_spatial:
            xyz = self._normalize_spatial(xyz)

        if self.voxelize:
            xyz, rgb, normals, scalar, labels = self._voxelize(
                xyz, rgb, normals, scalar, labels)

        if self.normalize_features:
            if rgb    is not None: rgb    = self._normalize_rgb(rgb)
            if scalar is not None: scalar = self._normalize_scalar(scalar)

        if self.estimate_normals:
            normals = self._estimate_normals(xyz, normals)

        extra = {
            'curvature'       : self._compute_curvature(xyz, normals),
            'density'         : self._compute_local_density(xyz),
            'variance'        : self._compute_local_variance(xyz),
            'surface_variation': self._compute_surface_variation(xyz, normals),
        }

        return {'xyz': xyz, 'rgb': rgb, 'normals': normals,
                'scalar': scalar, 'labels': labels, 'extra': extra}

    # ── helpers ──────────────────────────────────────────────────────────────

    def _remove_outliers(self, xyz, rgb, normals, scalar, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        _, ind = pcd.remove_statistical_outlier(self.nb_neighbors, self.std_ratio)
        ind = np.array(ind)
        return (xyz[ind],
                rgb[ind]    if rgb    is not None else None,
                normals[ind] if normals is not None else None,
                scalar[ind]  if scalar  is not None else None,
                labels[ind]  if labels  is not None else None)

    def _normalize_spatial(self, xyz):
        centroid = xyz.mean(0)
        xyz = xyz - centroid
        max_d = np.linalg.norm(xyz, axis=1).max()
        return (xyz / (max_d + 1e-8)).astype(np.float32)

    def _voxelize(self, xyz, rgb, normals, scalar, labels):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        if rgb     is not None: pcd.colors  = o3d.utility.Vector3dVector(rgb.astype(np.float64))
        if normals is not None: pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        pcd_down = pcd.voxel_down_sample(self.voxel_size)
        xyz_d   = np.asarray(pcd_down.points,  dtype=np.float32)
        rgb_d   = np.asarray(pcd_down.colors,  dtype=np.float32) if rgb     is not None else None
        nrm_d   = np.asarray(pcd_down.normals, dtype=np.float32) if normals is not None else None
        if scalar is not None or labels is not None:
            tree = cKDTree(np.asarray(pcd_down.points))
            _, idx = tree.query(xyz_d)
            if scalar is not None: scalar = scalar[idx]
            if labels is not None: labels = labels[idx]
        return xyz_d, rgb_d, nrm_d, scalar, labels

    def _normalize_rgb(self, rgb):
        if rgb.max() > 1.5: rgb = rgb / 255.0
        return rgb.astype(np.float32)

    def _normalize_scalar(self, scalar):
        lo, hi = scalar.min(), scalar.max()
        if hi - lo > 1e-8:
            return ((scalar - lo) / (hi - lo)).astype(np.float32)
        return np.zeros_like(scalar, dtype=np.float32)

    def _estimate_normals(self, xyz, normals=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        return np.asarray(pcd.normals, dtype=np.float32)

    def _compute_curvature(self, xyz, normals):
        tree = cKDTree(xyz)
        k = min(20, len(xyz))
        curv = np.zeros(len(xyz), dtype=np.float32)
        for i in range(len(xyz)):
            _, idx = tree.query(xyz[i], k=k)
            nb = xyz[idx]
            centered = nb - nb.mean(0)
            try:
                ev = np.linalg.eigvalsh(np.cov(centered.T))
                curv[i] = ev[0] / (ev.sum() + 1e-8)
            except: pass
        curv = (curv - curv.min()) / (curv.max() - curv.min() + 1e-8)
        return curv.reshape(-1, 1)

    def _compute_local_density(self, xyz, k=10):
        tree = cKDTree(xyz)
        k_a  = min(k + 1, len(xyz))
        dists, _ = tree.query(xyz, k=k_a)
        density = 1.0 / (dists[:, 1:].mean(1) + 1e-8)
        density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        return density.reshape(-1, 1).astype(np.float32)

    def _compute_local_variance(self, xyz, k=10):
        tree = cKDTree(xyz)
        k_a  = min(k + 1, len(xyz))
        _, idx = tree.query(xyz, k=k_a)
        var = np.array([np.var(xyz[i], axis=0).sum() for i in idx], dtype=np.float32)
        var = (var - var.min()) / (var.max() - var.min() + 1e-8)
        return var.reshape(-1, 1)

    def _compute_surface_variation(self, xyz, normals, k=10):
        if normals is None: normals = self._estimate_normals(xyz)
        tree = cKDTree(xyz)
        k_a  = min(k + 1, len(xyz))
        _, idx = tree.query(xyz, k=k_a)
        sv = np.zeros(len(xyz), dtype=np.float32)
        for i in range(len(xyz)):
            nb_nrm = normals[idx[i]]
            dots   = np.clip(nb_nrm @ normals[i], -1.0, 1.0)
            sv[i]  = np.std(np.arccos(dots))
        sv = (sv - sv.min()) / (sv.max() - sv.min() + 1e-8)
        return sv.reshape(-1, 1)


# ============================================================================
# AUGMENTAÇÃO (idêntica ao notebook)
# ============================================================================

class PointCloudAugmentation:
    """
    Augmentações que simulam fenômenos físicos reais:
    - Rotação Z 0–360° (invariância de posição do scanner), X/Y ±5°
    - Escala  (0.85–1.15)                    → diferenças de distância ao scanner
    - Jitter  (σ=0.01)                       → ruído de medição LiDAR
    - Dropout (10%)                          → oclusão parcial / reflexão especular
    - Spatial crop (r=0.4, p=0.4)           → força aprendizado de estatísticas locais
    Referência: Qi et al., 2017 (PointNet) | Thomas et al., 2019 (KPConv)
                R3D-AD (Zhou et al., ECCV 2024) | Uni-3DAD (Liu et al., 2025)
    """

    def __init__(self, scale_range=(0.85, 1.15), jitter_std=0.01,
                 dropout_ratio=0.10, spatial_crop=True,
                 spatial_crop_radius=0.4, spatial_crop_prob=0.4):
        self.scale_range         = scale_range
        self.jitter_std          = jitter_std
        self.dropout_ratio       = dropout_ratio
        self.spatial_crop        = spatial_crop
        self.spatial_crop_radius = spatial_crop_radius
        self.spatial_crop_prob   = spatial_crop_prob

    def __call__(self, xyz, features=None, labels=None):
        xyz = xyz.copy()

        # Rotação: Z irrestrito (0–360°), X/Y limitado a ±5° para preservar normais
        # R3D-AD (Zhou et al., ECCV 2024): rotação de eixo de gravidade completa (360°)
        # Uni-3DAD (Liu et al., 2025): X/Y ≤ ±5° para não inverter direção das normais
        if random.random() < 0.8:
            angles = [random.uniform(-5, 5),
                      random.uniform(-5, 5),
                      random.uniform(0, 360)]
            rot = Rotation.from_euler('xyz', angles, degrees=True)
            xyz = rot.apply(xyz)

        # Escala uniforme
        if random.random() < 0.5:
            s   = random.uniform(*self.scale_range)
            xyz = xyz * s

        # Jitter gaussiano
        if random.random() < 0.6:
            xyz = xyz + np.random.normal(0, self.jitter_std, xyz.shape)

        # Dropout aleatório de pontos
        if random.random() < 0.4 and self.dropout_ratio > 0:
            n_keep = max(128, int(len(xyz) * (1.0 - self.dropout_ratio)))
            idx    = np.sort(np.random.choice(len(xyz), n_keep, replace=False))
            xyz    = xyz[idx]
            if features is not None: features = features[idx]
            if labels   is not None: labels   = labels[idx]

        # Spatial crop: seleciona patch esférico local (radius=0.4)
        # Rachaduras são fenômenos locais — treinar em patches força o modelo a aprender
        # estatísticas de superfície em escala local (R3D-AD, Zhou et al., ECCV 2024).
        # Aplicado APENAS em nuvens normais (spatial_crop=False para avaria_*).
        if self.spatial_crop and random.random() < self.spatial_crop_prob:
            center = xyz[np.random.randint(len(xyz))]
            dists  = np.linalg.norm(xyz - center, axis=1)
            mask   = dists < self.spatial_crop_radius
            if mask.sum() >= 64:
                idx      = np.where(mask)[0]
                xyz      = xyz[idx]
                if features is not None: features = features[idx]
                if labels   is not None: labels   = labels[idx]

        return xyz.astype(np.float32), features, labels


# ============================================================================
# DATASET  (batch_size=1, nuvens completas, sem padding)
# ============================================================================

class PointCloudDataset(Dataset):
    """
    Dataset de nuvens pré-recortadas (segmentos de paredes).

    Cada PLY é um corte espacial com tamanho variável — NÃO se aplica
    random sampling nem zero-padding, pois isso destruiria a distribuição
    geométrica local que o modelo precisa aprender.

    • Treino  : nuvens SEM rachadura (has_crack=False)
      O WGAN-GP aprende P(superfície normal).
    • Avaliação: nuvens COM rachadura (has_crack=True)
      Labels usados APENAS como ground-truth na avaliação.
      [Protocolo semi-supervisionado — Uni-3DAD, Liu et al., 2024]
    """

    def __init__(self, data_list: list, augment: bool = False,
                 preprocessor: PointCloudPreprocessor = None,
                 spatial_crop: bool = True):
        self.data         = data_list
        self.augment      = augment
        self.preprocessor = preprocessor
        self.augmentor    = PointCloudAugmentation(spatial_crop=spatial_crop) if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        d        = self.data[idx]
        features = d['features'].copy()          # [N, 15] float32
        labels   = d['labels'].copy()            # [N]     int64

        if self.augment and self.augmentor is not None:
            xyz        = features[:, :3]
            other_feat = features[:, 3:]
            labels_f   = labels.astype(np.float32)
            xyz, other_feat, labels_f = self.augmentor(xyz, other_feat, labels_f)
            features = np.concatenate([xyz, other_feat], axis=1).astype(np.float32)
            labels   = labels_f.astype(np.int64)

        return {
            'features' : torch.tensor(features, dtype=torch.float32),  # [N, 15]
            'labels'   : torch.tensor(labels,   dtype=torch.long),      # [N]
            'filename' : d['filename'],
            'has_crack': d['has_crack'],
            'n_points' : len(features),
        }


def collate_fn(batch: list) -> dict:
    """
    Collate para nuvens de tamanho variável (batch_size=1).
    Retorna listas — o loop de treino extrai o item 0 diretamente.
    """
    return {
        'features' : [b['features']  for b in batch],
        'labels'   : [b['labels']    for b in batch],
        'filename' : [b['filename']  for b in batch],
        'has_crack': [b['has_crack'] for b in batch],
        'n_points' : [b['n_points']  for b in batch],
    }


# ============================================================================
# CARREGAMENTO DE PLY
# ============================================================================

PREPROCESSOR = PointCloudPreprocessor(
    remove_outliers=True, normalize_spatial=True,
    voxelize=True, voxel_size=VOXEL_SIZE,
    normalize_features=True, estimate_normals=True, verbose=False,
)


def load_ply_file(path: str, preprocessor: PointCloudPreprocessor = None) -> dict | None:
    """
    Lê um arquivo PLY e retorna dict com features [N,15] e labels [N].

    Campos PLY esperados:
      x, y, z, r, g, b, nx, ny, nz
      Scalar_scalar_field  (campo escalar do scanner)
      labels / label       (anotação manual: 0=normal, 1=rachadura)

    Se 'labels' não existir → nuvem sem rachadura (all zeros).
    Se todo label=0          → nuvem sem rachadura.
    """


    try:
        if preprocessor is None:          # ← adicionar estas duas linhas
            preprocessor = PREPROCESSOR   # ← resolve o valor real em chamada, não em definição
        plydata = PlyData.read(path)
        v       = plydata['vertex']
        props   = list(v.data.dtype.names)

        # ── XYZ ──────────────────────────────────────────────────────────────
        xyz = np.column_stack([
            v[c].astype(np.float32) for c in ['x', 'y', 'z'] if c in props
        ])
        if xyz.shape[1] < 3:
            log.warning(f"XYZ incompleto em {path}")
            return None

        # ── RGB ───────────────────────────────────────────────────────────────
        rgb_cols = [c for c in props if c.lower() in ('r', 'g', 'b', 'red', 'green', 'blue')]
        if len(rgb_cols) >= 3:
            rgb = np.column_stack([v[c].astype(np.float32) for c in rgb_cols[:3]])
            if rgb.max() > 1.5: rgb = rgb / 255.0
        else:
            rgb = np.full((len(xyz), 3), 0.5, dtype=np.float32)

        # ── Normais ───────────────────────────────────────────────────────────
        nrm_cols = [c for c in props if c.lower() in ('nx', 'ny', 'nz')]
        normals  = np.column_stack([v[c].astype(np.float32) for c in nrm_cols[:3]]) \
                   if len(nrm_cols) >= 3 else None

        # ── has_crack: vem do nome do arquivo ────────────────────────────────
        fname     = os.path.basename(path).lower()
        has_crack = ('avaria' in fname) and ('n_avaria' not in fname)

        # ── Scalar field (features — NÃO confundir com labels) ────────────────
        scalar_candidates = [c for c in props
                             if 'scalar' in c.lower() and 'label' not in c.lower()
                             and 'original' not in c.lower()]
        scalar = v[scalar_candidates[0]].reshape(-1, 1).astype(np.float32) \
                 if scalar_candidates else np.zeros((len(xyz), 1), dtype=np.float32)

        # ── Labels por ponto (ground-truth, NÃO usados no treino) ─────────────
        # Prioridade: scalar_labels > original_index/original > fallback por nome
        label_candidates = [c for c in props if c.lower() == 'scalar_labels']
        if not label_candidates:
            label_candidates = [c for c in props
                                if 'original_index' in c.lower()
                                or ('original' in c.lower() and c.lower() != 'scalar_labels')]
        if not label_candidates:
            label_candidates = [c for c in props if 'label' in c.lower()]

        if label_candidates:
            raw_labels = v[label_candidates[0]].astype(np.int64)
        else:
            # Fallback: todos 1 para avaria, todos 0 para n_avaria
            raw_labels = (np.ones(len(xyz), dtype=np.int64) if has_crack
                          else np.zeros(len(xyz), dtype=np.int64))

        # ── Pré-processamento ─────────────────────────────────────────────────
        proc = preprocessor(
            xyz, rgb, normals,
            scalar, raw_labels.reshape(-1, 1).astype(np.float32)
        )
        xyz     = proc['xyz']
        rgb     = proc['rgb']
        normals = proc['normals']
        scalar  = proc['scalar']
        labels  = proc['labels'].reshape(-1).astype(np.int64) if proc['labels'] is not None \
                  else np.zeros(len(xyz), dtype=np.int64)
        extra   = proc['extra']

        # ── Feature vector 16D ───────────────────────────────────────────────
        # [xyz(3), rgb(3), normals(3), scalar(1), curv(1), dens(1), var(1), sv(1), lum(1), sat(1)]
        # scalar_labels EXCLUÍDO: incluí-lo como feature vaza o ground truth para o modelo
        #   — durante treino é 0 em todos os pontos, durante avaliação é 1 nos cracks,
        #     gerando erro de reconstrução garantido nessa dimensão (avaliação inflada).
        # lum: luminosidade média (R+G+B)/3 normalizada — rachaduras ≈0.21 vs normal ≈0.64
        # sat: saturação cromática max(RGB)-min(RGB) — rachaduras ≈0.05 vs normal ≈0.35
        lum = rgb.mean(axis=1, keepdims=True)                        # (N,1) ∈ [0,1]
        sat = (rgb.max(axis=1) - rgb.min(axis=1)).reshape(-1, 1)     # (N,1) ∈ [0,1]

        features = np.concatenate([
            xyz,                      # 3  → cols 0-2
            rgb,                      # 3  → cols 3-5
            normals,                  # 3  → cols 6-8
            scalar,                   # 1  → col  9
            extra['curvature'],       # 1  → col 10
            extra['density'],         # 1  → col 11
            extra['variance'],        # 1  → col 12
            extra['surface_variation'],# 1  → col 13
            lum,                      # 1  → col 14
            sat,                      # 1  → col 15
        ], axis=1).astype(np.float32) # → [N, 16]

        # Garantir 16D exatamente
        if features.shape[1] != INPUT_DIM:
            pad = INPUT_DIM - features.shape[1]
            features = np.concatenate(
                [features, np.zeros((len(features), pad), dtype=np.float32)], axis=1
            ) if pad > 0 else features[:, :INPUT_DIM]

        return {
            'features' : features,
            'labels'   : labels,
            'xyz'      : xyz,          # ← adicionar esta linha
            'filename' : os.path.basename(path),
            'has_crack': has_crack,
            'n_points' : len(xyz),
        }

    except Exception as e:
        log.error(f"Erro ao carregar {path}: {e}")
        return None


def load_folder(folder: str) -> list:
    """Carrega todos os PLY de uma pasta, descartando duplicatas pós-processamento.

    Dois arquivos PLY diferentes podem produzir features idênticas após
    outlier removal + voxelização — deduplicação por hash MD5 das features
    evita que métricas de avaliação sejam distorcidas por nuvens repetidas.
    """
    ply_files = sorted(Path(folder).glob('**/*.ply'))
    log.info(f"Encontrados {len(ply_files)} arquivos PLY em {folder}")
    data        = []
    seen_hashes = set()
    for p in ply_files:
        d = load_ply_file(str(p))
        if d is not None:
            feat_hash = hashlib.md5(d['features'].tobytes()).hexdigest()
            if feat_hash in seen_hashes:
                log.warning(f"  DUPLICATA ignorada: {d['filename']} "
                            f"(features idênticas após pré-processamento)")
                continue
            seen_hashes.add(feat_hash)
            data.append(d)
            log.info(f"  OK {d['filename']} | {d['n_points'] if 'n_points' in d else len(d['features']):,} pts | crack={d['has_crack']}")
    log.info(f"  → {len(data)} nuvens carregadas")
    return data


def split_dataset(all_data: list) -> tuple:
    """
    Separa:
      train_list   → nuvens SEM rachadura (modelo aprende superfície normal)
      labeled_list → nuvens COM rachadura E scalar_labels (semi-supervised)
      eval_list    → todas as nuvens (avaliação com ground-truth)

    Protocolo semi-supervisionado:
    - Fase 1 (distilação): usa apenas train_list (sem labels)
    - Fase 2 (Push-Pull):  usa labeled_list (scalar_labels={0,1})
    """
    train_list   = [d for d in all_data if not d['has_crack']]
    labeled_list = [d for d in all_data if d['has_crack']]
    eval_list    = list(all_data)
    log.info(f"Split: {len(train_list)} normais (treino) | "
             f"{len(labeled_list)} rotulados (semi-sup) | "
             f"{len(eval_list)} total (avaliação)")
    if len(train_list) == 0:
        log.warning("AVISO: Nenhuma nuvem 'normal' encontrada. "
                    "Usando 80% aleatório para treino.")
        idx          = np.random.permutation(len(all_data))
        n_train      = max(1, int(0.8 * len(all_data)))
        train_list   = [all_data[i] for i in idx[:n_train]]
        labeled_list = [all_data[i] for i in idx[n_train:]]
        eval_list    = list(all_data)
    return train_list, labeled_list, eval_list


def make_loaders(train_list: list, eval_list: list,
                 labeled_list: list = None):
    """
    Cria DataLoaders de treino, avaliação e (opcional) supervisionado.

    labeled_list: nuvens avaria_* com scalar_labels — usado no treino semi-sup.
                  Se None, retorna labeled_dl=None (compatível com GAN v5).
    """
    nw = ENV_CONFIG['num_workers']
    train_ds = PointCloudDataset(train_list, augment=True,  preprocessor=None)
    eval_ds  = PointCloudDataset(eval_list,  augment=False, preprocessor=None)

    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True,
                          collate_fn=collate_fn, num_workers=nw,
                          pin_memory=ENV_CONFIG['pin_memory'],
                          persistent_workers=(nw > 0))
    eval_dl  = DataLoader(eval_ds,  batch_size=1, shuffle=False,
                          collate_fn=collate_fn, num_workers=nw,
                          pin_memory=ENV_CONFIG['pin_memory'])

    labeled_dl = None
    if labeled_list:
        # spatial_crop=False: preserva rachaduras completas nas nuvens avaria_*
        labeled_ds = PointCloudDataset(labeled_list, augment=True, preprocessor=None, spatial_crop=False)
        labeled_dl = DataLoader(labeled_ds, batch_size=1, shuffle=True,
                                collate_fn=collate_fn, num_workers=nw,
                                pin_memory=ENV_CONFIG['pin_memory'],
                                persistent_workers=(nw > 0))
        log.info(f"Train: {len(train_ds)} nuvens | "
                 f"Labeled: {len(labeled_ds)} nuvens | Eval: {len(eval_ds)} nuvens")
    else:
        log.info(f"Train: {len(train_ds)} nuvens | Eval: {len(eval_ds)} nuvens")

    return train_dl, labeled_dl, eval_dl


# ============================================================================
# MÓDULOS AUXILIARES DA ARQUITETURA
# ============================================================================



PREPROCESSOR = PointCloudPreprocessor()
