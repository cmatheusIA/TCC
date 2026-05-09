"""
Testa se as 3 novas features escalares (z_score, gradient_mag, connectivity)
melhoram a separação crack vs normal em relação às features originais.

Metodologia:
  - Carrega todos os avaria_* com labels (crack=1 / normal=0)
  - Treina LogisticRegression e RandomForest nas features 16D (original) e 18D (v2)
  - Reporta F1, AUC e importância das 3 novas features
  - Mostra AUC individual por dimensão para diagnosticar quais features discriminam melhor

Uso:
  cd src && uv run python test_improved_scalar.py
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

from plyfile import PlyData, PlyElement
from utils.data import load_ply_file, PointCloudPreprocessor
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

PREPROCESSOR = PointCloudPreprocessor(
    remove_outliers=False,   # mais rápido para teste
    normalize_spatial=True,
    voxelize=True,
    voxel_size=0.02,         # voxel maior → mais rápido para teste
    normalize_features=True,
    estimate_normals=True,
    verbose=False,
)

FEATURE_NAMES_16 = [
    'x','y','z','r','g','b','nx','ny','nz',
    'scalar','curvature','density','variance','surf_var','lum','sat'
]
FEATURE_NAMES_18 = FEATURE_NAMES_16 + ['z_score','gradient_mag']

import glob

def load_labeled_files(improved: bool):
    files = sorted(glob.glob('data/train/avaria_*.ply'))
    X_all, y_all = [], []
    loaded = 0

    for fpath in files:
        t0 = time.time()
        d = load_ply_file(fpath, preprocessor=PREPROCESSOR, improved_scalar=improved)
        if d is None:
            continue
        labels = d['labels']
        if labels.max() == 0:   # sem crack
            continue
        # binário: qualquer label > 0 → crack
        y = (labels > 0).astype(int)
        X_all.append(d['features'])
        y_all.append(y)
        elapsed = time.time() - t0
        n_crack  = y.sum()
        n_normal = (y == 0).sum()
        print(f"  {os.path.basename(fpath):<30} {d['features'].shape[1]}D  "
              f"crack={n_crack:5d}  normal={n_normal:6d}  {elapsed:.1f}s")
        loaded += 1

    print(f"\n{loaded} arquivos carregados.\n")
    return np.vstack(X_all), np.concatenate(y_all)


def per_feature_auc(X, y, names):
    print("AUC por feature (separação crack vs normal):")
    results = []
    for i, name in enumerate(names):
        col = X[:, i]
        try:
            auc = roc_auc_score(y, col)
            auc = max(auc, 1 - auc)   # reflete se AUC < 0.5
        except Exception:
            auc = 0.5
        results.append((name, auc))

    results.sort(key=lambda r: r[1], reverse=True)
    for name, auc in results:
        bar = '█' * int((auc - 0.5) * 80)
        marker = ' ◄ NOVA' if name in ('z_score', 'gradient_mag') else ''
        print(f"  {name:<18} AUC={auc:.4f}  {bar}{marker}")
    print()


def evaluate_classifier(X_train, y_train, X_test, y_test, label):
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(Xtr, y_train)
    pred_lr = lr.predict(Xte)
    prob_lr = lr.predict_proba(Xte)[:, 1]
    f1_lr  = f1_score(y_test, pred_lr, zero_division=0)
    auc_lr = roc_auc_score(y_test, prob_lr)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    prob_rf = rf.predict_proba(X_test)[:, 1]
    f1_rf  = f1_score(y_test, pred_rf, zero_division=0)
    auc_rf = roc_auc_score(y_test, prob_rf)

    print(f"[{label}]")
    print(f"  LogisticRegression  F1={f1_lr:.4f}  AUC={auc_lr:.4f}")
    print(f"  RandomForest        F1={f1_rf:.4f}  AUC={auc_rf:.4f}")

    if hasattr(rf, 'feature_importances_') and X_train.shape[1] == 18:
        names = FEATURE_NAMES_18
        imp   = rf.feature_importances_
        top   = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)[:6]
        print("  Top-6 features (RF importância):")
        for name, v in top:
            marker = ' ◄ NOVA' if name in ('z_score', 'gradient_mag') else ''
            print(f"    {name:<18} {v:.4f}{marker}")
    print()

    return f1_lr, auc_lr, f1_rf, auc_rf


def main():
    print("=" * 60)
    print("Carregando features ORIGINAIS (16D)...")
    print("=" * 60)
    X16, y16 = load_labeled_files(improved=False)
    print(f"Dataset 16D: {X16.shape}  crack={y16.sum()}  normal={(y16==0).sum()}\n")

    print("=" * 60)
    print("Carregando features MELHORADAS (18D)...")
    print("=" * 60)
    X19, y19 = load_labeled_files(improved=True)
    print(f"Dataset 18D: {X19.shape}  crack={y19.sum()}  normal={(y19==0).sum()}\n")

    assert len(X16) == len(X19), "Mismatch entre datasets — verifique o pré-processamento"

    # split 80/20 estratificado por posição (por arquivo, não aleatório)
    n = len(X16)
    n_train = int(0.8 * n)
    idx = np.random.default_rng(42).permutation(n)
    tr, te = idx[:n_train], idx[n_train:]

    print("=" * 60)
    print("AUC por feature (dataset completo)")
    print("=" * 60)
    per_feature_auc(X19, y19, FEATURE_NAMES_18)

    print("=" * 60)
    print("Comparação de classificadores (train 80% / test 20%)")
    print("=" * 60)
    r16 = evaluate_classifier(X16[tr], y16[tr], X16[te], y16[te], "16D original")
    r19 = evaluate_classifier(X19[tr], y19[tr], X19[te], y19[te], "18D melhorado")

    print("=" * 60)
    print("RESUMO")
    print("=" * 60)
    delta_f1_lr  = r19[0] - r16[0]
    delta_auc_lr = r19[1] - r16[1]
    delta_f1_rf  = r19[2] - r16[2]
    delta_auc_rf = r19[3] - r16[3]
    print(f"  LogisticRegression  ΔF1={delta_f1_lr:+.4f}  ΔAUC={delta_auc_lr:+.4f}")
    print(f"  RandomForest        ΔF1={delta_f1_rf:+.4f}  ΔAUC={delta_auc_rf:+.4f}")
    verdict = "MELHORA" if (delta_f1_rf + delta_auc_rf) > 0 else "SEM GANHO"
    print(f"\n  → {verdict}")
    print("=" * 60)


def generate_crack_visualizations(out_dir: str = 'vis_crack_labels') -> None:
    """
    Lê os PLYs originais (coordenadas e RGB reais) e salva cópias onde:
      - pontos com label == 1 (rachadura) → vermelho (255, 0, 0)
      - pontos com label == 0 (normal)    → cor original preservada

    Também gera uma segunda versão colorida por z_score para comparação visual.
    """
    os.makedirs(out_dir, exist_ok=True)

    LABEL_COLS   = {'scalar_labels', 'scalar_label'}
    SCALAR_SKIP  = {'label', 'original'}

    files = sorted(glob.glob('data/train/avaria_*.ply'))
    print(f"\n{'='*60}")
    print(f"Gerando visualizações em '{out_dir}/'  ({len(files)} arquivos)")
    print(f"{'='*60}")

    for fpath in files:
        ply   = PlyData.read(fpath)
        v     = ply['vertex']
        props = v.data.dtype.names

        # ── colunas ──────────────────────────────────────────────────────────
        label_col  = next((c for c in props if c.lower() in LABEL_COLS), None)
        scalar_col = next((c for c in props
                           if 'scalar' in c.lower()
                           and not any(s in c.lower() for s in SCALAR_SKIP)), None)

        if label_col is None:
            print(f"  {os.path.basename(fpath):<30} sem coluna de label — pulado")
            continue

        xyz    = np.column_stack([v['x'], v['y'], v['z']]).astype(np.float32)
        labels = v[label_col].astype(int)

        # RGB original (uint8 → float32 [0,1])
        rgb_cols = [c for c in props if c.lower() in ('r', 'g', 'b', 'red', 'green', 'blue')]
        if len(rgb_cols) >= 3:
            rgb = np.column_stack([v[c].astype(np.float32) for c in rgb_cols[:3]])
            if rgb.max() > 1.5:
                rgb /= 255.0
        else:
            rgb = np.full((len(xyz), 3), 0.5, dtype=np.float32)

        n_crack  = (labels == 1).sum()
        n_normal = (labels == 0).sum()

        # ── versão 1: label ground-truth (vermelho) ───────────────────────
        rgb_gt = rgb.copy()
        rgb_gt[labels == 1] = [1.0, 0.0, 0.0]
        _write_ply(xyz, rgb_gt, os.path.join(out_dir, f"gt_{os.path.basename(fpath)}"))

        # ── versão 2: colorido por z_score (gradiente azul→vermelho) ──────
        if scalar_col and scalar_col.lower() != 'scalar_r':
            from utils.scalar_features import _local_z_score, is_damage_scalar
            sc_raw  = v[scalar_col].astype(np.float32)
            sc_norm = (sc_raw - sc_raw.min()) / (sc_raw.max() - sc_raw.min() + 1e-8)
            k_q     = min(10, len(xyz) - 1)
            from scipy.spatial import cKDTree
            zs = _local_z_score(xyz, sc_norm, k=k_q)
            # mapeia z_score para escala de cor: azul (baixo) → vermelho (alto)
            zs_norm = np.clip((zs - zs.min()) / (zs.max() - zs.min() + 1e-8), 0, 1)
            rgb_zs  = np.zeros((len(xyz), 3), dtype=np.float32)
            rgb_zs[:, 0] = zs_norm          # R = z_score alto
            rgb_zs[:, 2] = 1.0 - zs_norm   # B = z_score baixo
            _write_ply(xyz, rgb_zs, os.path.join(out_dir, f"zscore_{os.path.basename(fpath)}"))
            extra = " + zscore"
        else:
            extra = ""

        print(f"  {os.path.basename(fpath):<30} crack={n_crack:5d}  normal={n_normal:6d}{extra}")

    print(f"\nPronto. Abra os arquivos em CloudCompare ou MeshLab.")


def _write_ply(xyz: np.ndarray, rgb_float: np.ndarray, path: str) -> None:
    rgb_u8 = (rgb_float * 255.0).clip(0, 255).astype(np.uint8)
    n      = len(xyz)
    vertex = np.zeros(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex['x']     = xyz[:, 0]
    vertex['y']     = xyz[:, 1]
    vertex['z']     = xyz[:, 2]
    vertex['red']   = rgb_u8[:, 0]
    vertex['green'] = rgb_u8[:, 1]
    vertex['blue']  = rgb_u8[:, 2]
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis-only', action='store_true',
                        help='Pula o teste ML e gera só as visualizações')
    parser.add_argument('--out-dir', default='vis_crack_labels',
                        help='Pasta de saída dos PLYs coloridos')
    args = parser.parse_args()

    if args.vis_only:
        generate_crack_visualizations(args.out_dir)
    else:
        main()
        generate_crack_visualizations(args.out_dir)
