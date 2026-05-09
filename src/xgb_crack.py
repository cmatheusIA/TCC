# ============================================================================
# XGB CRACK — Detecção supervisionada de rachaduras com XGBoost
# ============================================================================
#
# Path 6 do roadmap (CONTINUACAO_23042026.md):
#   Feature engineering: SF + SF_rank + sf_contrast + GMM stats por nuvem
#   Protocolo: leave-one-cloud-out cross-validation (35 nuvens avaria)
#   Baseline a superar: SF puro AUROC=0.935, SF-GMM AUROC=0.887
#   Clouds difíceis: avaria_17 (0.53), avaria_33 (0.60), avaria_21 (0.61)
#
# Features (12D por ponto):
#   sf_norm     — SF normalizado por nuvem [0,1]
#   sf_rank     — rank relativo dentro da nuvem
#   sf_contrast — sf / mean(sf_k_vizinhos): anomalia local de reflectância
#   z_crack     — (sf - mu_crack) / sigma_crack: z-score relativo ao componente crack
#   mu_crack, sigma_crack, crack_weight — parâmetros GMM do componente crack
#   overlap_ratio — fração de pontos normais dentro do intervalo de crack
#   is_bimodal  — 0/1 flag de bimodalidade
#   sf_p05, sf_p25, sf_p50 — percentis do SF da nuvem (contexto estatístico)
#
# Pós-processamento DBSCAN (Path 4):
#   Após predição XGBoost, filtra componentes conectados com < min_size pontos.
#   Remove ruído isolado sem usar labels.
#
# Referências:
#   Chen & Guestrin (2016) — XGBoost: A Scalable Tree Boosting System.
#   Ester et al. (1996) — DBSCAN, KDD.
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import csv, pickle, time
from datetime import datetime

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.cluster import DBSCAN
import xgboost as xgb

from utils.config import *
from utils.data import load_folder
from utils.evaluation import ScalarFieldGMM, save_colored_ply

log = setup_logging(f'{BASE_PATH}/logs_xgb')

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_XGB = f'{BASE_PATH}/results_xgb'
MODELS_XGB  = f'{BASE_PATH}/models_sf'
VIS_XGB     = f'{BASE_PATH}/visualizations_xgb'
PLY_XGB     = f'{BASE_PATH}/results_xgb/ply'
MODEL_PATH  = f'{MODELS_XGB}/xgb_crack.pkl'

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
K_CONTRAST    = 16    # vizinhos XYZ para sf_contrast
XGB_N_TREES   = 400
XGB_DEPTH     = 6
XGB_LR        = 0.05
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE = 0.8

DBSCAN_EPS     = 0.02   # em XYZ normalizado (esfera unitária)
DBSCAN_MINSIZE = 15     # componentes com < N pontos são ruído

FEATURE_NAMES = [
    'sf_norm', 'sf_rank', 'sf_contrast', 'z_crack',
    'mu_crack', 'sigma_crack', 'crack_weight', 'overlap_ratio', 'is_bimodal',
    'sf_p05', 'sf_p25', 'sf_p50',
    'sv', 'sat', 'sv_contrast',   # sinal residual confirmado por partial correlation
]


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def extract_gmm_stats(sf: np.ndarray) -> dict:
    """
    Ajusta ScalarFieldGMM e extrai parâmetros do componente de rachadura.
    Para nuvens unimodais: valores neutros (nuvem irrecuperável por SF).
    """
    gmm_obj = ScalarFieldGMM(sf).fit()

    if gmm_obj.modality == 'unimodal' or gmm_obj._gmm is None:
        mu    = float(np.percentile(sf, 10))
        sigma = float(sf.std()) + 1e-6
        return dict(mu_crack=mu, sigma_crack=sigma, crack_weight=0.5,
                    overlap_ratio=1.0, is_bimodal=0.0)

    g  = gmm_obj._gmm
    ci = gmm_obj._crack_idx

    mu_crack    = float(g.means_[ci, 0])
    sigma_crack = float(np.sqrt(g.covariances_[ci, 0, 0])) + 1e-6
    crack_wt    = float(g.weights_[ci])

    thr = gmm_obj.threshold
    normal_mask = sf > thr if mu_crack <= thr else sf < thr
    lo = mu_crack - 2 * sigma_crack
    hi = mu_crack + 2 * sigma_crack

    if normal_mask.sum() > 0:
        overlap = float(((sf[normal_mask] >= lo) & (sf[normal_mask] <= hi)).mean())
    else:
        overlap = 0.0

    return dict(mu_crack=mu_crack, sigma_crack=sigma_crack, crack_weight=crack_wt,
                overlap_ratio=overlap, is_bimodal=1.0)


def compute_contrast(xyz: np.ndarray, feat: np.ndarray, k: int = K_CONTRAST) -> np.ndarray:
    """
    contrast[i] = feat[i] / (mean(feat[vizinhos_i]) + 1e-8)
    Anomalia local: valores < 1 indicam ponto anômalo em relação à vizinhança.
    kNN em XYZ (já normalizado para esfera unitária).
    """
    tree = cKDTree(xyz)
    k_q  = min(k + 1, len(xyz))
    _, idx = tree.query(xyz, k=k_q, workers=-1)
    idx    = idx[:, 1:]
    neigh_mean = feat[idx].mean(axis=1)
    return feat / (neigh_mean + 1e-8)


def build_cloud_features(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Constrói feature matrix 15D e vetor de labels para uma nuvem.
    Retorna (X [N,15], y [N]).

    Features 0-11 : derivadas de SF + GMM (discriminação primária)
    Features 12-14: sv, sat, sv_contrast (sinal residual ortogonal ao SF —
                    confirmado por partial correlation AUROC 0.69 e 0.61)
    """
    sf     = d['features'][:, 9].astype(np.float32)
    sv     = d['features'][:, 13].astype(np.float32)  # surface_variation
    sat    = d['features'][:, 15].astype(np.float32)  # saturação cromática
    xyz    = d['features'][:, :3].astype(np.float32)
    labels = d['labels'] if d['labels'] is not None else np.zeros(len(sf), dtype=np.int64)

    sf_rank     = (rankdata(sf) / len(sf)).astype(np.float32)
    sf_contrast = compute_contrast(xyz, sf).astype(np.float32)
    sv_contrast = compute_contrast(xyz, sv).astype(np.float32)
    stats       = extract_gmm_stats(sf)

    mu_c    = float(stats['mu_crack'])
    sig_c   = float(stats['sigma_crack'])
    z_crack = ((sf - mu_c) / sig_c).astype(np.float32)

    p05, p25, p50 = (float(np.percentile(sf, q)) for q in (5, 25, 50))
    n = len(sf)

    X = np.column_stack([
        sf,
        sf_rank,
        sf_contrast,
        z_crack,
        np.full(n, mu_c,                     dtype=np.float32),
        np.full(n, sig_c,                    dtype=np.float32),
        np.full(n, stats['crack_weight'],    dtype=np.float32),
        np.full(n, stats['overlap_ratio'],   dtype=np.float32),
        np.full(n, stats['is_bimodal'],      dtype=np.float32),
        np.full(n, p05,                      dtype=np.float32),
        np.full(n, p25,                      dtype=np.float32),
        np.full(n, p50,                      dtype=np.float32),
        sv,                                              # surface_variation (residual 0.69)
        sat,                                             # saturação cromática (residual 0.61)
        sv_contrast,                                     # anomalia local de geometria
    ]).astype(np.float32)

    return X, labels.astype(np.int32)


# ============================================================================
# XGBOOST — TREINO
# ============================================================================

def make_xgb(scale_pos_weight: float) -> xgb.XGBClassifier:
    try:
        m = xgb.XGBClassifier(
            n_estimators=XGB_N_TREES, max_depth=XGB_DEPTH, learning_rate=XGB_LR,
            subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE,
            scale_pos_weight=scale_pos_weight, eval_metric='auc',
            tree_method='hist', device='cuda', random_state=42, verbosity=0,
        )
        # smoke test
        _X = np.random.rand(10, 12).astype(np.float32)
        _y = np.array([0]*8 + [1]*2, dtype=np.int32)
        m.fit(_X, _y)
        return xgb.XGBClassifier(
            n_estimators=XGB_N_TREES, max_depth=XGB_DEPTH, learning_rate=XGB_LR,
            subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE,
            scale_pos_weight=scale_pos_weight, eval_metric='auc',
            tree_method='hist', device='cuda', random_state=42, verbosity=0,
        )
    except Exception:
        log.info("GPU não disponível — usando CPU")
        return xgb.XGBClassifier(
            n_estimators=XGB_N_TREES, max_depth=XGB_DEPTH, learning_rate=XGB_LR,
            subsample=XGB_SUBSAMPLE, colsample_bytree=XGB_COLSAMPLE,
            scale_pos_weight=scale_pos_weight, eval_metric='auc',
            tree_method='hist', device='cpu', random_state=42, verbosity=0, n_jobs=-1,
        )


# ============================================================================
# PÓS-PROCESSAMENTO DBSCAN (Path 4)
# ============================================================================

def dbscan_postprocess(
    xyz: np.ndarray,
    y_pred: np.ndarray,
    eps: float = DBSCAN_EPS,
    min_size: int = DBSCAN_MINSIZE,
) -> np.ndarray:
    """
    Filtra componentes conectados de crack com < min_size pontos.
    Remove ruído isolado sem usar labels.
    Retorna pred_labels refinado [N].
    """
    result = np.zeros(len(y_pred), dtype=np.int32)
    crack_idx = np.where(y_pred == 1)[0]

    if len(crack_idx) < min_size:
        return result

    crack_xyz = xyz[crack_idx]
    db = DBSCAN(eps=eps, min_samples=3).fit(crack_xyz)
    comp_ids = db.labels_

    for cid in set(comp_ids) - {-1}:
        members = np.where(comp_ids == cid)[0]
        if len(members) >= min_size:
            result[crack_idx[members]] = 1

    return result


# ============================================================================
# LEAVE-ONE-CLOUD-OUT CV
# ============================================================================

def run_loco_cv(cloud_feats: dict, labeled: list, normals: list) -> list:
    """
    LOCO-CV: treina em todas as outras nuvens avaria + nuvens normais;
    testa na nuvem deixada de fora.

    cloud_feats: {filename: (X, y)}
    labeled    : nuvens avaria_* com labels
    normals    : nuvens n_avaria_* (labels = 0 em todos os pontos)
    """
    results = []

    for i, test_d in enumerate(labeled):
        fname = test_d['filename']
        X_test, y_test = cloud_feats[fname]

        if y_test.sum() < 5:
            log.info(f"[{i+1}/{len(labeled)}] {fname}: sem pontos crack — skip")
            continue

        train_fnames = [d['filename'] for d in labeled if d['filename'] != fname]
        train_fnames += [d['filename'] for d in normals]

        X_parts = [cloud_feats[f][0] for f in train_fnames]
        y_parts = [cloud_feats[f][1] for f in train_fnames]

        X_train = np.concatenate(X_parts, axis=0)
        y_train = np.concatenate(y_parts, axis=0)

        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        spw   = max(1.0, n_neg / max(n_pos, 1))

        model = make_xgb(spw)
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]

        try:
            auroc = roc_auc_score(y_test, y_prob)
            ap    = average_precision_score(y_test, y_prob)
        except Exception:
            auroc = ap = float('nan')

        # F1 com threshold ótimo
        best_f1, best_thr = 0.0, 0.5
        for thr in np.linspace(0.05, 0.95, 91):
            f = f1_score(y_test, (y_prob >= thr).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_thr = f, float(thr)

        # DBSCAN post-proc
        xyz_test = test_d['features'][:, :3]
        pred_raw  = (y_prob >= best_thr).astype(np.int32)
        pred_dbs  = dbscan_postprocess(xyz_test, pred_raw)
        f1_dbs    = f1_score(y_test, pred_dbs, zero_division=0)

        n_crack  = int(y_test.sum())
        n_normal = int((y_test == 0).sum())

        log.info(
            f"[{i+1:02d}/{len(labeled)}] {fname:<35} "
            f"AUROC={auroc:.4f}  F1={best_f1:.4f}  F1_dbs={f1_dbs:.4f}  "
            f"AP={ap:.4f}  (crack={n_crack:,} normal={n_normal:,})"
        )

        results.append({
            'filename' : fname,
            'n_crack'  : n_crack,
            'n_normal' : n_normal,
            'auroc'    : round(float(auroc),  4),
            'f1'       : round(float(best_f1), 4),
            'f1_dbscan': round(float(f1_dbs),  4),
            'ap'       : round(float(ap),      4),
            'thr'      : round(float(best_thr), 4),
        })

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print("  XGB CRACK — Detecção Supervisionada de Rachaduras")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    for p in [RESULTS_XGB, MODELS_XGB, VIS_XGB, PLY_XGB]:
        os.makedirs(p, exist_ok=True)

    run_dir = os.path.join(RESULTS_XGB, f'run_{ts}')
    os.makedirs(run_dir, exist_ok=True)

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    log.info("Carregando dados...")
    all_train = load_folder(DATA_TRAIN)
    all_test  = load_folder(DATA_TEST)
    all_data  = all_train + all_test

    labeled  = [d for d in all_data
                if d.get('has_crack') and d['labels'] is not None]
    normals  = [d for d in all_data if not d.get('has_crack', False)]

    log.info(f"Nuvens avaria (LOCO-CV): {len(labeled)}")
    log.info(f"Nuvens normais (treino): {len(normals)}")
    log.info(f"Features: {FEATURE_NAMES}")

    if not labeled:
        log.error("Sem nuvens avaria com labels — abortando.")
        return

    # ── 2. Feature engineering (pré-computado para todas as nuvens) ───────────
    log.info("\nComputando features por nuvem...")
    cloud_feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    all_clouds = labeled + normals
    for i, d in enumerate(all_clouds):
        t0 = time.time()
        X, y = build_cloud_features(d)
        cloud_feats[d['filename']] = (X, y)
        elapsed = time.time() - t0
        if (i + 1) % 10 == 0 or i == 0:
            log.info(f"  [{i+1}/{len(all_clouds)}] {d['filename']:<35} "
                     f"{len(y):>7,} pts  {elapsed:.1f}s")

    log.info(f"Features prontas: {len(cloud_feats)} nuvens")

    # ── 3. Baseline: SF puro ──────────────────────────────────────────────────
    log.info("\n" + "─"*60)
    log.info("BASELINE — Scalar Field puro (negado: SF baixo → crack)")
    aurocs_sf = []
    for d in labeled:
        sf  = d['features'][:, 9]
        lbl = d['labels']
        if lbl is None or lbl.sum() < 5:
            continue
        try:
            aurocs_sf.append(roc_auc_score(lbl, -sf))
        except Exception:
            pass
    log.info(f"SF puro: AUROC médio = {float(np.mean(aurocs_sf)):.4f} "
             f"(n={len(aurocs_sf)} nuvens)")

    # ── 4. LOCO-CV ────────────────────────────────────────────────────────────
    log.info("\n" + "─"*60)
    log.info("LEAVE-ONE-CLOUD-OUT CROSS-VALIDATION")
    log.info("─"*60)

    t_start = time.time()
    results = run_loco_cv(cloud_feats, labeled, normals)
    elapsed = (time.time() - t_start) / 60

    # ── 5. Sumário ────────────────────────────────────────────────────────────
    valid = [r for r in results if not np.isnan(r['auroc'])]

    aurocs = [r['auroc']    for r in valid]
    f1s    = [r['f1']       for r in valid]
    f1_dbs = [r['f1_dbscan'] for r in valid]
    aps    = [r['ap']       for r in valid]

    print("\n" + "="*70)
    print("  RESULTADOS — XGB CRACK (LOCO-CV)")
    print("="*70)
    print(f"  Folds avaliados : {len(valid)}/{len(labeled)}")
    print(f"  Tempo total     : {elapsed:.1f} min")
    print()
    print(f"  {'Métrica':<18} {'Média':>8}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print(f"  {'─'*52}")
    for name, vals in [('AUROC', aurocs), ('F1', f1s), ('F1+DBSCAN', f1_dbs), ('AP', aps)]:
        arr = np.array(vals)
        print(f"  {name:<18} {arr.mean():>8.4f}  {arr.std():>7.4f}  "
              f"{arr.min():>7.4f}  {arr.max():>7.4f}")
    print(f"\n  Baseline SF puro : AUROC = {float(np.mean(aurocs_sf)):.4f}")
    print(f"  SF-GMM (referência): AUROC ≈ 0.887")
    print("="*70)

    # Clouds difíceis
    hard = ['avaria_17.ply', 'avaria_33.ply', 'avaria_21.ply']
    hard_rows = [r for r in results if r['filename'] in hard]
    if hard_rows:
        print("\n  Clouds difíceis (baseline SF: ≤0.61):")
        for r in hard_rows:
            print(f"    {r['filename']:<35} AUROC={r['auroc']:.4f}  F1={r['f1']:.4f}")

    # ── 6. Salva CSV ──────────────────────────────────────────────────────────
    csv_path = os.path.join(run_dir, f'loco_results_{ts}.csv')
    if results:
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        log.info(f"\nResultados salvos: {csv_path}")

    # ── 7. Modelo final (treino em todos os dados) ────────────────────────────
    log.info("\nTreinando modelo final em todos os dados...")
    all_X = np.concatenate([cloud_feats[d['filename']][0] for d in labeled + normals])
    all_y = np.concatenate([cloud_feats[d['filename']][1] for d in labeled + normals])

    n_pos = int(all_y.sum())
    n_neg = int((all_y == 0).sum())
    spw   = max(1.0, n_neg / max(n_pos, 1))

    final_model = make_xgb(spw)
    final_model.fit(all_X, all_y)

    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(final_model, f)
    log.info(f"Modelo final salvo: {MODEL_PATH}")

    # Feature importances
    importances = final_model.feature_importances_
    log.info("\nImportância das features (gain):")
    for name, imp in sorted(zip(FEATURE_NAMES, importances),
                            key=lambda x: -x[1]):
        bar = '█' * int(imp * 40)
        log.info(f"  {name:<18} {imp:.4f}  {bar}")

    # ── 8. PLY coloridos (modelo final nas nuvens avaria) ────────────────────
    log.info("\nSalvando PLY coloridos...")
    run_ply = os.path.join(PLY_XGB, f'run_{ts}')
    os.makedirs(run_ply, exist_ok=True)

    for d in labeled:
        X, y = cloud_feats[d['filename']]
        y_prob = final_model.predict_proba(X)[:, 1]
        thr_row = next((r['thr'] for r in results if r['filename'] == d['filename']), 0.5)
        pred = dbscan_postprocess(
            d['features'][:, :3],
            (y_prob >= thr_row).astype(np.int32),
        )
        save_colored_ply(
            xyz=d['features'][:, :3],
            rgb_orig=d['features'][:, 3:6],
            pred_labels=pred,
            path=os.path.join(run_ply, d['filename'].replace('.ply', '_xgb.ply')),
        )

    log.info(f"PLY salvos em: {run_ply}")
    return final_model, results


if __name__ == '__main__':
    main()
