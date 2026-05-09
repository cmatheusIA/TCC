# ============================================================================
# TEACHER-STUDENT v1.4 — Z-score + Gate SF + Threshold GMM por nuvem + DBSCAN
# ============================================================================
# Delta vs v1.3: filtro DBSCAN com aspect ratio após o threshold.
#
# Problema corrigido:
#   Mesmo com z-score + gate SF + threshold por nuvem, podem restar FPs
#   isolados: manchas de sujeira, sombras pontuais, variações de textura.
#   Rachaduras são estruturas LINEARES e CONTÍNUAS. Clusters esféricos ou
#   pequenos demais não são rachaduras.
#
# Lógica:
#   1. DBSCAN em XYZ dos pontos preditos como crack (eps, min_samples)
#   2. Para cada componente conectado:
#      a. Remove se n_pontos < min_pts
#      b. PCA nos XYZ do componente → eigenvalues λ1 ≥ λ2 ≥ λ3
#         aspect_ratio = √(λ1 / λ2)
#      c. Remove se aspect_ratio < min_aspect (cluster esférico = ruído)
#   3. Pontos em componentes removidos → pred_label = 0
#
# Parâmetros:
#   eps=0.02        — raio de vizinhança em XYZ normalizado (esfera unitária)
#   min_pts=15      — componentes menores são ruído
#   min_aspect=2.5  — razão mínima λ1/λ2 para ser crack (linear)
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import copy, csv
from datetime import datetime

import numpy as np
import torch
from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree

from teacher_student_v1 import (
    TeacherStudentModel, NormalMemoryBank, build_model,
    compute_crack_sf_interval, apply_scalar_field_gate,
    evaluate, save_colored_ply, setup_logging,
)
from utils.config import *
from utils.data import load_folder, split_dataset
from utils.evaluation import ScalarFieldGMM

VERSION   = 'v1_4'
RESULTS   = f'{BASE_PATH}/results_ts_{VERSION}'
VIS_PATH  = f'{BASE_PATH}/visualizations_ts_{VERSION}'
PLY_PATH  = f'{RESULTS}/ply'
MODELS_V1 = f'{BASE_PATH}/models_ts'
log       = setup_logging(f'{BASE_PATH}/logs_ts_{VERSION}')

# Hiperparâmetros DBSCAN
DBSCAN_EPS    = 0.02   # raio em XYZ normalizado
DBSCAN_MINPTS = 15     # mínimo de pontos por componente
MIN_ASPECT    = 2.5    # aspect ratio mínimo (√(λ1/λ2))


# ============================================================================
# COMPUTE ANOMALY SCORES — z-score + gate SF  (idêntico ao v1.2/v1.3)
# ============================================================================

def compute_anomaly_scores(model, data_list, device,
                           sf_min: float, sf_max: float,
                           memory_bank=None,
                           alpha_dist=0.65, alpha_bank=0.3):
    model.eval()
    results = []

    for d in data_list:
        x      = torch.tensor(d['features'], dtype=torch.float32).to(device)
        labels = d['labels']

        raw_score = model.anomaly_score_per_point(x)

        if memory_bank is not None and memory_bank.bank is not None:
            btn, _    = model.teacher_features(x)
            bank_dist = memory_bank.score(btn.cpu(), k=5)
            lo, hi    = bank_dist.min(), bank_dist.max()
            bank_norm = (bank_dist - lo) / (hi - lo + 1e-8)
            lo2, hi2  = raw_score.min(), raw_score.max()
            dist_norm = (raw_score - lo2) / (hi2 - lo2 + 1e-8)
            combined  = alpha_dist * dist_norm + alpha_bank * bank_norm
        else:
            combined = raw_score

        mu    = combined.mean()
        sigma = combined.std() + 1e-8
        z     = (combined - mu) / sigma
        score = np.clip(z, 0, None)
        score = score / (score.max() + 1e-8)

        sf       = d['features'][:, 9]
        in_range = ((sf >= sf_min) & (sf <= sf_max)).astype(np.float32)
        score    = score * in_range
        score    = score / (score.max() + 1e-8)

        results.append({
            'filename'    : d['filename'],
            'has_crack'   : d['has_crack'],
            'score'       : score,
            'gt_labels'   : labels,
            'n_points'    : len(labels) if labels is not None else len(score),
            'xyz'         : d['features'][:, :3],
            'rgb'         : d['features'][:, 3:6],
            'scalar_field': sf,
        })

    return results


# ============================================================================
# THRESHOLD GMM POR NUVEM  (idêntico ao v1.3)
# ============================================================================

def apply_per_cloud_gmm_threshold(results: list,
                                   fallback_pctl: float = 90.0) -> list:
    for r in results:
        score = r['score']
        gmm   = ScalarFieldGMM(score).fit()
        thr   = gmm.threshold if gmm.modality == 'bimodal' \
                else float(np.percentile(score, fallback_pctl))
        r['pred_labels']   = (score >= thr).astype(np.int64)
        r['per_cloud_thr'] = thr
    return results


# ============================================================================
# FILTRO DBSCAN COM ASPECT RATIO
# ============================================================================

def _aspect_ratio(pts: np.ndarray) -> float:
    """√(λ1/λ2) dos 3 eigenvalues da covariância de pts (N,3)."""
    if len(pts) < 3:
        return 0.0
    try:
        cov = np.cov(pts.T)
        ev  = np.linalg.eigvalsh(cov)
        ev  = np.sort(ev[ev > 1e-10])[::-1]
        if len(ev) >= 2:
            return float(np.sqrt(ev[0] / (ev[1] + 1e-10)))
    except Exception:
        pass
    return 0.0


def apply_dbscan_aspect_filter(
    results: list,
    eps: float       = DBSCAN_EPS,
    min_pts: int     = DBSCAN_MINPTS,
    min_aspect: float = MIN_ASPECT,
) -> list:
    """
    Para cada nuvem, aplica DBSCAN nos pontos preditos como crack e remove:
      - Componentes com < min_pts pontos
      - Componentes com aspect_ratio < min_aspect (forma esférica = ruído)

    Rachaduras reais: componentes elongados (aspect_ratio alto), densos,
    contínuos. Ruído: clusters pequenos e esféricos.
    """
    for r in results:
        pred = r['pred_labels']
        xyz  = r['xyz']

        crack_idx = np.where(pred == 1)[0]
        if len(crack_idx) < min_pts:
            r['pred_labels'] = np.zeros_like(pred)
            continue

        crack_xyz = xyz[crack_idx]
        db        = DBSCAN(eps=eps, min_samples=3).fit(crack_xyz)
        comp_ids  = db.labels_

        result    = np.zeros_like(pred)
        n_kept = 0
        n_removed = 0

        for cid in set(comp_ids) - {-1}:
            members = np.where(comp_ids == cid)[0]

            if len(members) < min_pts:
                n_removed += len(members)
                continue

            ar = _aspect_ratio(crack_xyz[members])
            if ar < min_aspect:
                n_removed += len(members)
                continue

            result[crack_idx[members]] = 1
            n_kept += len(members)

        r['pred_labels'] = result
        log.debug(f"  {r['filename']}: {len(crack_idx)} → {n_kept} "
                  f"(removidos {n_removed} FP por DBSCAN/aspecto)")

    return results


# ============================================================================
# PIPELINE DE AVALIAÇÃO
# ============================================================================

def run_eval(ts: str):
    for p in [RESULTS, VIS_PATH, PLY_PATH]:
        os.makedirs(p, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device} | Versão: {VERSION}")
    log.info(f"DBSCAN: eps={DBSCAN_EPS}  min_pts={DBSCAN_MINPTS}  "
             f"min_aspect={MIN_ASPECT}")

    log.info("Carregando dados...")
    train_all = load_folder(DATA_TRAIN)
    test_all  = load_folder(DATA_TEST)
    all_data  = train_all + test_all
    train_list, _, eval_list = split_dataset(all_data)

    model = build_model(device)
    best_path = os.path.join(MODELS_V1, 'best_student.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.student.load_state_dict(ckpt['student'])
        log.info(f"Checkpoint v1 carregado: epoch {ckpt.get('epoch',0)+1}")
    else:
        log.error(f"Checkpoint não encontrado: {best_path}")
        return

    memory_bank = NormalMemoryBank(max_size=100_000, subsample_k=1_000)
    bank_path   = os.path.join(MODELS_V1, 'ts_memory_bank_clean.pt')
    if os.path.exists(bank_path):
        memory_bank.load(bank_path)
        log.info(f"Memory bank carregado: {len(memory_bank.bank):,} features")

    sf_min, sf_max = compute_crack_sf_interval(train_list)
    log.info(f"Intervalo crack SF: [{sf_min:.3f}, {sf_max:.3f}]")

    log.info("\nComputando anomaly scores (z-score + gate SF supervisionado)...")
    eval_data = eval_list if eval_list else train_list
    results   = compute_anomaly_scores(model, eval_data, device,
                                       sf_min=sf_min, sf_max=sf_max,
                                       memory_bank=memory_bank,
                                       alpha_dist=0.65, alpha_bank=0.3)

    log.info("Aplicando threshold GMM por nuvem...")
    results = apply_per_cloud_gmm_threshold(results, fallback_pctl=90.0)
    results = apply_scalar_field_gate(results, sf_min, sf_max)

    log.info("Aplicando filtro DBSCAN com aspect ratio...")
    results = apply_dbscan_aspect_filter(
        results, eps=DBSCAN_EPS, min_pts=DBSCAN_MINPTS, min_aspect=MIN_ASPECT
    )

    m = evaluate(results)
    log.info("\n" + "="*65)
    log.info(f"RESULTADO — {VERSION} (z-score + gate SF + GMM/nuvem + DBSCAN)")
    log.info("="*65)
    log.info(f"  P={m.get('precision',0):.4f}  R={m.get('recall',0):.4f}  "
             f"F1={m.get('f1',0):.4f}  IoU={m.get('iou',0):.4f}  "
             f"AUROC={m.get('auroc',0):.4f}")

    row = {
        'estrategia'   : 'z-score+gate+gmm+dbscan',
        'threshold'    : 'adaptativo',
        'precision'    : round(m.get('precision',         0), 4),
        'recall'       : round(m.get('recall',            0), 4),
        'f1'           : round(m.get('f1',                0), 4),
        'iou'          : round(m.get('iou',               0), 4),
        'auroc'        : round(m.get('auroc',             0), 4),
        'avg_precision': round(m.get('average_precision', 0), 4),
    }
    csv_path = os.path.join(RESULTS, f'comparacao_thresholds_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        w.writeheader(); w.writerow(row)
    log.info(f"\nResultados salvos: {csv_path}")

    for r in results:
        if not r['has_crack']:
            continue
        save_colored_ply(
            xyz=r['xyz'], rgb_orig=r['rgb'],
            pred_labels=r['pred_labels'],
            path=os.path.join(PLY_PATH, r['filename'].replace('.ply', f'_{VERSION}.ply')),
        )
    log.info(f"PLY salvos em: {PLY_PATH}")
    return row


def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print(f"  TEACHER-STUDENT {VERSION}")
    print(f"  Z-score + Gate SF + Threshold GMM/nuvem + DBSCAN aspect ratio")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)
    run_eval(ts)


if __name__ == '__main__':
    main()
