# ============================================================================
# TEACHER-STUDENT v1.2 — Z-score por nuvem + Gate supervisionado de SF
# ============================================================================
# Delta vs v1.1: zera o score em pontos cujo SF está fora do intervalo
# [sf_min, sf_max] aprendido dos labels de treino.
#
# Problema corrigido:
#   Mesmo com z-score, pontos com SF fora do intervalo observado em rachaduras
#   reais nunca deveriam ter score alto — não são rachadura por definição.
#   O gate usa o intervalo supervisionado do treino, não GMM por nuvem.
#
# Por que supervisionado e não GMM por nuvem?
#   O GMM identifica o componente MINORITÁRIO como crack — mas pode errar
#   (avaria_33: identificou componente com mu=0.62 em vez de mu=0.12).
#   O intervalo supervisionado [sf_min, sf_max] cobre TODOS os pontos label=1
#   de todas as nuvens de treino, incluindo casos invertidos (avaria_36 com
#   SF alto, avaria_17 com SF misto). É um hard gate baseado em evidência real.
#
# Lógica:
#   z_score    = clip((raw - µ) / σ, 0) / max(...)   [por nuvem]
#   sf_mask[i] = 1 se sf_min ≤ SF[i] ≤ sf_max, 0 caso contrário
#   score_final = z_score * sf_mask   (re-normalizado)
#
# sf_min e sf_max são passados como argumento — calculados uma vez em
# compute_crack_sf_interval(train_list) antes de chamar esta função.
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import copy, csv
from datetime import datetime

import numpy as np
import torch

from teacher_student_v1 import (
    TeacherStudentModel, NormalMemoryBank, build_model,
    compute_crack_sf_interval, apply_scalar_field_gate,
    fit_gmm_threshold, apply_threshold, evaluate, save_colored_ply,
    calibrate_threshold_f1, calibrate_threshold_gmean, calibrate_threshold_fbeta,
    setup_logging,
)
from utils.config import *
from utils.data import load_folder, split_dataset

VERSION   = 'v1_2'
RESULTS   = f'{BASE_PATH}/results_ts_{VERSION}'
VIS_PATH  = f'{BASE_PATH}/visualizations_ts_{VERSION}'
PLY_PATH  = f'{RESULTS}/ply'
MODELS_V1 = f'{BASE_PATH}/models_ts'
log       = setup_logging(f'{BASE_PATH}/logs_ts_{VERSION}')


# ============================================================================
# COMPUTE ANOMALY SCORES — z-score + gate SF
# ============================================================================

def compute_anomaly_scores(model, data_list, device,
                           sf_min: float, sf_max: float,
                           memory_bank=None,
                           alpha_dist=0.65, alpha_bank=0.3):
    """
    v1.1 (z-score) + gate supervisionado de SF.

    Gate: score[i] = 0 se SF[i] < sf_min ou SF[i] > sf_max
    [sf_min, sf_max] = intervalo aprendido dos pontos label=1 no treino.

    Cobre todos os casos corretamente:
    - Cracks escuros (maioria): sf_min baixo, sf_max médio
    - Cracks claros (avaria_36, 40): sf_max alto → esses pontos NÃO são zerados
    - Clouds difíceis (avaria_17): SF de crack cobre range amplo → poucos zeros
    - GMM com componente errado (avaria_33): não afeta — usa intervalo global
    """
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

        # ── Z-score por nuvem ───────────────────────────────────────────────
        mu    = combined.mean()
        sigma = combined.std() + 1e-8
        z     = (combined - mu) / sigma
        score = np.clip(z, 0, None)
        score = score / (score.max() + 1e-8)

        # ── Gate supervisionado: zera fora do intervalo SF de rachadura ─────
        # Intervalo aprendido de TODOS os pontos label=1 no treino.
        # Pontos fora de [sf_min, sf_max] jamais foram observados como rachadura
        # → score = 0 (hard gate, sem risco de falsa supressão de cracks reais).
        sf      = d['features'][:, 9]
        in_range = ((sf >= sf_min) & (sf <= sf_max)).astype(np.float32)
        score   = score * in_range
        score   = score / (score.max() + 1e-8)

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
# PIPELINE DE AVALIAÇÃO
# ============================================================================

def run_eval(ts: str):
    for p in [RESULTS, VIS_PATH, PLY_PATH]:
        os.makedirs(p, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device} | Versão: {VERSION}")

    log.info("Carregando dados...")
    train_all = load_folder(DATA_TRAIN)
    test_all  = load_folder(DATA_TEST)
    all_data  = train_all + test_all
    train_list, labeled_list, eval_list = split_dataset(all_data)

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

    # Intervalo supervisionado: aprendido dos labels reais do treino
    sf_min, sf_max = compute_crack_sf_interval(train_list)
    log.info(f"Intervalo crack SF: [{sf_min:.3f}, {sf_max:.3f}]")

    log.info("\nComputando anomaly scores (z-score + gate SF supervisionado)...")
    eval_data  = eval_list if eval_list else train_list
    results    = compute_anomaly_scores(model, eval_data, device,
                                        sf_min=sf_min, sf_max=sf_max,
                                        memory_bank=memory_bank,
                                        alpha_dist=0.65, alpha_bank=0.3)
    normal_ref = compute_anomaly_scores(model, train_list[:20], device,
                                        sf_min=sf_min, sf_max=sf_max,
                                        memory_bank=memory_bank)

    thr, gmm_info  = fit_gmm_threshold(results, normal_results=normal_ref)
    log.info(f"Threshold ({gmm_info['method']}): {thr:.4f}")

    results = apply_threshold(results, thr)
    results = apply_scalar_field_gate(results, sf_min, sf_max)

    strategies = {
        'F1'         : calibrate_threshold_f1(results),
        'G-mean'     : calibrate_threshold_gmean(results),
        'F-beta(0.5)': calibrate_threshold_fbeta(results, beta=0.5),
    }

    comparison = []
    log.info("\n" + "="*65)
    log.info(f"COMPARAÇÃO — {VERSION}")
    log.info("="*65)

    for name, thr_s in strategies.items():
        res_s = apply_threshold(copy.deepcopy(results), thr_s)
        res_s = apply_scalar_field_gate(res_s, sf_min, sf_max)
        m_s   = evaluate(res_s)
        row   = {
            'estrategia'   : name,
            'threshold'    : round(thr_s, 6),
            'precision'    : round(m_s.get('precision',         0), 4),
            'recall'       : round(m_s.get('recall',            0), 4),
            'f1'           : round(m_s.get('f1',                0), 4),
            'iou'          : round(m_s.get('iou',               0), 4),
            'auroc'        : round(m_s.get('auroc',             0), 4),
            'avg_precision': round(m_s.get('average_precision', 0), 4),
        }
        comparison.append(row)
        log.info(f"\n  [{name}]  thr={thr_s:.4f}")
        log.info(f"    P={row['precision']:.4f}  R={row['recall']:.4f}  "
                 f"F1={row['f1']:.4f}  IoU={row['iou']:.4f}  AUROC={row['auroc']:.4f}")

    csv_path = os.path.join(RESULTS, f'comparacao_thresholds_{ts}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=comparison[0].keys())
        w.writeheader(); w.writerows(comparison)
    log.info(f"\nResultados salvos: {csv_path}")

    gmean_thr   = calibrate_threshold_gmean(results)
    results_out = apply_threshold(copy.deepcopy(results), gmean_thr)
    results_out = apply_scalar_field_gate(results_out, sf_min, sf_max)
    for r in results_out:
        if not r['has_crack']:
            continue
        save_colored_ply(
            xyz=r['xyz'], rgb_orig=r['rgb'],
            pred_labels=r['pred_labels'],
            path=os.path.join(PLY_PATH, r['filename'].replace('.ply', f'_{VERSION}.ply')),
        )
    log.info(f"PLY salvos em: {PLY_PATH}")
    return comparison


def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print(f"  TEACHER-STUDENT {VERSION} — Z-score + Gate SF")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)
    run_eval(ts)


if __name__ == '__main__':
    main()
