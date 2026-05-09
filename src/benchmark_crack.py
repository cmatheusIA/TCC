"""
Benchmark binário — compara os 5 modelos no mesmo protocolo LOCO.
Lê CSVs de resultado de cada modelo e gera tabela comparativa.

Modelos:
  gmm_scalar    → results_gmm_scalar/loco_*.csv
  dgcnn_bigan   → results_crack_gan/loco_*.csv
  scalar_mae    → results_scalar_mae/loco_*.csv
  scalar_gat    → results_scalar_gat/loco_*.csv
  ptv3_binary   → results_ptv3_binary/loco_*.csv

Baseline:
  xgboost       → AUROC=0.930, F1=0.595 (hardcoded — medido em 23/04/2026)
"""
import sys, os, glob, csv
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from utils.config import BASE_PATH


MODELS = {
    'gmm_scalar'  : f'{BASE_PATH}/results_gmm_scalar',
    'gmm_ae'      : f'{BASE_PATH}/results_gmm_ae',
    'gmm_vae'     : f'{BASE_PATH}/results_gmm_vae',
    'dgcnn_bigan' : f'{BASE_PATH}/results_crack_gan',
    'scalar_mae'  : f'{BASE_PATH}/results_scalar_mae',
    'scalar_gat'  : f'{BASE_PATH}/results_scalar_gat',
    'ptv3_binary' : f'{BASE_PATH}/results_ptv3_binary',
}

XGBOOST_BASELINE = {'model': 'xgboost', 'auroc': 0.930, 'f1': 0.595, 'ap': None}


def load_latest_csv(results_dir: str) -> list[dict] | None:
    """Carrega o CSV mais recente em results_dir."""
    pattern = os.path.join(results_dir, 'loco_*.csv')
    files   = sorted(glob.glob(pattern))
    if not files:
        return None
    with open(files[-1], newline='') as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict]) -> dict:
    aurocs = [float(r['auroc']) for r in rows if r.get('auroc')]
    f1s    = [float(r['f1'])    for r in rows if r.get('f1')]
    aps    = [float(r['ap'])    for r in rows if r.get('ap')]
    return {
        'auroc_mean': round(np.mean(aurocs), 4) if aurocs else float('nan'),
        'auroc_std' : round(np.std(aurocs),  4) if aurocs else float('nan'),
        'f1_mean'   : round(np.mean(f1s),    4) if f1s    else float('nan'),
        'f1_std'    : round(np.std(f1s),     4) if f1s    else float('nan'),
        'ap_mean'   : round(np.mean(aps),    4) if aps    else float('nan'),
        'n_folds'   : len(aurocs),
    }


def main():
    print(f"\n{'='*72}")
    print(f"  BENCHMARK BINÁRIO — Detecção de Rachaduras (LOCO, n=35)")
    print(f"  Baseline XGBoost: AUROC=0.930  F1=0.595")
    print(f"{'='*72}")
    print(f"  {'Modelo':<18} {'AUROC':>8} {'±':>5} {'F1':>8} {'±':>5} {'AP':>8} {'Folds':>6}")
    print(f"  {'-'*65}")

    # Baseline
    b = XGBOOST_BASELINE
    print(f"  {'xgboost (baseline)':<18} {b['auroc']:>8.4f} {'—':>5} "
          f"{b['f1']:>8.4f} {'—':>5} {'—':>8} {'—':>6}")

    summaries = {}
    for name, rdir in MODELS.items():
        rows = load_latest_csv(rdir)
        if rows is None:
            print(f"  {name:<18} {'(sem resultados)':>40}")
            continue
        s = summarize(rows)
        summaries[name] = s
        auroc_flag = ' ★' if s['auroc_mean'] > 0.930 else ''
        f1_flag    = ' ★' if s['f1_mean']    > 0.595 else ''
        print(f"  {name:<18} {s['auroc_mean']:>8.4f} {s['auroc_std']:>5.4f} "
              f"{s['f1_mean']:>8.4f}{f1_flag} {s['f1_std']:>5.4f} "
              f"{s['ap_mean']:>8.4f}{auroc_flag} {s['n_folds']:>6}")

    print(f"{'='*72}")
    print("  ★ = supera baseline XGBoost\n")

    # Salvar CSV consolidado
    out_csv = f'{BASE_PATH}/results_benchmark/summary_binary.csv'
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = ['model', 'auroc_mean', 'auroc_std', 'f1_mean', 'f1_std',
                  'ap_mean', 'n_folds']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'model': 'xgboost', 'auroc_mean': 0.930, 'f1_mean': 0.595,
                         'auroc_std': 0, 'f1_std': 0, 'ap_mean': 'N/A', 'n_folds': 35})
        for name, s in summaries.items():
            writer.writerow({'model': name, **s})
    print(f"  CSV consolidado: {out_csv}")


if __name__ == '__main__':
    main()
