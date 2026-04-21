"""
test_rank_norm.py — Confirma resultado esperado após fixes de 04/04/2026.

Fixes aplicados:
  - Rank normalization revertida (contraproducente — score já perfeito)
  - apply_spatial_coherence desativada (removia 64.4% dos TPs)

Carrega predictions_04042026_0400.csv (scores pré-filtro) e simula o
pipeline sem spatial_coherence para confirmar que o próximo treino
deve reportar recall≈0.9994 e F1≈0.9996.

Uso:
    cd ~/projects/TCC
    uv run python src/test_rank_norm.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import (
    precision_recall_fscore_support, jaccard_score,
    roc_auc_score, roc_curve, precision_recall_curve
)

# ── Configuração ──────────────────────────────────────────────────────────────
CSV_PRED = os.path.join(os.path.dirname(__file__), '..', 'visualizations_ts',
                        'predictions_04042026_0400.csv')
CSV_XYZ  = os.path.join(os.path.dirname(__file__), '..', 'visualizations_ts',
                        'predictions_04042026_0400.csv')   # mesmo arquivo tem xyz? não — ver abaixo

# ── Funções ───────────────────────────────────────────────────────────────────

def rank_norm(v: np.ndarray) -> np.ndarray:
    """Rank-based normalization (PointCore 2024)."""
    n = len(v)
    if n == 0:
        return v
    ranks = np.argsort(np.argsort(v))
    return ranks.astype(np.float32) / (n - 1 + 1e-8)


def metrics(gt: np.ndarray, pred: np.ndarray, scores: np.ndarray = None) -> dict:
    prec, rec, f1, _ = precision_recall_fscore_support(
        gt, pred, average='binary', zero_division=0)
    iou = jaccard_score(gt, pred, zero_division=0)
    auroc = roc_auc_score(gt, scores) if scores is not None else float('nan')
    return dict(precision=prec, recall=rec, f1=f1, iou=iou, auroc=auroc)


def best_gmean_thr(gt, scores):
    fpr, tpr, thrs = roc_curve(gt, scores)
    tnr   = 1.0 - fpr
    gmean = np.sqrt(tpr * tnr)
    return float(thrs[np.argmax(gmean[:-1])])


def best_f1_thr(gt, scores):
    prec, rec, thrs = precision_recall_curve(gt, scores)
    f1s = 2 * prec * rec / (prec + rec + 1e-8)
    return float(thrs[np.argmax(f1s[:-1])])


def spatial_coherence_filter(pred: np.ndarray, xyz: np.ndarray,
                              min_neighbors: int, k: int) -> np.ndarray:
    """Reimplementação local de apply_spatial_coherence para testar parâmetros."""
    pred = pred.copy()
    if pred.sum() == 0 or len(pred) < k + 1:
        return pred
    tree   = cKDTree(xyz)
    pos_idx = np.where(pred == 1)[0]
    if len(pos_idx) == 0:
        return pred
    dists, idxs = tree.query(xyz[pos_idx], k=k + 1)
    idxs = idxs[:, 1:]   # exclui o próprio ponto
    for i, orig_i in enumerate(pos_idx):
        neighbor_preds = pred[idxs[i]].sum()
        if neighbor_preds < min_neighbors:
            pred[orig_i] = 0
    return pred


# ── Carregar CSV ──────────────────────────────────────────────────────────────
print(f"\nCarregando: {os.path.abspath(CSV_PRED)}")
df = pd.read_csv(CSV_PRED)
print(f"Pontos totais : {len(df):,}")
print(f"Nuvens únicas : {df['filename'].nunique()}")
print(f"Colunas       : {list(df.columns)}")
print(f"Positivos GT  : {df['gt'].sum():,} ({df['gt'].mean()*100:.1f}%)")

# Apenas nuvens COM rachadura para avaliação (igual ao pipeline original)
crack_files = df[df['gt'] == 1]['filename'].unique()
df_crack    = df[df['filename'].isin(crack_files)].copy()
print(f"\nNuvens com rachadura: {len(crack_files)}")
print(f"Pontos nessas nuvens: {len(df_crack):,}")
print(f"Pontos crack:         {df_crack['gt'].sum():,} "
      f"({df_crack['gt'].mean()*100:.1f}%)")

gt_all     = df_crack['gt'].values
score_orig = df_crack['score'].values
pred_saved = df_crack['pred'].values   # já tem spatial coherence aplicado

# ── Seção 1: Score direto SEM filtro espacial ─────────────────────────────────
print("\n" + "=" * 65)
print("1. SCORE ANTES DO FILTRO ESPACIAL")
print("   (threshold direto em score, sem spatial_coherence)")
print("=" * 65)

thr_gmean = best_gmean_thr(gt_all, score_orig)
thr_f1    = best_f1_thr(gt_all, score_orig)
score_rn  = df_crack.groupby('filename')['score'].transform(rank_norm).values

print(f"\n  Estratégia    | Threshold | Precision | Recall  |   F1    |  IoU")
print(f"  {'─'*68}")

for label, scores, thr in [
    ("Min-Max G-mean", score_orig, thr_gmean),
    ("Min-Max F1    ", score_orig, thr_f1),
    ("Rank-Norm Gmean", score_rn, best_gmean_thr(gt_all, score_rn)),
    ("Rank-Norm F1  ", score_rn, best_f1_thr(gt_all, score_rn)),
    ("Pred salvo    ", None,      None),   # pred do CSV (com spatial coherence)
]:
    if label == "Pred salvo    ":
        pred = pred_saved
        sc   = score_orig
    else:
        pred = (scores >= thr).astype(int)
        sc   = scores
    m = metrics(gt_all, pred, sc)
    thr_str = f"{thr:.4f}" if thr is not None else "  CSV "
    print(f"  {label:16s}| {thr_str:>9} | {m['precision']:>9.4f} "
          f"| {m['recall']:>7.4f} | {m['f1']:>7.4f} | {m['iou']:>6.4f}")

# ── Seção 2: Spatial coherence — sweep de min_neighbors ──────────────────────
print("\n" + "=" * 65)
print("2. IMPACTO DO FILTRO ESPACIAL (min_neighbors × k)")
print("   Threshold fixo: G-mean (min-max)")
print("=" * 65)

# Verificar se temos XYZ no CSV
has_xyz = 'x' in df.columns

if not has_xyz:
    print("\n  AVISO: CSV não tem colunas x/y/z — não é possível reaplicar")
    print("  o filtro espacial aqui. Lendo data/ diretamente seria necessário.")
    print("\n  Mas os dados do pred salvo confirmam o efeito:")
    m_sem  = metrics(gt_all, (score_orig >= thr_gmean).astype(int), score_orig)
    m_com  = metrics(gt_all, pred_saved, score_orig)
    print(f"\n  {'':20s} | {'Precision':>9} | {'Recall':>7} | {'F1':>7} | {'IoU':>6}")
    print(f"  {'─'*60}")
    print(f"  {'Sem spatial_coherence':20s} | {m_sem['precision']:>9.4f} "
          f"| {m_sem['recall']:>7.4f} | {m_sem['f1']:>7.4f} | {m_sem['iou']:>6.4f}")
    print(f"  {'Com spatial_coherence':20s} | {m_com['precision']:>9.4f} "
          f"| {m_com['recall']:>7.4f} | {m_com['f1']:>7.4f} | {m_com['iou']:>6.4f}")

    delta_rec = m_com['recall'] - m_sem['recall']
    delta_pre = m_com['precision'] - m_sem['precision']
    print(f"\n  Delta recall    : {delta_rec:+.4f}  ← spatial_coherence REMOVE esses TPs")
    print(f"  Delta precision : {delta_pre:+.4f}  ← spatial_coherence REMOVE esses FPs")
    frac_removed = 1.0 - (m_com['recall'] / (m_sem['recall'] + 1e-9))
    print(f"\n  ► {frac_removed*100:.1f}% dos crack detectados pelo score são removidos "
          f"pelo filtro espacial!")

# ── Seção 3: Análise de distribuição dos scores ───────────────────────────────
print("\n" + "=" * 65)
print("3. DISTRIBUIÇÃO DOS SCORES POR CLASSE (min-max, crack clouds only)")
print("=" * 65)

for label, mask in [("Normal (gt=0)", gt_all == 0), ("Crack  (gt=1)", gt_all == 1)]:
    s = score_orig[mask]
    p5, p25, p50, p75, p95 = np.percentile(s, [5, 25, 50, 75, 95])
    print(f"\n  {label}: n={mask.sum():,}")
    print(f"    P5={p5:.3f}  P25={p25:.3f}  P50={p50:.3f}  "
          f"P75={p75:.3f}  P95={p95:.3f}  Max={s.max():.3f}")

# Fração de crack com score acima de vários thresholds
print("\n  Fração de pontos CRACK acima de cada threshold:")
crack_scores = score_orig[gt_all == 1]
for thr in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6]:
    frac = (crack_scores >= thr).mean()
    bar  = "█" * int(frac * 30)
    print(f"    thr={thr:.2f}: {frac*100:5.1f}%  {bar}")

# ── Conclusão ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CONCLUSÃO")
print("=" * 65)
m_bruto = metrics(gt_all, (score_orig >= thr_gmean).astype(int), score_orig)
m_final = metrics(gt_all, pred_saved, score_orig)
print(f"\n  Score bruto (sem filtro): Recall={m_bruto['recall']:.4f}  "
      f"F1={m_bruto['f1']:.4f}")
print(f"  Pred final (com filtro) : Recall={m_final['recall']:.4f}  "
      f"F1={m_final['f1']:.4f}")
print(f"\n  O score gerado {'É' if m_bruto['recall'] > 0.8 else 'NÃO é'} bom "
      f"(recall {m_bruto['recall']:.1%} antes do filtro).")
print(f"  O problema {'É' if m_final['recall'] < m_bruto['recall'] * 0.6 else 'NÃO é'} "
      f"o filtro espacial apply_spatial_coherence.")
print(f"\n  Próximo passo: ajustar min_neighbors (atual=3) e k (atual=20)")
print(f"  em evaluation.py:661 e teacher_student_v1.py:932.")
