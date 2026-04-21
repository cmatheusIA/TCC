# ============================================================================
# DIAGNÓSTICO — Hipótese do Spatial Refinement GNN
# ============================================================================
#
# Hipótese a testar:
#   Pontos de rachadura real têm vizinhos (em XYZ) com SF-GMM score alto.
#   Falsos positivos do SF-GMM têm vizinhos com SF-GMM score baixo.
#   → O GNN, ao agregar informação da vizinhança, suprime FPs e reforça TPs.
#
# Três fases:
#
#   Fase 1 — Qualidade dos pseudo-labels SF-GMM:
#     Dos pontos com SF-GMM prob > 0.85, qual % são realmente rachadura?
#     Dos pontos com SF-GMM prob < 0.10, qual % são realmente normais?
#     → Confirma que os pseudo-labels têm qualidade suficiente para treinar.
#
#   Fase 2 — Conectividade de vizinhança (hipótese principal):
#     Para pontos de rachadura (GT=1): média do SF-GMM score dos k-vizinhos XYZ
#     Para FPs (SF-GMM alto mas GT=0): média do SF-GMM score dos k-vizinhos XYZ
#     → Se crack_neighbor_score >> fp_neighbor_score: GNN pode discriminar.
#
#   Fase 3 — GNN rápido (15 epochs):
#     AUROC do score refinado vs. AUROC do SF-GMM bruto
#     Precision a threshold alto (0.8): GNN vs SF-GMM
#     → Valida se o refinamento melhora a precision especificamente.
#
# Uso: python3 src/test_spatial_gnn_hypothesis.py
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import *
from utils.data import load_folder
from utils.evaluation import ScalarFieldGMM
from teacher_student_v3 import _xyz_knn
from spatial_refinement_gnn import (
    SpatialRefinementGNN, train_spatial_gnn, compute_gnn_scores,
    compute_sf_gmm_score, build_node_features, compute_pseudo_labels,
    GNN_INPUT_DIM, K_GNN, PSEUDO_THR_POS, PSEUDO_THR_NEG,
)

OUT_DIR       = f'{BASE_PATH}/visualizations_gnn/hypothesis_test'
MAX_CLOUDS    = 15
EPOCHS_FAST   = 15
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# FASE 1 — QUALIDADE DOS PSEUDO-LABELS
# ============================================================================

def phase1_pseudo_label_quality(labeled: list) -> dict:
    """
    Avalia qualidade dos pseudo-labels SF-GMM usando GT labels como referência.

    Métricas:
      precision_pos: dos pseudo-positivos (prob>THR_POS), % que são GT crack
      recall_pos   : dos GT cracks, % capturados como pseudo-positivos
      precision_neg: dos pseudo-negativos (prob<THR_NEG), % que são GT normal
    """
    print("\n" + "─"*55)
    print("FASE 1 — Qualidade dos pseudo-labels SF-GMM")
    print(f"  THR positivo: prob > {PSEUDO_THR_POS}")
    print(f"  THR negativo: prob < {PSEUDO_THR_NEG}")
    print("─"*55)

    tp_pos = tn_neg = fp_pos = fn_pos = fp_neg = 0
    aurocs = []

    for d in labeled:
        gt      = d['labels']
        sf_prob = compute_sf_gmm_score(d)

        mask_pos = sf_prob > PSEUDO_THR_POS
        mask_neg = sf_prob < PSEUDO_THR_NEG

        # Pseudo-positivos
        if mask_pos.sum() > 0:
            tp_pos += (gt[mask_pos] == 1).sum()
            fp_pos += (gt[mask_pos] == 0).sum()
        # Pseudo-negativos
        if mask_neg.sum() > 0:
            tn_neg += (gt[mask_neg] == 0).sum()
            fp_neg += (gt[mask_neg] == 1).sum()
        # Recall de crack
        fn_pos += (gt[mask_neg] == 1).sum()

        # AUROC do SF-GMM para referência
        if gt.sum() > 5 and (gt == 0).sum() > 5:
            try:
                aurocs.append(roc_auc_score(gt, sf_prob))
            except Exception:
                pass

    prec_pos = tp_pos / (tp_pos + fp_pos + 1e-8)
    prec_neg = tn_neg / (tn_neg + fp_neg + 1e-8)
    recall_c = tp_pos / (tp_pos + fn_pos + 1e-8)
    auroc_sf = float(np.mean(aurocs)) if aurocs else float('nan')

    print(f"\n  Pseudo-positivos (prob > {PSEUDO_THR_POS}):")
    print(f"    Precision : {prec_pos:.3f}  "
          f"({'✓ boa' if prec_pos > 0.7 else '✗ baixa'})")
    print(f"    Recall GT : {recall_c:.3f}  "
          f"({'✓ boa' if recall_c > 0.4 else '~ parcial — aceitável'})")
    print(f"\n  Pseudo-negativos (prob < {PSEUDO_THR_NEG}):")
    print(f"    Precision : {prec_neg:.3f}  "
          f"({'✓ boa' if prec_neg > 0.9 else '✗ baixa'})")
    print(f"\n  AUROC SF-GMM (referência): {auroc_sf:.4f}")

    if prec_pos > 0.6 and prec_neg > 0.85:
        print("\n  ✓ Pseudo-labels têm qualidade suficiente para treinar o GNN.")
    elif prec_pos > 0.4:
        print("\n  ~ Pseudo-labels razoáveis. GNN pode tolerar ruído na loss.")
    else:
        print("\n  ✗ Pseudo-labels com baixa precision. "
              "Considerar aumentar PSEUDO_THR_POS.")

    return {
        'prec_pos': prec_pos, 'recall_crack': recall_c,
        'prec_neg': prec_neg, 'auroc_sf': auroc_sf,
    }


# ============================================================================
# FASE 2 — CONECTIVIDADE DE VIZINHANÇA (hipótese principal)
# ============================================================================

def phase2_neighborhood_connectivity(labeled: list) -> dict:
    """
    Testa se pontos de rachadura têm vizinhos com SF-GMM score alto,
    enquanto FPs têm vizinhos com SF-GMM score baixo.

    Se crack_neighbor_mean >> fp_neighbor_mean → o GNN pode discriminar
    usando o contexto espacial, mesmo que o score individual seja similar.
    """
    print("\n" + "─"*55)
    print("FASE 2 — Conectividade de vizinhança (hipótese principal)")
    print(f"  k={K_GNN} vizinhos em XYZ")
    print("─"*55)

    crack_own   = []   # SF-GMM score dos pontos de rachadura
    crack_neigh = []   # SF-GMM score médio dos vizinhos de rachadura
    fp_own      = []   # SF-GMM score dos falsos positivos (SF alto, GT=0)
    fp_neigh    = []   # SF-GMM score médio dos vizinhos de FP
    normal_neigh= []   # SF-GMM score médio dos vizinhos de pontos normais

    for d in labeled:
        gt      = d['labels']
        sf_prob = compute_sf_gmm_score(d)
        xyz     = d['features'][:, :3]
        knn     = _xyz_knn(xyz, K_GNN)   # (N, k)

        # Score médio dos vizinhos por ponto
        neighbor_mean = sf_prob[knn].mean(axis=1)  # (N,)

        # Pontos de rachadura (GT=1)
        is_crack = gt == 1
        if is_crack.sum() > 0:
            crack_own.extend(sf_prob[is_crack].tolist())
            crack_neigh.extend(neighbor_mean[is_crack].tolist())

        # Falsos positivos do SF-GMM: score alto mas GT=0
        is_fp = (sf_prob > PSEUDO_THR_POS) & (gt == 0)
        if is_fp.sum() > 0:
            fp_own.extend(sf_prob[is_fp].tolist())
            fp_neigh.extend(neighbor_mean[is_fp].tolist())

        # Pontos normais (GT=0) com score baixo
        is_normal = (gt == 0) & (sf_prob < PSEUDO_THR_NEG)
        if is_normal.sum() > 0:
            normal_neigh.extend(neighbor_mean[is_normal].tolist())

    crack_own    = np.array(crack_own)
    crack_neigh  = np.array(crack_neigh)
    fp_own       = np.array(fp_own)
    fp_neigh     = np.array(fp_neigh)
    normal_neigh = np.array(normal_neigh)

    print(f"\n  Pontos de rachadura (GT=1) — n={len(crack_own):,}:")
    print(f"    SF-GMM score próprio    : {crack_own.mean():.4f} ± {crack_own.std():.4f}")
    print(f"    SF-GMM score vizinhos   : {crack_neigh.mean():.4f} ± {crack_neigh.std():.4f}")

    if len(fp_own) > 0:
        print(f"\n  Falsos positivos SF-GMM (score alto, GT=0) — n={len(fp_own):,}:")
        print(f"    SF-GMM score próprio    : {fp_own.mean():.4f} ± {fp_own.std():.4f}")
        print(f"    SF-GMM score vizinhos   : {fp_neigh.mean():.4f} ± {fp_neigh.std():.4f}")
    else:
        print("\n  Nenhum falso positivo SF-GMM encontrado nas nuvens avaliadas.")
        fp_neigh = np.array([0.0])

    print(f"\n  Pontos normais (GT=0, score baixo) — n={len(normal_neigh):,}:")
    print(f"    SF-GMM score vizinhos   : {normal_neigh.mean():.4f} ± {normal_neigh.std():.4f}")

    # Separação chave: vizinhos de crack vs vizinhos de FP
    delta = crack_neigh.mean() - fp_neigh.mean()
    sep   = delta / (fp_neigh.std() + 1e-8)
    print(f"\n  Δ vizinhos (crack - FP)  : {delta:+.4f}  (sep={sep:.2f}σ)")

    if sep > 1.0:
        verdict = "✓ HIPÓTESE VÁLIDA — vizinhança discrimina crack de FP."
    elif sep > 0.3:
        verdict = "~ Separação moderada — GNN pode aprender com epochs suficientes."
    else:
        verdict = "✗ HIPÓTESE FRACA — vizinhança de crack e FP são similares."
    print(f"  {verdict}")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Score próprio vs score vizinhos (rachaduras)
    axes[0].scatter(crack_own[:3000], crack_neigh[:3000],
                    alpha=0.3, s=3, c='tomato', label='Rachadura (GT=1)')
    if len(fp_own) > 0:
        axes[0].scatter(fp_own[:1000], fp_neigh[:1000],
                        alpha=0.3, s=3, c='orange', label=f'FP SF-GMM (GT=0)')
    axes[0].set_xlabel('SF-GMM score (ponto)')
    axes[0].set_ylabel('SF-GMM score (média vizinhos)')
    axes[0].set_title('Score próprio vs vizinhança')
    axes[0].legend(markerscale=5)

    # Distribuição do score de vizinhança
    axes[1].hist(crack_neigh,   bins=50, alpha=0.6, color='tomato',
                 label='Vizinhos de rachadura', density=True)
    if len(fp_neigh) > 1:
        axes[1].hist(fp_neigh, bins=50, alpha=0.6, color='orange',
                     label='Vizinhos de FP', density=True)
    axes[1].hist(normal_neigh,  bins=50, alpha=0.4, color='steelblue',
                 label='Vizinhos de normal', density=True)
    axes[1].set_xlabel('SF-GMM score médio dos vizinhos XYZ')
    axes[1].set_ylabel('Densidade')
    axes[1].set_title(f'Conectividade de vizinhança\nΔ={delta:+.4f}, sep={sep:.2f}σ')
    axes[1].legend()

    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'neighborhood_connectivity.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Plot salvo: {out}")

    return {
        'crack_neigh_mean': float(crack_neigh.mean()),
        'fp_neigh_mean'   : float(fp_neigh.mean()),
        'delta'           : float(delta),
        'separation_sigma': float(sep),
        'verdict'         : verdict,
    }


# ============================================================================
# FASE 3 — GNN RÁPIDO
# ============================================================================

def phase3_quick_gnn(all_data: list, labeled: list,
                     device: torch.device) -> dict:
    """
    Treina GNN por EPOCHS_FAST epochs e mede melhora sobre SF-GMM.

    Métricas comparadas:
      AUROC: SF-GMM vs GNN refinado
      Precision@0.8: dos pontos com score > 0.8, qual % são crack?
    """
    print("\n" + "─"*55)
    print(f"FASE 3 — GNN rápido ({EPOCHS_FAST} epochs)")
    print("─"*55)

    # SF-GMM baseline
    aurocs_sf, prec_sf_08 = [], []
    for d in labeled:
        gt      = d['labels']
        sf_prob = compute_sf_gmm_score(d)
        if gt.sum() < 5:
            continue
        try:
            aurocs_sf.append(roc_auc_score(gt, sf_prob))
        except Exception:
            pass
        high_mask = sf_prob > 0.8
        if high_mask.sum() > 0:
            prec_sf_08.append((gt[high_mask] == 1).mean())

    auroc_sf_mean = float(np.mean(aurocs_sf)) if aurocs_sf else float('nan')
    prec_sf_mean  = float(np.mean(prec_sf_08)) if prec_sf_08 else float('nan')
    print(f"\n  SF-GMM baseline:")
    print(f"    AUROC     = {auroc_sf_mean:.4f}")
    print(f"    Prec@0.8  = {prec_sf_mean:.4f}")

    # Treina GNN
    ckp = os.path.join(OUT_DIR, 'gnn_fast.pth')
    model = SpatialRefinementGNN(input_dim=GNN_INPUT_DIM, k=K_GNN)
    model = train_spatial_gnn(model, all_data, device,
                               num_epochs=EPOCHS_FAST, lr=3e-4,
                               save_path=ckp)

    # Avalia GNN
    aurocs_gnn, prec_gnn_08 = [], []
    model.eval()
    for d in labeled:
        gt       = d['labels']
        sf_prob  = compute_sf_gmm_score(d)
        node_f   = build_node_features(d, sf_prob)
        knn      = _xyz_knn(d['features'][:, :3], K_GNN)
        x        = torch.tensor(node_f, dtype=torch.float32).to(device)
        with torch.no_grad():
            score = model(x, knn).cpu().numpy().astype(np.float32)
        del x
        if gt.sum() < 5:
            continue
        try:
            aurocs_gnn.append(roc_auc_score(gt, score))
        except Exception:
            pass
        high_mask = score > 0.8
        if high_mask.sum() > 0:
            prec_gnn_08.append((gt[high_mask] == 1).mean())

    auroc_gnn_mean = float(np.mean(aurocs_gnn)) if aurocs_gnn else float('nan')
    prec_gnn_mean  = float(np.mean(prec_gnn_08)) if prec_gnn_08 else float('nan')
    print(f"\n  GNN ({EPOCHS_FAST} epochs):")
    print(f"    AUROC     = {auroc_gnn_mean:.4f}  "
          f"(Δ = {auroc_gnn_mean - auroc_sf_mean:+.4f})")
    print(f"    Prec@0.8  = {prec_gnn_mean:.4f}  "
          f"(Δ = {prec_gnn_mean - prec_sf_mean:+.4f})")

    delta_auroc = auroc_gnn_mean - auroc_sf_mean
    delta_prec  = prec_gnn_mean  - prec_sf_mean

    if delta_prec > 0.05:
        print(f"\n  ✓ GNN melhora precision em +{delta_prec:.3f}. "
              "Prosseguir com treino completo.")
    elif delta_auroc > 0.02:
        print(f"\n  ~ GNN melhora AUROC em +{delta_auroc:.3f}. "
              "Treino completo pode solidificar a melhora.")
    else:
        print("\n  ~ GNN não melhora significativamente em 15 epochs. "
              "Verificar qualidade dos pseudo-labels antes de treino completo.")

    # Plot comparativo
    fig, ax = plt.subplots(figsize=(6, 4))
    cats   = ['SF-GMM\n(baseline)', f'GNN\n({EPOCHS_FAST}ep)']
    aurocs = [auroc_sf_mean, auroc_gnn_mean]
    precs  = [prec_sf_mean,  prec_gnn_mean]
    x_pos  = np.arange(2)

    ax.bar(x_pos - 0.2, aurocs, 0.35, label='AUROC', color='steelblue', alpha=0.8)
    ax.bar(x_pos + 0.2, precs,  0.35, label='Prec@0.8', color='tomato', alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Score')
    ax.set_title('SF-GMM vs GNN Refinado')
    ax.legend()
    ax.axhline(0.887, color='green', linestyle=':', linewidth=1, label='SF teto')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'gnn_vs_sfgmm.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Plot salvo: {out}")

    return {
        'auroc_sf'  : auroc_sf_mean,
        'auroc_gnn' : auroc_gnn_mean,
        'prec_sf_08': prec_sf_mean,
        'prec_gnn_08': prec_gnn_mean,
        'delta_auroc': delta_auroc,
        'delta_prec' : delta_prec,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*65)
    print("  DIAGNÓSTICO — Hipótese Spatial Refinement GNN")
    print("="*65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    print("\nCarregando dados...")
    all_train = load_folder(DATA_TRAIN)
    all_test  = load_folder(DATA_TEST)
    all_data  = all_train + all_test

    labeled = [d for d in all_train
               if d.get('has_crack') and d['labels'] is not None
               ][:MAX_CLOUDS]
    print(f"Nuvens avaria (avaliação): {len(labeled)}")

    if not labeled:
        print("ERRO: nenhuma nuvem com labels.")
        return

    # ── Fase 1 ────────────────────────────────────────────────────────────────
    r1 = phase1_pseudo_label_quality(labeled)

    # ── Fase 2 ────────────────────────────────────────────────────────────────
    r2 = phase2_neighborhood_connectivity(labeled)

    # ── Fase 3 ────────────────────────────────────────────────────────────────
    r3 = phase3_quick_gnn(all_data, labeled, device)

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RESUMO FINAL")
    print("="*65)
    print(f"  Pseudo-label precision pos : {r1['prec_pos']:.3f}")
    print(f"  Pseudo-label precision neg : {r1['prec_neg']:.3f}")
    print(f"  Separação vizinhança       : {r2['separation_sigma']:.2f}σ")
    print(f"  AUROC SF-GMM               : {r3['auroc_sf']:.4f}")
    print(f"  AUROC GNN ({EPOCHS_FAST}ep)        : {r3['auroc_gnn']:.4f}  "
          f"(Δ={r3['delta_auroc']:+.4f})")
    print(f"  Prec@0.8 SF-GMM            : {r3['prec_sf_08']:.4f}")
    print(f"  Prec@0.8 GNN ({EPOCHS_FAST}ep)     : {r3['prec_gnn_08']:.4f}  "
          f"(Δ={r3['delta_prec']:+.4f})")
    print()

    # Veredicto consolidado
    votes_ok = sum([
        r1['prec_pos'] > 0.6,
        r2['separation_sigma'] > 0.3,
        r3['delta_prec'] > 0.0 or r3['delta_auroc'] > 0.0,
    ])

    if votes_ok == 3:
        print("  ✓ HIPÓTESE VÁLIDA — prosseguir com treino completo do GNN.")
    elif votes_ok == 2:
        print("  ~ HIPÓTESE PARCIALMENTE VÁLIDA — prosseguir com cautela.")
        print("    Treino completo (80 epochs) pode consolidar a melhora.")
    else:
        print("  ✗ HIPÓTESE FRACA — revisar arquitetura ou pseudo-labels.")
        print("    Considerar fine-tuning supervisionado como alternativa.")

    print(f"\n  Resultados em: {OUT_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()
