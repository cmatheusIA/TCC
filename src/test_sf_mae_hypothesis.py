# ============================================================================
# DIAGNÓSTICO — Hipótese do Scalar Field MAE (Caminho B)
# ============================================================================
#
# Hipótese a testar:
#   Um modelo treinado em superfícies normais comete erro maior ao prever
#   o scalar_field de pontos de rachadura do que de pontos normais.
#
# Estrutura do diagnóstico (3 fases):
#
#   Fase 1 — Modelo NÃO treinado (aleatório):
#     AUROC(|SF_pred_random - SF_real|, labels) ≈ 0.5
#     Confirma que o sinal NÃO vem da arquitetura, e sim do treino.
#
#   Fase 2 — Modelo treinado em N epochs rápidos:
#     AUROC deve ser > 0.5 e próximo do baseline SF.
#     Se AUROC ≈ 0.5 após treino → hipótese INVÁLIDA (modelo não aprende
#     a distinguir crack de normal pelo erro de reconstrução).
#
#   Fase 3 — Comparação com baselines:
#     • Scalar field bruto invertido  (AUROC ≈ 0.887 — teto de referência)
#     • Apenas RGB → SF (modelo sem XYZ/normais)  — quanto XYZ/normais ajudam?
#     • Modelo completo (XYZ + RGB + normais)
#
# Interpretação esperada se a hipótese for válida:
#   AUROC_random ≈ 0.50
#   AUROC_trained > 0.70
#   AUROC_trained ≈ AUROC_sf_baseline → modelo redescobre o sinal do SF
#                                        a partir de geometria + cor
#
# Uso: python3 src/test_sf_mae_hypothesis.py
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import *
from utils.data import load_folder
from scalar_field_mae import (
    LocalSFPredictor, train_sf_predictor, _edge_conv_chunked,
    K_EDGE, INPUT_COLS, EPOCHS_SF,
)
from teacher_student_v3 import _xyz_knn

OUT_DIR        = f'{BASE_PATH}/visualizations_sf_mae/hypothesis_test'
MAX_CLOUDS     = 12    # nuvens avaria com labels para avaliar
EPOCHS_FAST    = 15    # epochs rápidos para diagnóstico (vs. 100 completo)
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================================
# EXTRAÇÃO DE ERROS DE RECONSTRUÇÃO
# ============================================================================

@torch.no_grad()
def compute_recon_errors(
    model: LocalSFPredictor,
    labeled_data: list,
    device: torch.device,
) -> list:
    """
    Computa |SF_pred - SF_real/255| para cada ponto de cada nuvem.
    Retorna lista de dicts com 'error', 'labels', 'sf_real', 'filename'.
    """
    model.eval()
    results = []

    for d in labeled_data:
        x      = torch.tensor(d['features'][:, INPUT_COLS],
                               dtype=torch.float32).to(device)
        sf_raw = d['features'][:, 9]
        knn    = _xyz_knn(d['features'][:, :3], K_EDGE)

        sf_pred = model(x, knn).cpu().numpy().astype(np.float32)
        error   = np.abs(sf_pred - sf_raw / 255.0).astype(np.float32)

        results.append({
            'filename': d['filename'],
            'error'   : error,
            'labels'  : d['labels'],
            'sf_real' : sf_raw,
        })
        del x
        torch.cuda.empty_cache()

    return results


# ============================================================================
# AVALIAÇÃO: AUROC + DISTRIBUIÇÃO
# ============================================================================

def evaluate_and_plot(results: list, tag: str, color: str = 'steelblue'):
    """
    Calcula AUROC médio e plota histograma de erros por classe.
    Retorna AUROC médio.
    """
    aurocs       = []
    err_crack    = []
    err_normal   = []

    for d in results:
        err    = d['error']
        labels = d['labels']
        if labels is None or labels.sum() < 5 or (labels == 0).sum() < 5:
            continue
        try:
            aurocs.append(roc_auc_score(labels, err))
        except Exception:
            pass
        err_crack.extend(err[labels == 1].tolist())
        err_normal.extend(err[labels == 0].tolist())

    mean_auroc = float(np.mean(aurocs)) if aurocs else float('nan')

    print(f"\n[{tag}]")
    print(f"  AUROC médio   : {mean_auroc:.4f}  (n={len(aurocs)} nuvens)")
    print(f"  Erro crack    : média={np.mean(err_crack):.4f}  "
          f"std={np.std(err_crack):.4f}")
    print(f"  Erro normal   : média={np.mean(err_normal):.4f}  "
          f"std={np.std(err_normal):.4f}")

    sep = (np.mean(err_crack) - np.mean(err_normal)) / (
        np.std(err_normal) + 1e-8)
    print(f"  Separação     : {sep:.3f} σ  "
          f"({'boa' if sep > 1 else 'fraca' if sep > 0.3 else 'nula'})")

    # Histograma
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(err_normal, bins=60, alpha=0.6, color='steelblue',
            label='Normal', density=True)
    ax.hist(err_crack,  bins=60, alpha=0.6, color='tomato',
            label='Rachadura', density=True)
    ax.set_xlabel('|SF_pred − SF_real/255|')
    ax.set_ylabel('Densidade')
    ax.set_title(f'{tag}\nAUROC={mean_auroc:.4f}  |  sep={sep:.2f}σ')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'dist_{tag.replace(" ", "_")}.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Histograma    : {out}")

    return mean_auroc, err_crack, err_normal


# ============================================================================
# BASELINE RGB-ONLY (sem XYZ/normais)
# ============================================================================

class RGBOnlySFPredictor(nn.Module):
    """
    Versão ablação: prediz SF apenas a partir de RGB (3D).
    Sem contexto local (sem EdgeConv), sem XYZ, sem normais.
    Objetivo: quantificar quanto XYZ + normais + contexto local adicionam.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        return self.net(rgb).squeeze(-1)


@torch.no_grad()
def compute_rgb_errors(
    rgb_model: RGBOnlySFPredictor,
    labeled_data: list,
    device: torch.device,
) -> list:
    rgb_model.eval()
    results = []
    for d in labeled_data:
        rgb    = torch.tensor(d['features'][:, 3:6],
                               dtype=torch.float32).to(device)
        sf_raw = d['features'][:, 9]
        pred   = rgb_model(rgb).cpu().numpy().astype(np.float32)
        error  = np.abs(pred - sf_raw / 255.0).astype(np.float32)
        results.append({'filename': d['filename'], 'error': error,
                        'labels': d['labels'], 'sf_real': sf_raw})
        del rgb
    return results


def train_rgb_model(
    rgb_model: RGBOnlySFPredictor,
    normal_data: list,
    device: torch.device,
    epochs: int = EPOCHS_FAST,
) -> RGBOnlySFPredictor:
    rgb_model = rgb_model.to(device)
    opt = torch.optim.AdamW(rgb_model.parameters(), lr=1e-3, weight_decay=1e-4)
    rgb_model.train()
    for epoch in range(epochs):
        for d in normal_data:
            rgb    = torch.tensor(d['features'][:, 3:6],
                                   dtype=torch.float32).to(device)
            sf_tgt = torch.tensor(d['features'][:, 9] / 255.0,
                                   dtype=torch.float32).to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.smooth_l1_loss(rgb_model(rgb), sf_tgt, beta=0.05)
            loss.backward()
            opt.step()
            del rgb, sf_tgt
    return rgb_model


# ============================================================================
# COMPARAÇÃO RESUMO — GRÁFICO DE AUROC
# ============================================================================

def plot_auroc_comparison(auroc_dict: dict):
    names  = list(auroc_dict.keys())
    values = [auroc_dict[n] for n in names]
    colors = ['#aaaaaa', '#f4a261', '#2a9d8f', '#e76f51', '#264653']

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, values, color=colors[:len(names)])
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Aleatório')
    ax.axvline(0.887, color='green', linestyle=':', linewidth=1,
               label='SF bruto (teto)')
    for bar, val in zip(bars, values):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('AUROC')
    ax.set_title('Hipótese SF MAE — Comparação de AUROC')
    ax.legend(loc='lower right')
    fig.tight_layout()
    out = os.path.join(OUT_DIR, 'auroc_comparison.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Comparação salva: {out}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*65)
    print("  DIAGNÓSTICO — Hipótese Scalar Field MAE (Caminho B)")
    print("="*65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    print("\nCarregando dados...")
    all_train   = load_folder(DATA_TRAIN)
    labeled     = [d for d in all_train
                   if d.get('has_crack') and d['labels'] is not None
                   ][:MAX_CLOUDS]
    normal_data = [d for d in all_train if not d.get('has_crack', True)]

    print(f"Nuvens avaria (avaliação): {len(labeled)}")
    print(f"Nuvens normais (treino)  : {len(normal_data)}")

    if not labeled:
        print("ERRO: nenhuma nuvem avaria com labels encontrada.")
        return

    # ── 2. FASE 1 — Modelo aleatório (controle) ───────────────────────────────
    print("\n" + "─"*50)
    print("FASE 1 — Modelo NÃO treinado (aleatório)")
    print("─"*50)

    model_random = LocalSFPredictor(k=K_EDGE).to(device)
    res_random   = compute_recon_errors(model_random, labeled, device)
    auroc_random, _, _ = evaluate_and_plot(res_random, 'Modelo aleatório')

    # ── 3. FASE 2 — Baseline: RGB only ────────────────────────────────────────
    print("\n" + "─"*50)
    print("FASE 2 — Ablação: apenas RGB (sem XYZ/normais/contexto)")
    print("─"*50)

    rgb_model = RGBOnlySFPredictor()
    print(f"Treinando RGB-only por {EPOCHS_FAST} epochs...")
    rgb_model = train_rgb_model(rgb_model, normal_data, device, EPOCHS_FAST)
    res_rgb   = compute_rgb_errors(rgb_model, labeled, device)
    auroc_rgb, _, _ = evaluate_and_plot(res_rgb, f'RGB-only ({EPOCHS_FAST} epochs)')

    # ── 4. FASE 3 — Modelo completo (XYZ + RGB + normais + EdgeConv) ──────────
    print("\n" + "─"*50)
    print(f"FASE 3 — Modelo completo ({EPOCHS_FAST} epochs rápidos)")
    print("─"*50)

    ckp_fast = os.path.join(OUT_DIR, 'sf_predictor_fast.pth')
    model_full = LocalSFPredictor(k=K_EDGE)
    n_params   = sum(p.numel() for p in model_full.parameters())
    print(f"LocalSFPredictor: {n_params:,} params | k={K_EDGE}")
    print(f"Treinando por {EPOCHS_FAST} epochs...")

    model_full = train_sf_predictor(
        model_full, all_train, device,
        num_epochs=EPOCHS_FAST, lr=3e-4,
        save_path=ckp_fast, normal_only=True,
    )
    res_full   = compute_recon_errors(model_full, labeled, device)
    auroc_full, _, _ = evaluate_and_plot(
        res_full, f'SF MAE completo ({EPOCHS_FAST} epochs)')

    # ── 5. Baseline: scalar_field bruto ───────────────────────────────────────
    print("\n" + "─"*50)
    print("REFERÊNCIA — Scalar field bruto (teto esperado)")
    print("─"*50)

    aurocs_sf = []
    for d in labeled:
        sf_inv = -d['features'][:, 9]
        lbl    = d['labels']
        if lbl is None or lbl.sum() < 5:
            continue
        try:
            aurocs_sf.append(roc_auc_score(lbl, sf_inv))
        except Exception:
            pass
    auroc_sf = float(np.mean(aurocs_sf)) if aurocs_sf else float('nan')
    print(f"  AUROC SF bruto: {auroc_sf:.4f}")

    # ── 6. Gráfico de comparação ──────────────────────────────────────────────
    auroc_dict = {
        'Aleatório (controle)' : auroc_random,
        f'RGB-only ({EPOCHS_FAST}ep)'   : auroc_rgb,
        f'SF MAE ({EPOCHS_FAST}ep)'     : auroc_full,
        'SF bruto (teto)'      : auroc_sf,
    }
    plot_auroc_comparison(auroc_dict)

    # ── 7. Resumo e interpretação ─────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RESUMO")
    print("="*65)
    print(f"  Aleatório          : AUROC = {auroc_random:.4f}  (controle — deve ser ≈0.5)")
    print(f"  RGB-only           : AUROC = {auroc_rgb:.4f}  (contribuição do RGB)")
    print(f"  SF MAE completo    : AUROC = {auroc_full:.4f}  (XYZ+RGB+normais+EdgeConv)")
    print(f"  SF bruto (teto)    : AUROC = {auroc_sf:.4f}")
    print()

    # Decisão
    delta_rgb  = auroc_rgb  - auroc_random
    delta_full = auroc_full - auroc_rgb
    delta_geo  = auroc_full - auroc_random

    print(f"  Ganho do treino (vs aleatório) : +{delta_geo:.4f}")
    print(f"  Ganho do contexto geométrico   : +{delta_full:.4f}  "
          f"(SF MAE vs RGB-only)")

    print()
    if auroc_full > 0.75:
        print("  ✓ Hipótese VÁLIDA — SF MAE é discriminativo. Prosseguir com Caminho B.")
    elif auroc_full > 0.60:
        print("  ~ Hipótese PARCIALMENTE válida. Treino completo (100 epochs) pode ajudar.")
        print("    Comparar com teacher_student_v1 antes de decidir.")
    else:
        print("  ✗ Hipótese INVÁLIDA — erro de reconstrução não discrimina rachaduras.")
        print("    Recomendação: partir para fine-tuning supervisionado.")

    if delta_full > 0.05:
        print(f"\n  ✓ Contexto geométrico (EdgeConv) agrega +{delta_full:.4f} AUROC vs RGB-only.")
        print("    A arquitetura com local context é justificada.")
    else:
        print(f"\n  ~ EdgeConv agrega apenas +{delta_full:.4f} vs RGB-only.")
        print("    Talvez só o RGB já seja suficiente para prever SF.")

    print(f"\n  Resultados em: {OUT_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()
