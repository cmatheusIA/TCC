# ============================================================================
# DIAGNÓSTICO — Hipótese do Domain Adapter
# ============================================================================
# Valida se a consistência de vizinhança no espaço de features do Teacher
# é discriminativa para rachaduras ANTES e DEPOIS de treinar o adapter.
#
# Hipótese a testar:
#   score(i) = 1 - cosine_similarity(h_i, mean(h_vizinhos_XYZ))
#   deve ser maior em pontos de rachadura (descontinuidade de material)
#   do que em pontos de superfície normal.
#
# Resultado esperado se a hipótese for válida:
#   AUROC(score_raw, labels) > 0.5   [Teacher puro — baseline]
#   AUROC(score_adapted, labels) > AUROC(score_raw)  [adapter melhora]
#
# Resultado que invalida a abordagem:
#   AUROC(score_raw, labels) ≈ 0.5   [Teacher features são iid em XYZ]
#   → adapter não tem sinal para aprender; trocar por outra abordagem.
#
# Uso: python3 src/test_adapter_hypothesis.py
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.config import *
from utils.data import load_folder, split_dataset
from teacher_student_v2 import build_teacher, TEACHER_CHUNK_SIZE
from teacher_student_v3 import (
    DomainAdapter, pretrain_adapter, _xyz_knn,
    ADAPTER_DIM, ADAPTER_K, EPOCHS_ADAPTER, LR_ADAPTER,
)

OUT_DIR = f'{BASE_PATH}/visualizations_ts_v3/adapter_diagnostic'
os.makedirs(OUT_DIR, exist_ok=True)

K         = ADAPTER_K   # vizinhos XYZ
MAX_CLOUDS = 10          # nuvens avaria com labels para avaliar
ADAPTER_EPOCHS_FAST = 20 # epochs rápidos só para ver a tendência


# ============================================================================
# FUNÇÃO CORE — consistency score em numpy (sem GPU)
# ============================================================================

def consistency_score_np(h: np.ndarray, xyz: np.ndarray, k: int = K) -> np.ndarray:
    """
    score(i) = 1 - cosine_similarity(h_i, mean(h_j))
    onde j são os k vizinhos mais próximos de i em XYZ.

    h   : (N, D) float32
    xyz : (N, 3) float32
    Returns: (N,) float32 ∈ [0, 2] — 0=idêntico à vizinhança, 2=oposto
    """
    knn    = _xyz_knn(xyz, k)          # (N, k) int64
    h_norm = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-8)

    neighbors = h_norm[knn]            # (N, k, D)
    h_local   = neighbors.mean(axis=1) # (N, D)
    h_local   = h_local / (np.linalg.norm(h_local, axis=1, keepdims=True) + 1e-8)

    cos = (h_norm * h_local).sum(axis=1)  # (N,)
    return (1.0 - cos).astype(np.float32)


# ============================================================================
# EXTRAÇÃO DE BOTTLENECKS DO TEACHER
# ============================================================================

@torch.no_grad()
def extract_bottlenecks(teacher: torch.nn.Module,
                        data_list: list,
                        device: torch.device) -> list:
    """
    Extrai Teacher bottleneck (N, 512) para cada nuvem.
    Retorna lista de dicts com 'h', 'xyz', 'labels', 'filename'.
    """
    teacher.eval()
    results = []
    for d in data_list:
        if d['labels'] is None:
            continue
        x   = torch.tensor(d['features'], dtype=torch.float32).to(device)
        xyz = d['features'][:, :3]
        lbl = d['labels']

        # Forward em chunks (evita OOM)
        parts = []
        for s in range(0, x.size(0), TEACHER_CHUNK_SIZE):
            parts.append(teacher(x[s:min(s + TEACHER_CHUNK_SIZE, x.size(0))]))
        h = torch.cat(parts, dim=0).cpu().numpy().astype(np.float32)

        results.append({
            'filename': d['filename'],
            'h'       : h,
            'xyz'     : xyz,
            'labels'  : lbl,
        })
        del x
        torch.cuda.empty_cache()
        print(f"  Extraído: {d['filename']} ({len(lbl):,} pts, "
              f"{lbl.sum():,} crack)")

    return results


# ============================================================================
# AVALIAÇÃO: AUROC + distribuição por classe
# ============================================================================

def evaluate_scores(cache: list, score_key: str, tag: str):
    """
    Calcula AUROC médio e plota distribuição crack vs. normal
    para o score em score_key.
    """
    aurocs = []
    all_scores_crack  = []
    all_scores_normal = []

    for d in cache:
        score  = d[score_key]
        labels = d['labels']

        if labels.sum() < 5 or (labels == 0).sum() < 5:
            continue

        try:
            auroc = roc_auc_score(labels, score)
            aurocs.append(auroc)
        except Exception:
            pass

        all_scores_crack.extend(score[labels == 1].tolist())
        all_scores_normal.extend(score[labels == 0].tolist())

    mean_auroc = float(np.mean(aurocs)) if aurocs else float('nan')
    print(f"\n[{tag}]")
    print(f"  AUROC médio : {mean_auroc:.4f}  (n={len(aurocs)} nuvens)")
    print(f"  Score crack  — média: {np.mean(all_scores_crack):.4f}  "
          f"std: {np.std(all_scores_crack):.4f}")
    print(f"  Score normal — média: {np.mean(all_scores_normal):.4f}  "
          f"std: {np.std(all_scores_normal):.4f}")

    # Interpretação
    if mean_auroc > 0.70:
        print(f"  ✓ AUROC > 0.70 → hipótese VÁLIDA: consistency discrimina rachaduras")
    elif mean_auroc > 0.55:
        print(f"  ~ AUROC 0.55–0.70 → sinal fraco, adapter pode amplificar")
    else:
        print(f"  ✗ AUROC ≈ 0.5 → hipótese INVÁLIDA: trocar abordagem")

    # Plot distribuição
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(all_scores_normal, bins=60, alpha=0.6, color='steelblue',
            label='Normal', density=True)
    ax.hist(all_scores_crack,  bins=60, alpha=0.6, color='tomato',
            label='Rachadura', density=True)
    ax.set_xlabel('Consistency score')
    ax.set_ylabel('Densidade')
    ax.set_title(f'{tag} | AUROC={mean_auroc:.4f}')
    ax.legend()
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f'dist_{tag.replace(" ", "_")}.png')
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Distribuição salva: {out}")

    return mean_auroc, all_scores_crack, all_scores_normal


# ============================================================================
# VALIDAÇÃO EXTRA: scalar_field como baseline de comparação
# ============================================================================

def evaluate_sf_baseline(cache: list):
    """
    AUROC do scalar_field bruto (col 9, invertido: SF baixo = crack).
    Serve como teto de referência — o adapter_score não deve ser muito
    inferior ao SF bruto para ser útil.
    """
    aurocs = []
    for d in cache:
        sf_inv = -d['sf']   # SF baixo = rachadura → invertemos para AUROC
        labels = d['labels']
        if labels.sum() < 5:
            continue
        try:
            aurocs.append(roc_auc_score(labels, sf_inv))
        except Exception:
            pass
    mean_auroc = float(np.mean(aurocs)) if aurocs else float('nan')
    print(f"\n[Baseline — scalar_field bruto (invertido)]")
    print(f"  AUROC médio: {mean_auroc:.4f}  (referência máxima esperada)")
    return mean_auroc


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*65)
    print("  DIAGNÓSTICO — Hipótese do Domain Adapter")
    print("="*65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Dispositivo: {device}")

    # ── 1. Carrega dados com labels (avaria_*) ────────────────────────────────
    print("\nCarregando dados...")
    train_all = load_folder(DATA_TRAIN)
    # Apenas nuvens com labels (avaria_*) e com crack
    labeled   = [d for d in train_all
                 if d.get('has_crack') and d['labels'] is not None][:MAX_CLOUDS]

    if not labeled:
        print("ERRO: nenhuma nuvem com labels encontrada.")
        return

    print(f"Nuvens avaliadas: {len(labeled)}")

    # ── 2. Carrega Teacher ────────────────────────────────────────────────────
    print("\nCarregando Teacher...")
    teacher = build_teacher(
        input_dim=INPUT_DIM,
        ptv3_ckpt=PTRANSF_WEIGHTS,
        s3dis_ckpt=PTRANSF_WEIGHTS_S3DIS,
    ).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    # ── 3. Extrai bottlenecks do Teacher ─────────────────────────────────────
    print("\nExtraindo bottlenecks do Teacher...")
    cache = extract_bottlenecks(teacher, labeled, device)

    # Adiciona SF bruto ao cache para baseline
    for d_cache, d_raw in zip(cache, labeled):
        d_cache['sf'] = d_raw['features'][:, 9]

    # ── 4. Computa consistency score — Teacher RAW (sem adapter) ──────────────
    print("\nComputando consistency scores (Teacher raw, sem adapter)...")
    for d in cache:
        d['score_raw'] = consistency_score_np(d['h'], d['xyz'], k=K)

    # ── 5. AUROC baseline: scalar_field ──────────────────────────────────────
    auroc_sf = evaluate_sf_baseline(cache)

    # ── 6. AUROC: Teacher raw ─────────────────────────────────────────────────
    auroc_raw, _, _ = evaluate_scores(cache, 'score_raw', 'Teacher raw (sem adapter)')

    # ── 7. Treina adapter (versão rápida, apenas para diagnóstico) ────────────
    print(f"\nTreinando adapter por {ADAPTER_EPOCHS_FAST} epochs (diagnóstico rápido)...")
    all_data = load_folder(DATA_TRAIN) + load_folder(DATA_TEST)
    adapter  = DomainAdapter(d=512, d_bot=ADAPTER_DIM).to(device)

    adapter = pretrain_adapter(
        adapter, teacher, all_data, device,
        num_epochs=ADAPTER_EPOCHS_FAST,
        lr=LR_ADAPTER,
        k=K,
        save_path=os.path.join(OUT_DIR, 'adapter_diagnostic.pth'),
        normal_only=True,
    )
    adapter.eval()

    # ── 8. Computa consistency score — com adapter treinado ───────────────────
    print("\nComputando consistency scores (com adapter)...")
    for d in cache:
        h_t = torch.tensor(d['h'], dtype=torch.float32).to(device)
        with torch.no_grad():
            h_adapted = adapter(h_t).cpu().numpy().astype(np.float32)
        d['score_adapted'] = consistency_score_np(h_adapted, d['xyz'], k=K)

    # ── 9. AUROC: adapter ─────────────────────────────────────────────────────
    auroc_adapted, _, _ = evaluate_scores(
        cache, 'score_adapted', f'Adapter ({ADAPTER_EPOCHS_FAST} epochs)')

    # ── 10. Resumo e interpretação ────────────────────────────────────────────
    print("\n" + "="*65)
    print("  RESUMO")
    print("="*65)
    print(f"  Baseline scalar_field : AUROC = {auroc_sf:.4f}  (teto de referência)")
    print(f"  Teacher raw           : AUROC = {auroc_raw:.4f}  (antes do adapter)")
    print(f"  Adapter {ADAPTER_EPOCHS_FAST} epochs     : AUROC = {auroc_adapted:.4f}  (após adapter)")
    print()

    if auroc_raw > 0.55:
        print("  → Teacher já mostra sinal de inconsistência nas rachaduras.")
        print("    Adapter deve amplificar esse sinal. Prosseguir com Caminho A.")
    else:
        print("  → Teacher raw não discrimina rachaduras por inconsistência local.")
        if auroc_adapted > auroc_raw + 0.05:
            print("    Adapter consegue aprender o sinal do zero. Prosseguir com cautela.")
        else:
            print("    Adapter também não discrimina. Hipótese INVÁLIDA.")
            print("    Recomendação: partir para Caminho B (Scalar Field MAE).")

    delta = auroc_adapted - auroc_raw
    print(f"\n  Delta AUROC (adapter - raw): {delta:+.4f}")
    if delta > 0.05:
        print("  → Adapter melhora significativamente o sinal. ✓")
    elif delta > 0:
        print("  → Adapter melhora marginalmente. Treino completo pode ajudar.")
    else:
        print("  → Adapter não melhora (ou piora). Revisar loss de alinhamento.")

    print(f"\n  Distribuições salvas em: {OUT_DIR}")
    print("="*65)


if __name__ == '__main__':
    main()
