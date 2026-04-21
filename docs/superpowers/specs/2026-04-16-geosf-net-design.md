# Design: GeoSF-Net — Detecção de Rachaduras por Fusão Scalar Field + Geometria Local

**Data:** 2026-04-16
**Status:** Aprovado pelo usuário
**Contexto:** TCC — detecção de rachaduras em nuvens de pontos 3D

---

## Sumário

Arquitetura híbrida (clássico + neural) que combina descritores de vizinhança local (scalar field,
normais, curvatura) com uma GNN de dois branches e cross-attention bidirecional, treinada com
supervisão nos avaria_*.ply. Pós-processamento via DBSCAN elimina pontos espalhados (falha
central do scalar_field_unsup).

Documento técnico completo: `text_for_ia/GEOSF_NET.md`

---

## Motivação

`scalar_field_unsup.py` produziu ~70% da nuvem como crack (imagem avaria_32). Causa raiz:
DGCNN operava em features 512D do Teacher sem restrição espacial, e a contrastive loss com
pseudo-labels ruidosos corrompeu a representação.

---

## Arquitetura (4 estágios)

```
(N, 16) → Neighborhood Descriptor Builder → (N, 26)
        → GeoSF-GNN [Branch SF + Branch Geo + Cross-Attention] → (N, 128)
        → Score Head + Uncertainty Head → score ∈ [0,1], conf ∈ [0,1]
        → DBSCAN Espacial → pred_labels (N,) binário
```

---

## Robustez

- **Ruído:** attention weights + uncertainty head + DBSCAN min_samples
- **Desbalanceamento (6.62%):** Focal Loss (γ=2, α=0.85) + class-balanced sampling + warm-up curriculum
- **Generalização:** features relativas (`sf_rel = sf[i] - sf_mean_k`), per-cloud normalization, GMM prior por nuvem, augmentação Z+jitter

---

## Baseline

F1=90.3%, IoU=82.4%, AUROC=95.2% (teacher_student_v1, run 13/04/2026)

**Alvo:** F1 >92%, IoU >85%, AUROC >96%, pontos vermelhos localizados na rachadura.

---

## Arquivos planejados

```
src/geosf_net.py
src/utils/geo_descriptors.py
src/utils/geosf_gnn.py
```
