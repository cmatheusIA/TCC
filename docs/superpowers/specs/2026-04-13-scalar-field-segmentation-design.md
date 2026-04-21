# Design: Segmentação de Rachaduras via Scalar Field

**Data:** 2026-04-13  
**Status:** Aprovado  
**Contexto:** TCC — detecção de rachaduras em nuvens de pontos 3D

---

## Motivação

Análise exploratória revelou que 15/34 arquivos de avaria possuem `scalar_Scalar_field` com
distribuição bimodal perfeita (gap=1 entre cluster de rachadura e superfície normal, F1=1.0
com threshold simples). Os outros 19 usam `scalar_R` (canal RGB vermelho) — sem poder
discriminativo. O v1 usa um gate hard global `[sf_min, sf_max]` aprendido dos labels — frágil
porque o intervalo varia de `[0,29]` a `[42,83]` entre nuvens.

Objetivo: explorar o scalar_field como sinal de segmentação em dois novos sistemas
comparáveis ao v1 por métricas idênticas.

---

## Arquivos produzidos

```
src/
  scalar_field_unsup.py        ← novo: Teacher frozen + DGCNN self-supervised
  teacher_student_v2.py        ← novo: fork do v1 com 3 upgrades
  utils/
    evaluation.py              ← adições: ScalarFieldGMM, save_colored_ply,
                                  evaluate_ablation, compare_models
```

Nenhum arquivo existente é alterado. v1 continua funcionando sem modificação.

---

## Seção 1 — `src/scalar_field_unsup.py`

### Componente estatístico: `ScalarFieldPrior`

Executado por nuvem, sem rede:

```
scalar_field (N,) → valley_detection → threshold
                 ↘ GMM 1D (fallback) ↗
→ pseudo_labels (N,)  [0/1 ou None se unimodal]
→ confidence float    [0.0–1.0]  = gap / (std_total + ε)
→ modality: 'bimodal' | 'unimodal'
```

- **Bimodal** (`scalar_Scalar_field`, gap ≥ 0): pseudo-labels confiáveis para contrastive loss
- **Unimodal** (`scalar_R`): `pseudo_labels = None`, confidence = 0.0 — usadas só como regularização

### Arquitetura: `ScalarFieldDGCNN`

```
features (N, 15)
  ↓
Teacher (frozen) → bottleneck (N, 512)   [mesmo build_teacher() do v1]
  ↓
EdgeConvBlock(512→256, k=20)   [k-NN no espaço XYZ, edge features: concat(h_i, h_j - h_i)]
  ↓
EdgeConvBlock(256→128, k=20)   [grafo reconstruído por bloco com features atualizadas]
  ↓
EdgeConvBlock(128→64,  k=20)
  ↓
Linear(64→1) + Sigmoid → anomaly_score (N,) ∈ [0,1]
```

Referência: Wang et al. (2019) — Dynamic Graph CNN for Learning on Point Clouds (DGCNN).

### Loss de treino

```python
# Bimodal clouds — contrastive loss ponderada por confidence:
L_contrast = mean( max(0, margin - score[crack] + score[normal]) ) * confidence

# Unimodal clouds — reconstrução como regularização (impede colapso):
L_recon = MSE(features_reconstructed, features_original)

L_total = 0.7 * L_contrast + 0.3 * L_recon
```

### Protocolo de treino — 2 fases

| Fase | Epochs | Dados | Loss ativa |
|---|---|---|---|
| Warmup | 0–30 | Todas as nuvens | `L_recon` apenas |
| Self-supervised | 31–150 | Bimodal + unimodal | `L_total` completo |

Early stopping: patience=20 monitorando `L_total`.

### Inferência

```
x (N,15) → ScalarFieldDGCNN → raw_score (N,)
         → min-max norm por nuvem
         → GMM threshold (fit_gmm_threshold() existente)
         → pred_labels (N,)
         → save_colored_ply()
```

### Saídas

```
results_sf/  comparacao_thresholds_<ts>.csv
             relatorio_severidade_<ts>.csv
             ply/<filename>_pred.ply      ← rachaduras em vermelho
visualizations_sf/  metrics_<ts>.json
                    predictions_<ts>.csv
                    training_history_<ts>.csv
                    score_distribution_<ts>.png
```

---

## Seção 2 — `src/teacher_student_v2.py`

Fork do v1 com três pontos de integração cirúrgicos. Arquitetura Teacher-Student inalterada.

### Diferenças v1 → v2

| Componente | v1 | v2 |
|---|---|---|
| Scalar field gate | Hard, global `[sf_min, sf_max]` | Soft, GMM por nuvem |
| Score final | `distill_score` normalizado | `0.7·distill_norm + 0.3·sf_gmm` |
| Push-Pull | `y` binário flat | `y · sf_confidence` ponderado |
| Saída PLY | Não existe | Rachaduras em vermelho |
| Nuvens unimodais | Gate desativado | Transparente por design |

### Ponto 1 — Score fusion (em `compute_anomaly_scores`)

```python
sf_gmm    = ScalarFieldGMM(scalar_field_per_point)
sf_score  = sf_gmm.anomaly_probability()   # (N,) ∈ [0,1]
dist_norm = min_max_norm(distill_score)
score     = 0.7 * dist_norm + 0.3 * sf_score
```

Para nuvens unimodais: `sf_score ≈ uniforme` → degrada gracefully para v1.

### Ponto 2 — Gate soft (substitui `apply_scalar_field_gate`)

```python
# v2: modula score antes do threshold (em vez de zerar pred depois)
soft_weight    = ScalarFieldGMM(scalar_field).soft_weights()  # (N,) ∈ [0,1]
score_modulated = score * soft_weight
pred = (score_modulated > threshold).astype(int)
```

### Ponto 3 — Push-Pull ponderado (fase supervisionada)

```python
sf_confidence = ScalarFieldGMM(x[:, 9]).pseudo_label_confidence()  # (N,)
y_weighted    = y * sf_confidence   # pontos no vale recebem gradiente reduzido
loss = push_pull(teacher_scales, student_scales, y_weighted)
```

Motivo: pontos na fronteira rachadura/normal têm scalar_field ambíguo — reduzir gradiente
nessas regiões evita que o modelo aprenda fronteiras ruidosas das anotações.

### Saídas

```
results_ts/ply/<filename>_pred_v2.ply    ← rachaduras em vermelho
results_ts/model_comparison_<ts>.csv    ← v1 vs v2 vs unsup
```

---

## Seção 3 — Adições em `utils/evaluation.py`

Todas as funções existentes permanecem intactas (backward compatible).

### `ScalarFieldGMM` (classe)

```python
class ScalarFieldGMM:
    def __init__(self, scalar: np.ndarray, n_components: int = 2)
    def fit(self) -> 'ScalarFieldGMM'
    def anomaly_probability(self) -> np.ndarray       # (N,) ∈ [0,1]
    def pseudo_label_confidence(self) -> np.ndarray   # (N,) ∈ [0,1]
    def soft_weights(self) -> np.ndarray              # (N,) — gate soft
    def crack_interval(self) -> tuple[float, float]   # (x_min, x_max)
    @property
    def threshold(self) -> float
    @property
    def modality(self) -> str   # 'bimodal' | 'unimodal'
```

Implementação interna: valley_detection com fallback para GMM sklearn.

### `save_colored_ply()`

```python
def save_colored_ply(
    xyz: np.ndarray,           # (N, 3)
    rgb_orig: np.ndarray,      # (N, 3) float32 ∈ [0,1]
    pred_labels: np.ndarray,   # (N,) int
    path: str,
    crack_color: tuple = (255, 0, 0)
) -> None
```

Pontos normais: cor original preservada. Pontos de rachadura: `crack_color` (vermelho padrão).

### `evaluate_ablation()`

Compara contribuição marginal de cada componente do v2 — pronto para tabela de ablation do TCC:

```python
configs = [
    {'name': 'distill_only',    'use_sf_fusion': False, 'use_sf_gate': False},
    {'name': 'distill+gate',    'use_sf_fusion': False, 'use_sf_gate': True },
    {'name': 'distill+fusion',  'use_sf_fusion': True,  'use_sf_gate': False},
    {'name': 'v2_completo',     'use_sf_fusion': True,  'use_sf_gate': True },
]
# Retorna: pd.DataFrame com F1, IoU, AUROC, AP por config
```

### `compare_models()`

```python
def compare_models(
    results_v1: list,
    results_v2: list,
    results_unsup: list,
    output_dir: str
) -> pd.DataFrame
```

Produz tabela de métricas + teste de Wilcoxon pareado (reutiliza `statistical_comparison()`).
Salva `model_comparison_<ts>.csv`.

---

## Protocolo de avaliação para o TCC

```
Comparação principal:
  v1  (Teacher-Student, gate hard, sem fusion)
  v2  (Teacher-Student, gate soft GMM, score fusion, Push-Pull ponderado)
  unsup (Teacher frozen + DGCNN self-supervised)

Ablation (apenas v2):
  distill_only → distill+gate → distill+fusion → v2_completo

Métricas reportadas:
  Threshold-dependent : Precision, Recall, F1, F1-macro, IoU
  Threshold-free      : AUROC, Average Precision
  Geométrica          : Chamfer Distance
  Estatística         : Wilcoxon pareado com correção Bonferroni

Saída visual:
  PLY colorido por nuvem (vermelho = rachadura predita, original = normal)
```

---

## Restrições e decisões

- **Sem alteração ao v1**: backward compatible total.
- **Nuvens unimodais**: `ScalarFieldGMM` degrada para comportamento neutro (weights=1) —
  nunca piora o resultado vs. v1 em arquivos com `scalar_R`.
- **Pseudo-labels não são labels humanos**: derivados da distribuição física do scalar_field
  (valley detection automático) — protocolo legitimamente self-supervised.
- **DGCNN k=20**: mesmo valor do DGCNN original (Wang et al., 2019). Grafo reconstruído por
  bloco para capturar estrutura local em escalas diferentes de features.
- **Cores no PLY**: vermelho `(255,0,0)` para rachaduras, cor original preservada para normais.
  Configurável via parâmetro `crack_color`.
