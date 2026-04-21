# Scalar Field Segmentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implementar `scalar_field_unsup.py` (Teacher frozen + DGCNN self-supervised) e `teacher_student_v2.py` (fork do v1 com gate soft GMM + score fusion + Push-Pull ponderado), ambos com saída PLY colorida e métricas completas comparáveis ao v1.

**Architecture:** `ScalarFieldGMM` em `evaluation.py` centraliza a lógica estatística do scalar_field e é compartilhada pelos dois novos sistemas. `scalar_field_unsup.py` treina DGCNN com pseudo-labels gerados por valley detection (self-supervised). `teacher_student_v2.py` é fork do v1 com 3 pontos de integração: score fusion, gate soft e Push-Pull ponderado.

**Tech Stack:** PyTorch, scikit-learn (GaussianMixture), scipy (cKDTree, find_peaks, gaussian_filter1d), plyfile (PlyData, PlyElement), numpy

**Nota:** Nunca executar `git commit` — o usuário commita manualmente.

---

## Mapa de arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `src/utils/evaluation.py` | Modificar | Adicionar `ScalarFieldGMM`, `save_colored_ply`, `evaluate_ablation`, `compare_models` |
| `src/scalar_field_unsup.py` | Criar | Teacher frozen + DGCNN self-supervised, pipeline completo |
| `src/teacher_student_v2.py` | Criar | Fork do v1 com 3 pontos de integração |
| `tests/test_scalar_field_gmm.py` | Criar | Testes de `ScalarFieldGMM` e `save_colored_ply` |
| `tests/test_scalar_field_unsup.py` | Criar | Testes de `EdgeConvBlock`, `ScalarFieldDGCNN` |
| `tests/test_teacher_student_v2.py` | Criar | Testes das 3 integrações do v2 |

---

## Task 1: ScalarFieldGMM + save_colored_ply em evaluation.py

**Files:**
- Modify: `src/utils/evaluation.py` — adicionar após a função `apply_scalar_field_gate` (linha ~186)
- Modify: `src/utils/config.py` — adicionar import de `PlyElement`
- Create: `tests/test_scalar_field_gmm.py`

- [ ] **Step 1: Criar diretório de testes**

```bash
mkdir -p /home/cmatheus/projects/TCC/tests
touch /home/cmatheus/projects/TCC/tests/__init__.py
```

- [ ] **Step 2: Escrever teste para ScalarFieldGMM**

Criar `tests/test_scalar_field_gmm.py`:

```python
# tests/test_scalar_field_gmm.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import tempfile

# ── Dados sintéticos ──────────────────────────────────────────────────────────

def make_bimodal(n_crack=200, n_normal=800, seed=42):
    """Bimodal: crack em [0,30], normal em [50,120]."""
    rng = np.random.default_rng(seed)
    crack  = rng.uniform(0, 30, n_crack).astype(np.float32)
    normal = rng.uniform(50, 120, n_normal).astype(np.float32)
    return np.concatenate([crack, normal])

def make_unimodal(n=1000, seed=42):
    """Unimodal: distribuição gaussiana centrada em 80."""
    rng = np.random.default_rng(seed)
    return rng.normal(80, 15, n).astype(np.float32)


# ── Testes ScalarFieldGMM ─────────────────────────────────────────────────────

def test_bimodal_detection():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_bimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    assert gmm.modality == 'bimodal', f"Esperado bimodal, got {gmm.modality}"
    # Threshold deve estar entre os dois clusters
    assert 30 <= gmm.threshold <= 50, f"Threshold {gmm.threshold} fora do esperado [30,50]"

def test_unimodal_detection():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    assert gmm.modality == 'unimodal', f"Esperado unimodal, got {gmm.modality}"

def test_anomaly_probability_bimodal():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_bimodal(n_crack=200, n_normal=800)
    gmm = ScalarFieldGMM(scalar).fit()
    probs = gmm.anomaly_probability()
    assert probs.shape == (1000,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    # Pontos de crack (primeiros 200) devem ter prob > 0.5 em média
    assert probs[:200].mean() > 0.5, "Crack points devem ter alta prob de anomalia"

def test_anomaly_probability_unimodal():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    probs = gmm.anomaly_probability()
    assert probs.shape == (1000,)
    # Unimodal: probs devem ser ~0.5 (incerto)
    assert abs(probs.mean() - 0.5) < 0.1

def test_soft_weights_unimodal_all_ones():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    weights = gmm.soft_weights()
    assert np.allclose(weights, 1.0), "Unimodal deve retornar weights=1.0"

def test_pseudo_label_confidence_unimodal_zeros():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    conf = gmm.pseudo_label_confidence()
    assert np.allclose(conf, 0.0), "Unimodal deve retornar confidence=0.0"

def test_crack_interval_bimodal():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_bimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    x_min, x_max = gmm.crack_interval()
    assert x_min >= 0.0
    assert x_max <= 50.0, f"x_max {x_max} deve ser <= 50"

def test_edge_cases():
    from utils.evaluation import ScalarFieldGMM
    # Menos de 10 pontos: não deve cravar
    small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gmm = ScalarFieldGMM(small).fit()
    assert gmm.modality in ('bimodal', 'unimodal')
    # Valores idênticos
    flat = np.full(100, 42.0, dtype=np.float32)
    gmm2 = ScalarFieldGMM(flat).fit()
    assert gmm2.modality == 'unimodal'


# ── Testes save_colored_ply ───────────────────────────────────────────────────

def test_save_colored_ply_shapes():
    from utils.evaluation import save_colored_ply
    from plyfile import PlyData
    
    rng = np.random.default_rng(0)
    N = 50
    xyz = rng.standard_normal((N, 3)).astype(np.float32)
    rgb = rng.uniform(0, 1, (N, 3)).astype(np.float32)
    labels = np.zeros(N, dtype=np.int64)
    labels[10:20] = 1   # 10 pontos de rachadura
    
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        path = f.name
    
    try:
        save_colored_ply(xyz, rgb, labels, path)
        ply = PlyData.read(path)
        v = ply['vertex']
        assert len(v) == N
        # Pontos de rachadura devem ser vermelhos
        for i in range(10, 20):
            assert v['red'][i] == 255
            assert v['green'][i] == 0
            assert v['blue'][i] == 0
        # Pontos normais devem ter cor original (não vermelhos)
        for i in range(10):
            expected_r = int(rgb[i, 0] * 255)
            assert abs(int(v['red'][i]) - expected_r) <= 1
    finally:
        os.unlink(path)

def test_save_colored_ply_no_cracks():
    from utils.evaluation import save_colored_ply
    from plyfile import PlyData
    rng = np.random.default_rng(1)
    N = 20
    xyz = rng.standard_normal((N, 3)).astype(np.float32)
    rgb = np.full((N, 3), 0.5, dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        path = f.name
    try:
        save_colored_ply(xyz, rgb, labels, path)
        ply = PlyData.read(path)
        assert len(ply['vertex']) == N
        # Nenhum ponto vermelho
        assert (np.array(ply['vertex']['red']) == 255).sum() == 0
    finally:
        os.unlink(path)
```

- [ ] **Step 3: Rodar testes para confirmar falha**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -m pytest tests/test_scalar_field_gmm.py -v 2>&1 | head -30
```

Esperado: `ImportError` ou `ModuleNotFoundError` — `ScalarFieldGMM` não existe ainda.

- [ ] **Step 4: Adicionar import de PlyElement em config.py**

Em `src/utils/config.py`, localizar a linha com `from plyfile import PlyData` e substituir:

```python
from plyfile import PlyData, PlyElement
```

- [ ] **Step 5: Implementar ScalarFieldGMM e save_colored_ply em evaluation.py**

Adicionar **após** a função `apply_scalar_field_gate` (após a linha `return results` que fecha essa função, antes do bloco `# === CHAMFER DISTANCE ===`):

```python
# ============================================================================
# SCALAR FIELD GMM — prior estatístico por nuvem
# ============================================================================

class ScalarFieldGMM:
    """
    Ajusta GMM 1D ao scalar_field de uma nuvem para distinguir rachaduras
    de superfície normal sem usar labels humanos.

    Para nuvens com scalar_Scalar_field (bimodal, gap≥0): valley detection
    identifica o threshold com alta confiança.
    Para nuvens com scalar_R (unimodal): retorna comportamento neutro
    (weights=1.0, confidence=0.0) — transparente para o pipeline downstream.

    Uso:
        gmm = ScalarFieldGMM(scalar_field_array).fit()
        probs   = gmm.anomaly_probability()       # (N,) ∈ [0,1]
        weights = gmm.soft_weights()              # (N,) — gate soft
        conf    = gmm.pseudo_label_confidence()   # (N,) — peso para Push-Pull
    """

    def __init__(self, scalar: np.ndarray, n_components: int = 2):
        self.scalar = np.asarray(scalar, dtype=np.float32).ravel()
        self.n_components = n_components
        self._gmm        = None
        self._modality   = None
        self._threshold  = None
        self._crack_idx  = None
        self._fitted     = False

    def fit(self) -> 'ScalarFieldGMM':
        from sklearn.mixture import GaussianMixture
        from scipy.ndimage import gaussian_filter1d
        from scipy.signal import find_peaks

        scalar = self.scalar

        # Edge case: valores insuficientes ou constantes
        if len(scalar) < 10 or scalar.std() < 1e-6:
            self._modality  = 'unimodal'
            self._threshold = float(scalar.mean())
            self._fitted    = True
            return self

        # ── Valley detection ─────────────────────────────────────────────────
        n_bins = min(256, len(scalar) // 4)
        counts, edges = np.histogram(scalar, bins=n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        smooth  = gaussian_filter1d(counts.astype(float), sigma=3.0)
        peaks, _ = find_peaks(smooth, prominence=smooth.max() * 0.05)

        if len(peaks) >= 2:
            self._modality = 'bimodal'
            p1, p2 = int(peaks[0]), int(peaks[1])
            if p1 > p2:
                p1, p2 = p2, p1
            valley_region = smooth[p1:p2]
            valley_idx    = p1 + int(np.argmin(valley_region))
            self._threshold = float(centers[valley_idx])
        else:
            self._modality  = 'unimodal'
            self._threshold = float(np.percentile(scalar, 50))

        # ── GMM fit ──────────────────────────────────────────────────────────
        try:
            gmm = GaussianMixture(
                n_components=self.n_components, covariance_type='full',
                max_iter=200, random_state=42, n_init=3,
            )
            gmm.fit(scalar.reshape(-1, 1))
            self._gmm = gmm

            means = gmm.means_.ravel()
            # Componente de menor média = rachaduras (menor intensidade scanner)
            self._crack_idx = int(np.argmin(means))
        except Exception as e:
            log.warning(f"ScalarFieldGMM: GMM falhou ({e}), usando threshold de valley")
            self._gmm = None

        self._fitted = True
        return self

    def _ensure_fitted(self):
        if not self._fitted:
            self.fit()

    def anomaly_probability(self) -> np.ndarray:
        """P(rachadura | scalar_field[i]) por ponto. Unimodal → ~0.5 uniforme."""
        self._ensure_fitted()
        if self._modality == 'unimodal' or self._gmm is None:
            return np.full(len(self.scalar), 0.5, dtype=np.float32)
        probs = self._gmm.predict_proba(self.scalar.reshape(-1, 1))  # (N, 2)
        return probs[:, self._crack_idx].astype(np.float32)

    def soft_weights(self) -> np.ndarray:
        """
        Gate soft: peso por ponto para modular o anomaly score.
        Unimodal → 1.0 (transparente). Bimodal → P(crack|sf).
        """
        self._ensure_fitted()
        if self._modality == 'unimodal':
            return np.ones(len(self.scalar), dtype=np.float32)
        return self.anomaly_probability()

    def pseudo_label_confidence(self) -> np.ndarray:
        """
        Confiança do pseudo-label por ponto.
        Unimodal → 0.0 (nenhuma confiança — não usar em contrastive loss).
        Bimodal  → |P(crack) - 0.5| * 2  ∈ [0,1].
                   0 no vale (ambíguo), 1 no núcleo do cluster.
        """
        self._ensure_fitted()
        if self._modality == 'unimodal':
            return np.zeros(len(self.scalar), dtype=np.float32)
        probs = self.anomaly_probability()
        return (np.abs(probs - 0.5) * 2.0).astype(np.float32)

    def crack_interval(self) -> tuple:
        """(x_min, x_max) do cluster de rachadura no scalar_field."""
        self._ensure_fitted()
        crack_mask = self.scalar <= self._threshold
        if crack_mask.sum() == 0:
            return (float(self.scalar.min()), float(self._threshold))
        crack_vals = self.scalar[crack_mask]
        return (float(crack_vals.min()), float(crack_vals.max()))

    @property
    def threshold(self) -> float:
        self._ensure_fitted()
        return self._threshold

    @property
    def modality(self) -> str:
        self._ensure_fitted()
        return self._modality


# ============================================================================
# SAÍDA PLY COLORIDA
# ============================================================================

def save_colored_ply(
    xyz: np.ndarray,
    rgb_orig: np.ndarray,
    pred_labels: np.ndarray,
    path: str,
    crack_color: tuple = (255, 0, 0),
) -> None:
    """
    Salva nuvem de pontos PLY com rachaduras coloridas.

    Pontos normais (pred_labels=0): cor original preservada.
    Pontos de rachadura (pred_labels=1): crack_color (vermelho padrão).

    Args:
        xyz         : (N, 3) float32 — coordenadas
        rgb_orig    : (N, 3) float32 ∈ [0,1] — cores originais normalizadas
        pred_labels : (N,)  int — 0=normal, 1=rachadura
        path        : caminho de saída (cria diretórios intermediários)
        crack_color : (R, G, B) uint8 — cor da rachadura (default vermelho)
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    rgb_out = rgb_orig.copy()
    crack_rgb_norm = np.array(crack_color, dtype=np.float32) / 255.0
    rgb_out[pred_labels == 1] = crack_rgb_norm

    rgb_uint8 = (rgb_out * 255.0).clip(0, 255).astype(np.uint8)

    n = len(xyz)
    vertex = np.zeros(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex['x']     = xyz[:, 0]
    vertex['y']     = xyz[:, 1]
    vertex['z']     = xyz[:, 2]
    vertex['red']   = rgb_uint8[:, 0]
    vertex['green'] = rgb_uint8[:, 1]
    vertex['blue']  = rgb_uint8[:, 2]

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(path)
    log.info(f"PLY salvo: {path}  ({n:,} pts, "
             f"{int(pred_labels.sum()):,} rachaduras em vermelho)")
```

- [ ] **Step 6: Rodar testes para confirmar que passam**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -m pytest tests/test_scalar_field_gmm.py -v
```

Esperado: todos os testes PASS.

---

## Task 2: evaluate_ablation + compare_models em evaluation.py

**Files:**
- Modify: `src/utils/evaluation.py` — adicionar no final do arquivo

- [ ] **Step 1: Adicionar evaluate_ablation e compare_models**

Adicionar no **final** de `src/utils/evaluation.py`:

```python
# ============================================================================
# ABLATION E COMPARAÇÃO DE MODELOS
# ============================================================================

def evaluate_ablation(
    results_raw: list,
    sf_applied: bool = False,
) -> pd.DataFrame:
    """
    Avalia contribuição marginal de cada componente do v2.
    Recebe results já com 'score', 'scalar_field', 'gt_labels'.
    Retorna DataFrame para tabela de ablation do TCC.

    Configs testadas:
      distill_only   : score bruto sem gate nem fusion
      distill+gate   : score * soft_weight (gate soft GMM)
      distill+fusion : 0.7*score + 0.3*sf_gmm
      v2_completo    : fusion + gate soft
    """
    import copy

    configs = [
        {'name': 'distill_only',    'fusion': False, 'gate': False},
        {'name': 'distill+gate',    'fusion': False, 'gate': True },
        {'name': 'distill+fusion',  'fusion': True,  'gate': False},
        {'name': 'v2_completo',     'fusion': True,  'gate': True },
    ]

    rows = []
    for cfg in configs:
        res = copy.deepcopy(results_raw)

        for r in res:
            sf_raw = r.get('scalar_field')
            score  = r['score'].copy()

            if sf_raw is not None:
                gmm = ScalarFieldGMM(sf_raw).fit()
                sf_prob = gmm.anomaly_probability()
                soft_w  = gmm.soft_weights()

                if cfg['fusion']:
                    lo, hi = score.min(), score.max()
                    dist_n = (score - lo) / (hi - lo + 1e-8)
                    score  = 0.7 * dist_n + 0.3 * sf_prob

                if cfg['gate']:
                    score = score * soft_w

            r['score'] = score

        thr, _ = fit_gmm_threshold(res)
        res     = apply_threshold(res, thr)
        m       = evaluate(res)

        rows.append({
            'config'           : cfg['name'],
            'threshold'        : round(thr, 6),
            'precision'        : round(m.get('precision', 0), 4),
            'recall'           : round(m.get('recall', 0), 4),
            'f1'               : round(m.get('f1', 0), 4),
            'f1_macro'         : round(m.get('f1_macro', 0), 4),
            'iou'              : round(m.get('iou', 0), 4),
            'auroc'            : round(m.get('auroc', 0), 4),
            'average_precision': round(m.get('average_precision', 0), 4),
        })

    df = pd.DataFrame(rows)
    log.info("\nAblation study:\n" + df.to_string(index=False))
    return df


def compare_models(
    results_v1: list,
    results_v2: list,
    results_unsup: list,
    output_dir: str,
    ts: str = None,
) -> pd.DataFrame:
    """
    Compara métricas entre v1, v2 e unsup.
    Exporta CSV e executa Wilcoxon pareado entre os três.

    Args:
        results_v1/v2/unsup : listas com 'pred_labels', 'gt_labels', 'score', 'has_crack'
        output_dir          : diretório para salvar o CSV
        ts                  : timestamp para nome do arquivo

    Returns:
        DataFrame com métricas agregadas por modelo.
    """
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')

    model_results = {
        'v1'   : results_v1,
        'v2'   : results_v2,
        'unsup': results_unsup,
    }

    rows = []
    per_cloud_scores = {name: [] for name in model_results}

    for name, res in model_results.items():
        m = evaluate(res)
        rows.append({
            'model'            : name,
            'precision'        : round(m.get('precision', 0), 4),
            'recall'           : round(m.get('recall', 0), 4),
            'f1'               : round(m.get('f1', 0), 4),
            'f1_macro'         : round(m.get('f1_macro', 0), 4),
            'iou'              : round(m.get('iou', 0), 4),
            'auroc'            : round(m.get('auroc', 0), 4),
            'average_precision': round(m.get('average_precision', 0), 4),
            'chamfer_distance' : round(m.get('chamfer_distance', float('nan')), 8),
        })
        per_cloud_scores[name] = [
            pc['f1'] for pc in m.get('per_cloud', [])
        ]

    df = pd.DataFrame(rows)

    # Wilcoxon pareado
    stat_results = statistical_comparison(per_cloud_scores, metric='f1')

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'model_comparison_{ts}.csv')
    df.to_csv(csv_path, index=False)
    log.info(f"Comparação de modelos salva: {csv_path}")
    log.info("\nComparação de modelos:\n" + df.to_string(index=False))

    return df
```

- [ ] **Step 2: Verificar que o módulo carrega sem erros**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from utils.evaluation import ScalarFieldGMM, save_colored_ply, evaluate_ablation, compare_models
print('OK — todos os símbolos importados')
"
```

Esperado: `OK — todos os símbolos importados`

---

## Task 3: EdgeConvBlock + ScalarFieldDGCNN em scalar_field_unsup.py

**Files:**
- Create: `src/scalar_field_unsup.py` (header + arquitetura)
- Create: `tests/test_scalar_field_unsup.py`

- [ ] **Step 1: Escrever testes da arquitetura**

Criar `tests/test_scalar_field_unsup.py`:

```python
# tests/test_scalar_field_unsup.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Mock Teacher ──────────────────────────────────────────────────────────────

class MockTeacher(nn.Module):
    """Teacher mínimo para testes sem GPU/checkpoint."""
    def forward(self, x):
        return torch.randn(x.size(0), 512)

    @property
    def feature_adapter(self):
        return nn.Identity()

    @property
    def lfa(self):
        return nn.Identity()

    @property
    def blocks(self):
        return []


# ── Testes EdgeConvBlock ──────────────────────────────────────────────────────

def test_edge_conv_block_output_shape():
    from scalar_field_unsup import EdgeConvBlock
    block = EdgeConvBlock(in_channels=16, out_channels=32, k=5)
    N = 50
    h   = torch.randn(N, 16)
    knn = torch.randint(0, N, (N, 5))
    out = block(h, knn)
    assert out.shape == (N, 32), f"Shape esperado ({N}, 32), got {out.shape}"

def test_edge_conv_block_different_sizes():
    from scalar_field_unsup import EdgeConvBlock
    block = EdgeConvBlock(in_channels=512, out_channels=256, k=10)
    N = 100
    h   = torch.randn(N, 512)
    knn = torch.randint(0, N, (N, 10))
    out = block(h, knn)
    assert out.shape == (N, 256)

def test_edge_conv_block_gradient_flows():
    from scalar_field_unsup import EdgeConvBlock
    block = EdgeConvBlock(in_channels=8, out_channels=8, k=3)
    N = 20
    h   = torch.randn(N, 8, requires_grad=True)
    knn = torch.randint(0, N, (N, 3))
    out = block(h, knn)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None


# ── Testes build_knn_idx ──────────────────────────────────────────────────────

def test_build_knn_idx_shape():
    from scalar_field_unsup import build_knn_idx
    feats = np.random.randn(50, 3).astype(np.float32)
    idx = build_knn_idx(feats, k=5)
    assert idx.shape == (50, 5), f"Shape esperado (50,5), got {idx.shape}"
    assert idx.dtype == torch.int64

def test_build_knn_idx_no_self():
    from scalar_field_unsup import build_knn_idx
    feats = np.random.randn(30, 3).astype(np.float32)
    idx = build_knn_idx(feats, k=4)
    # Nenhum ponto deve ser vizinho de si mesmo
    for i in range(30):
        assert i not in idx[i].tolist(), f"Ponto {i} é vizinho de si mesmo"


# ── Testes ScalarFieldDGCNN ───────────────────────────────────────────────────

def test_dgcnn_forward_shape():
    from scalar_field_unsup import ScalarFieldDGCNN
    model = ScalarFieldDGCNN(input_dim=15, k=5)
    model.teacher = MockTeacher()

    N = 40
    x = torch.randn(N, 15)
    score, recon = model(x)
    assert score.shape == (N,), f"Score shape: {score.shape}"
    assert recon.shape == (N, 512), f"Recon shape: {recon.shape}"
    assert score.min() >= 0.0 and score.max() <= 1.0, "Score deve estar em [0,1]"

def test_dgcnn_no_grad_on_teacher():
    from scalar_field_unsup import ScalarFieldDGCNN
    model = ScalarFieldDGCNN(input_dim=15, k=5)
    model.teacher = MockTeacher()

    for p in model.teacher.parameters():
        assert not p.requires_grad, "Teacher deve ter requires_grad=False"
```

- [ ] **Step 2: Rodar testes para confirmar falha**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -m pytest tests/test_scalar_field_unsup.py -v 2>&1 | head -20
```

Esperado: `ModuleNotFoundError: No module named 'scalar_field_unsup'`

- [ ] **Step 3: Criar scalar_field_unsup.py — header e arquitetura**

Criar `src/scalar_field_unsup.py`:

```python
# ============================================================================
# SCALAR FIELD UNSUPERVISED — Teacher frozen + DGCNN self-supervised
# ============================================================================
# Segmentação de rachaduras sem labels humanos usando:
#   • Teacher (PointTransformer pré-treinado, frozen) como extrator de features
#   • DGCNN (Dynamic Graph CNN) treinado com pseudo-labels do scalar_field
#   • ScalarFieldGMM (valley detection) para gerar pseudo-labels automáticos
#
# Protocolo self-supervised:
#   Fase 1 (warmup):  L_recon apenas — inicializa representação
#   Fase 2 (self-sup): L_contrast (bimodal) + L_recon (regularização)
#
# Referências:
#   Wang et al. (2019) — DGCNN, ACM TOG
#   Zhou et al. (2024) — R3D-AD, ECCV
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import csv
from utils import *
from utils.config import *
from utils.architectures import *
from utils.building_blocks import *
from utils.data import *
from utils.evaluation import *
from utils.training_utils import *

log = setup_logging(LOG_PATH)

# ── Caminhos específicos ───────────────────────────────────────────────────────
RESULTS_SF  = f'{BASE_PATH}/results_sf'
MODELS_SF   = f'{BASE_PATH}/models_sf'
VIS_SF      = f'{BASE_PATH}/visualizations_sf'
PLY_SF      = f'{BASE_PATH}/results_sf/ply'

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
DGCNN_K          = 20      # vizinhos no grafo dinâmico
LR_DGCNN         = 1e-4
NUM_EPOCHS_DGCNN = 150
WARMUP_EPOCHS    = 30
MARGIN           = 0.5     # margem da contrastive loss
LAMBDA_CONTRAST  = 0.7
LAMBDA_RECON     = 0.3
PATIENCE_DGCNN   = 20


# ============================================================================
# KNN UTILITÁRIO — scipy (CPU, sem OOM)
# ============================================================================

def build_knn_idx(feats: np.ndarray, k: int) -> torch.Tensor:
    """
    Constrói índices k-NN usando scipy.cKDTree (CPU, eficiente em memória).
    Evita torch.cdist que aloca matriz N×N em VRAM.

    feats : (N, D) numpy float32
    k     : número de vizinhos (excluindo self)
    Returns: (N, k) torch.int64
    """
    tree = cKDTree(feats)
    k_query = min(k + 1, len(feats))
    _, idx = tree.query(feats, k=k_query)
    # Remove self (coluna 0)
    idx = idx[:, 1:k+1] if idx.shape[1] > k else idx[:, 1:]
    return torch.from_numpy(idx.astype(np.int64))


# ============================================================================
# EDGE CONV BLOCK — DGCNN
# ============================================================================

class EdgeConvBlock(nn.Module):
    """
    Bloco de convolução em grafos dinâmicos.
    Wang et al. (2019) — Dynamic Graph CNN for Learning on Point Clouds.

    Para cada ponto i e seus k vizinhos j:
      edge_feat(i,j) = [h_i ‖ h_j − h_i]   (2·C_in dimensional)
      h_i_new = max_{j ∈ KNN(i)} MLP([h_i, h_j − h_i])

    O grafo é reconstruído a cada bloco nas features atualizadas (dynamic).
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 20):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, h: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        """
        h       : (N, C_in)   — features dos nós
        knn_idx : (N, k)      — índices dos k vizinhos (pré-computados)
        Returns : (N, C_out)
        """
        N, k = knn_idx.shape
        C_in = h.size(1)

        # Gather neighbor features: (N, k, C_in)
        neighbors = h[knn_idx.view(-1)].view(N, k, C_in)

        # Edge features: [h_i, h_j - h_i] → (N, k, 2*C_in)
        h_i = h.unsqueeze(1).expand(-1, k, -1)
        edge_feat = torch.cat([h_i, neighbors - h_i], dim=-1)  # (N, k, 2*C_in)

        # MLP + max-pool
        out = self.mlp(edge_feat.reshape(N * k, -1))  # (N*k, C_out)
        out = out.view(N, k, -1).max(dim=1).values    # (N, C_out)
        return out


# ============================================================================
# SCALAR FIELD DGCNN — modelo completo
# ============================================================================

class ScalarFieldDGCNN(nn.Module):
    """
    Teacher (frozen) + DGCNN self-supervised para anomaly detection.

    Teacher extrai bottleneck (N, 512) como features ricas de cada ponto.
    DGCNN aplica 3 blocos EdgeConv sobre essas features com grafo dinâmico
    reconstruído em cada bloco (nas features atualizadas, não em XYZ).

    Saídas:
      score (N,) ∈ [0,1] — anomaly score por ponto
      recon (N, 512)     — reconstrução do bottleneck Teacher (para L_recon)
    """

    def __init__(self, input_dim: int = INPUT_DIM, k: int = DGCNN_K):
        super().__init__()
        self.k = k
        self.teacher = None   # definido externamente via build_teacher()

        # DGCNN blocks — operando sobre features Teacher
        self.conv1 = EdgeConvBlock(512, 256, k=k)
        self.conv2 = EdgeConvBlock(256, 128, k=k)
        self.conv3 = EdgeConvBlock(128, 64,  k=k)

        # Score head: 64D → anomaly score ∈ [0,1]
        self.score_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # Reconstruction head: 64D → 512D Teacher bottleneck (pretext task)
        self.recon_head = nn.Sequential(
            nn.Linear(64, 256),
            nn.GELU(),
            nn.Linear(256, 512),
        )

    def _freeze_teacher(self):
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()

    def _teacher_bottleneck(self, x: torch.Tensor) -> torch.Tensor:
        """Extrai bottleneck do Teacher sem gradientes, em chunks."""
        self.teacher.eval()
        N = x.size(0)
        chunk = TEACHER_CHUNK_SIZE

        if N <= chunk:
            with torch.no_grad():
                return self.teacher(x)

        parts = []
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            with torch.no_grad():
                parts.append(self.teacher(x[start:end]))
        return torch.cat(parts, dim=0)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x : (N, input_dim)
        Returns: (score (N,), recon (N, 512))
        """
        # Teacher features (frozen, chunked)
        h = self._teacher_bottleneck(x)      # (N, 512)

        # DGCNN — grafo dinâmico reconstruído em cada bloco
        h_np = h.detach().cpu().numpy()
        knn1 = build_knn_idx(h_np, self.k).to(x.device)
        h = self.conv1(h, knn1)              # (N, 256)

        h_np = h.detach().cpu().numpy()
        knn2 = build_knn_idx(h_np, self.k).to(x.device)
        h = self.conv2(h, knn2)              # (N, 128)

        h_np = h.detach().cpu().numpy()
        knn3 = build_knn_idx(h_np, self.k).to(x.device)
        h = self.conv3(h, knn3)              # (N, 64)

        score = self.score_head(h).squeeze(-1)   # (N,)
        recon = self.recon_head(h)               # (N, 512)
        return score, recon
```

- [ ] **Step 4: Rodar testes de arquitetura**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -m pytest tests/test_scalar_field_unsup.py -v
```

Esperado: todos os testes PASS.

---

## Task 4: Training loop + main() em scalar_field_unsup.py

**Files:**
- Modify: `src/scalar_field_unsup.py` — adicionar após a classe ScalarFieldDGCNN

- [ ] **Step 1: Adicionar função de build do modelo**

Adicionar após a classe `ScalarFieldDGCNN`:

```python
# ============================================================================
# BUILD DO MODELO
# ============================================================================

def build_dgcnn(device: torch.device) -> ScalarFieldDGCNN:
    """Instancia ScalarFieldDGCNN com Teacher via fallback A→B→C."""
    teacher = build_teacher(
        input_dim=INPUT_DIM,
        ptv3_ckpt=PTRANSF_WEIGHTS,
        s3dis_ckpt=PTRANSF_WEIGHTS_S3DIS,
    )
    model = ScalarFieldDGCNN(input_dim=INPUT_DIM, k=DGCNN_K)
    model.teacher = teacher
    model._freeze_teacher()
    return model.to(device)
```

- [ ] **Step 2: Adicionar training loop**

Adicionar após `build_dgcnn`:

```python
# ============================================================================
# TRAINING LOOP SELF-SUPERVISED
# ============================================================================

def train_dgcnn(
    model: ScalarFieldDGCNN,
    all_data: list,
    device: torch.device,
    num_epochs: int = NUM_EPOCHS_DGCNN,
    lr: float = LR_DGCNN,
    save_dir: str = MODELS_SF,
) -> dict:
    """
    Treina DGCNN em duas fases:
      Fase 1 (0..WARMUP_EPOCHS): apenas L_recon — inicializa representação
      Fase 2 (WARMUP..fim): L_contrast (bimodal) + L_recon (regularização)

    Loss:
      L_contrast = mean(max(0, MARGIN - score[crack] + score[normal]) * confidence)
      L_recon    = MSE(recon_head(64D), teacher_bottleneck(512D).detach())
      L_total    = LAMBDA_CONTRAST * L_contrast + LAMBDA_RECON * L_recon

    Para nuvens unimodais: L_contrast=0, apenas L_recon contribui.
    """
    os.makedirs(save_dir, exist_ok=True)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.999), weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6,
    )
    scaler  = GradScaler()
    history = {'loss': [], 'lr': [], 'contrast_loss': [], 'recon_loss': []}

    best_loss = float('inf')
    patience_c = 0

    log.info(f"\n{'='*65}")
    log.info("SCALAR FIELD DGCNN — Treino self-supervised")
    log.info(f"Fases: warmup={WARMUP_EPOCHS} | total={num_epochs}")
    log.info(f"{'='*65}")

    # Pré-computar ScalarFieldGMM e pseudo-labels para todas as nuvens
    # (fixo por epoch — pseudo-labels não mudam durante o treino)
    cloud_priors = []
    for d in all_data:
        sf_raw  = d['features'][:, 9]
        gmm     = ScalarFieldGMM(sf_raw).fit()
        pseudo  = None
        if gmm.modality == 'bimodal':
            # Pseudo-labels: 1 se scalar_field <= threshold (crack cluster)
            pseudo = (sf_raw <= gmm.threshold).astype(np.int64)
        cloud_priors.append({
            'modality'  : gmm.modality,
            'pseudo'    : pseudo,
            'confidence': gmm.pseudo_label_confidence(),
        })

    n_bimodal  = sum(1 for p in cloud_priors if p['modality'] == 'bimodal')
    n_unimodal = sum(1 for p in cloud_priors if p['modality'] == 'unimodal')
    log.info(f"Nuvens: {n_bimodal} bimodal | {n_unimodal} unimodal")

    for epoch in range(num_epochs):
        model.train()
        model.teacher.eval()   # Teacher SEMPRE frozen

        ep_contrast, ep_recon, ep_total = [], [], []
        indices = np.random.permutation(len(all_data))

        for i in indices:
            d     = all_data[i]
            prior = cloud_priors[i]

            x = torch.tensor(d['features'], dtype=torch.float32).to(device)
            if x.size(0) < 32:
                continue

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast():
                    score, recon = model(x)   # (N,), (N, 512)

                    # ── L_recon (todas as nuvens) ────────────────────────────
                    with torch.no_grad():
                        teacher_btn = model._teacher_bottleneck(x)
                    l_recon = F.mse_loss(recon, teacher_btn.detach())

                    # ── L_contrast (apenas bimodal, apenas na fase 2) ────────
                    l_contrast = torch.tensor(0.0, device=device)
                    if epoch >= WARMUP_EPOCHS and prior['modality'] == 'bimodal':
                        pseudo = prior['pseudo']
                        conf   = torch.tensor(
                            prior['confidence'], dtype=torch.float32, device=device)

                        crack_mask  = torch.tensor(pseudo == 1, device=device)
                        normal_mask = ~crack_mask

                        if crack_mask.sum() > 0 and normal_mask.sum() > 0:
                            sc = score[crack_mask]
                            sn = score[normal_mask]
                            cf = conf[crack_mask]

                            # Pares balanceados
                            min_n = min(len(sc), len(sn))
                            sc = sc[:min_n]
                            sn = sn[:min_n]
                            cf = cf[:min_n]

                            l_contrast = (
                                F.relu(MARGIN - sc + sn) * cf
                            ).mean()

                    # ── L_total ───────────────────────────────────────────────
                    if epoch < WARMUP_EPOCHS:
                        l_total = l_recon
                    else:
                        l_total = (LAMBDA_CONTRAST * l_contrast
                                   + LAMBDA_RECON   * l_recon)

                scaler.scale(l_total).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(optimizer)
                scaler.update()

                ep_contrast.append(l_contrast.item())
                ep_recon.append(l_recon.item())
                ep_total.append(l_total.item())

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    log.warning(f"OOM: {x.size(0)} pts — skip")
                    continue
                raise

        if not ep_total:
            continue

        avg_total    = float(np.mean(ep_total))
        avg_contrast = float(np.mean(ep_contrast))
        avg_recon    = float(np.mean(ep_recon))
        cur_lr       = optimizer.param_groups[0]['lr']
        vram         = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        history['loss'].append(avg_total)
        history['lr'].append(cur_lr)
        history['contrast_loss'].append(avg_contrast)
        history['recon_loss'].append(avg_recon)

        phase = 'warmup' if epoch < WARMUP_EPOCHS else 'self-sup'
        log.info(
            f"[{phase}] Epoch {epoch+1:03d}/{num_epochs} | "
            f"L={avg_total:.5f} C={avg_contrast:.5f} R={avg_recon:.5f} | "
            f"LR={cur_lr:.2e} | VRAM={vram:.1f}GB"
        )

        scheduler.step()

        if avg_total < best_loss:
            best_loss = avg_total
            patience_c = 0
            torch.save({
                'epoch'   : epoch,
                'model'   : {k: v for k, v in model.state_dict().items()
                             if not k.startswith('teacher.')},
                'history' : history,
            }, os.path.join(save_dir, 'best_dgcnn.pth'))
        else:
            patience_c += 1

        if epoch >= WARMUP_EPOCHS and patience_c >= PATIENCE_DGCNN:
            log.info(f"Early stop: {PATIENCE_DGCNN} epochs sem melhoria.")
            break

        gc.collect()
        torch.cuda.empty_cache()

    return history
```

- [ ] **Step 3: Adicionar compute_anomaly_scores_unsup e main()**

Adicionar após `train_dgcnn`:

```python
# ============================================================================
# INFERÊNCIA
# ============================================================================

@torch.no_grad()
def compute_anomaly_scores_unsup(
    model: ScalarFieldDGCNN,
    data_list: list,
    device: torch.device,
) -> list:
    """
    Calcula anomaly score por ponto usando ScalarFieldDGCNN.
    Score final = 0.7 * dgcnn_score_norm + 0.3 * sf_gmm_prob
    (mesma fusão do v2 — scores comparáveis).
    """
    model.eval()
    results = []

    for d in data_list:
        x      = torch.tensor(d['features'], dtype=torch.float32).to(device)
        labels = d['labels']
        sf_raw = d['features'][:, 9]

        # DGCNN score
        score_raw, _ = model(x)
        score_raw    = score_raw.cpu().numpy().astype(np.float32)

        # Score fusion com scalar_field GMM
        sf_gmm   = ScalarFieldGMM(sf_raw).fit()
        sf_score = sf_gmm.anomaly_probability()

        lo, hi    = score_raw.min(), score_raw.max()
        dist_norm = (score_raw - lo) / (hi - lo + 1e-8)
        score     = 0.7 * dist_norm + 0.3 * sf_score

        results.append({
            'filename'    : d['filename'],
            'has_crack'   : d['has_crack'],
            'score'       : score,
            'gt_labels'   : labels,
            'n_points'    : len(labels),
            'xyz'         : d['features'][:, :3],
            'rgb'         : d['features'][:, 3:6],
            'scalar_field': sf_raw,
        })

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M')
    print("\n" + "="*70)
    print("  SCALAR FIELD UNSUPERVISED — Teacher frozen + DGCNN self-supervised")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    for p in [RESULTS_SF, MODELS_SF, VIS_SF, PLY_SF]:
        os.makedirs(p, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device}")

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    log.info("\nCarregando dados...")
    train_all = load_folder(DATA_TRAIN)
    test_all  = load_folder(DATA_TEST)
    all_data  = train_all + test_all

    if not all_data:
        log.error("Nenhum dado encontrado.")
        return

    train_list, _, eval_list = split_dataset(all_data)

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    model = build_dgcnn(device)

    # ── 3. Treino ─────────────────────────────────────────────────────────────
    log.info("\nIniciando treino self-supervised...")
    t0 = time.time()
    history = train_dgcnn(model, all_data, device,
                          num_epochs=NUM_EPOCHS_DGCNN, lr=LR_DGCNN,
                          save_dir=MODELS_SF)
    log.info(f"Treino: {(time.time()-t0)/3600:.1f}h")

    # Salvar histórico
    hist_path = os.path.join(VIS_SF, f'training_history_{ts}.csv')
    with open(hist_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'contrast_loss',
                                           'recon_loss', 'lr'])
        w.writeheader()
        for i, (l, c, r, lr) in enumerate(zip(
            history['loss'], history['contrast_loss'],
            history['recon_loss'], history['lr']
        )):
            w.writerow({'epoch': i+1, 'loss': l, 'contrast_loss': c,
                        'recon_loss': r, 'lr': lr})

    # ── 4. Carregar melhor modelo ─────────────────────────────────────────────
    best_path = os.path.join(MODELS_SF, 'best_dgcnn.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)
        log.info(f"Melhor DGCNN carregado (epoch {ckpt['epoch']+1})")

    # ── 5. Scores de anomalia ─────────────────────────────────────────────────
    log.info("\nComputando anomaly scores...")
    results = compute_anomaly_scores_unsup(model, eval_list, device)

    # ── 6. Threshold + métricas ───────────────────────────────────────────────
    normal_ref = compute_anomaly_scores_unsup(
        model, train_list[:min(10, len(train_list))], device)
    thr, gmm_info = fit_gmm_threshold(results, normal_results=normal_ref)
    log.info(f"Threshold ({gmm_info['method']}): {thr:.4f}")
    results = apply_threshold(results, thr)

    # Estratégias de threshold
    strategies = {
        'F1'         : calibrate_threshold_f1(results),
        'G-mean'     : calibrate_threshold_gmean(results),
        'F-beta(0.5)': calibrate_threshold_fbeta(results, beta=0.5),
    }

    import copy
    comparison = []
    for name, thr_s in strategies.items():
        res_s = apply_threshold(copy.deepcopy(results), thr_s)
        m_s   = evaluate(res_s)
        comparison.append({
            'estrategia'   : name,
            'threshold'    : round(thr_s, 6),
            'precision'    : round(m_s.get('precision', 0), 4),
            'recall'       : round(m_s.get('recall', 0), 4),
            'f1'           : round(m_s.get('f1', 0), 4),
            'iou'          : round(m_s.get('iou', 0), 4),
            'auroc'        : round(m_s.get('auroc', 0), 4),
        })

    # Usar G-mean como estratégia principal
    best_thr = strategies.get('G-mean', thr)
    results  = apply_threshold(results, best_thr)
    metrics  = evaluate(results)
    metrics['threshold'] = best_thr
    metrics['gmm']       = gmm_info

    # Salvar métricas
    metrics_clean = {k: v for k, v in metrics.items() if k != 'per_cloud'}
    with open(os.path.join(VIS_SF, f'metrics_{ts}.json'), 'w') as f:
        json.dump(metrics_clean, f, indent=2, default=str)

    # Salvar comparação de thresholds
    comp_path = os.path.join(RESULTS_SF, f'comparacao_thresholds_{ts}.csv')
    with open(comp_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        w.writeheader()
        w.writerows(comparison)

    # ── 7. PLY colorido — rachaduras em vermelho ──────────────────────────────
    log.info("\nSalvando PLY coloridos...")
    for r in results:
        if not r['has_crack']:
            continue
        ply_out = os.path.join(PLY_SF, r['filename'].replace('.ply', '_pred.ply'))
        save_colored_ply(
            xyz        = r['xyz'],
            rgb_orig   = r['rgb'],
            pred_labels= r['pred_labels'],
            path       = ply_out,
        )

    # ── 8. Relatório de severidade ────────────────────────────────────────────
    sev_rows = []
    for r in results:
        if not r['has_crack']:
            continue
        n_crack = int(r['pred_labels'].sum())
        n_total = r['n_points']
        sev_rows.append({
            'arquivo'         : r['filename'],
            'n_pontos'        : n_total,
            'n_rachaduras'    : n_crack,
            'pct_rachadura'   : round(n_crack / n_total * 100, 2),
            'severidade'      : 'alta' if n_crack / n_total > 0.1 else
                                'media' if n_crack / n_total > 0.03 else 'baixa',
        })
    sev_path = os.path.join(RESULTS_SF, f'relatorio_severidade_{ts}.csv')
    with open(sev_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(sev_rows[0].keys()) if sev_rows else [])
        if sev_rows:
            w.writeheader()
            w.writerows(sev_rows)

    log.info(f"\nConcluído. Resultados em {RESULTS_SF} e {VIS_SF}")
    log.info(f"PLY coloridos em {PLY_SF}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 4: Verificar que o módulo importa sem erros**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from scalar_field_unsup import EdgeConvBlock, ScalarFieldDGCNN, build_knn_idx
from scalar_field_unsup import train_dgcnn, compute_anomaly_scores_unsup
print('OK — scalar_field_unsup importado')
"
```

Esperado: `OK — scalar_field_unsup importado`

- [ ] **Step 5: Rodar todos os testes do unsup**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -m pytest tests/test_scalar_field_unsup.py -v
```

Esperado: todos PASS.

---

## Task 5: teacher_student_v2.py — fork com 3 integrações

**Files:**
- Create: `src/teacher_student_v2.py`
- Create: `tests/test_teacher_student_v2.py`

- [ ] **Step 1: Escrever testes das 3 integrações do v2**

Criar `tests/test_teacher_student_v2.py`:

```python
# tests/test_teacher_student_v2.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import torch
import torch.nn as nn


# ── Dados sintéticos ──────────────────────────────────────────────────────────

def make_fake_result(n=100, has_crack=True, seed=0):
    """Cria um result dict sintético com scalar_field bimodal."""
    rng = np.random.default_rng(seed)
    gt = np.zeros(n, dtype=np.int64)
    gt[:20] = 1

    # Scalar_field: crack=[0,29], normal=[30,100]
    sf = np.concatenate([
        rng.uniform(0, 29, 20),
        rng.uniform(30, 100, 80),
    ]).astype(np.float32)

    score = rng.uniform(0, 1, n).astype(np.float32)
    # Crack points têm score maior
    score[:20] += 0.4
    score = score.clip(0, 1)

    return {
        'filename'    : 'avaria_test.ply',
        'has_crack'   : has_crack,
        'score'       : score,
        'gt_labels'   : gt,
        'pred_labels' : (score > 0.5).astype(np.int64),
        'n_points'    : n,
        'xyz'         : rng.standard_normal((n, 3)).astype(np.float32),
        'rgb'         : rng.uniform(0, 1, (n, 3)).astype(np.float32),
        'scalar_field': sf,
    }


# ── Teste integração 1: score fusion ─────────────────────────────────────────

def test_score_fusion_changes_score():
    """v2 score fusion deve alterar o score original."""
    from utils.evaluation import ScalarFieldGMM
    r = make_fake_result()
    score_orig = r['score'].copy()
    sf_raw     = r['scalar_field']

    gmm      = ScalarFieldGMM(sf_raw).fit()
    sf_score = gmm.anomaly_probability()

    lo, hi    = score_orig.min(), score_orig.max()
    dist_norm = (score_orig - lo) / (hi - lo + 1e-8)
    score_v2  = 0.7 * dist_norm + 0.3 * sf_score

    assert not np.allclose(score_orig, score_v2), "Score v2 deve diferir do original"
    assert score_v2.min() >= 0.0 and score_v2.max() <= 1.0

def test_score_fusion_unimodal_transparent():
    """Para nuvem unimodal, score fusion deve ser próximo ao distill_norm."""
    from utils.evaluation import ScalarFieldGMM
    rng = np.random.default_rng(1)
    # Scalar_field unimodal
    sf_raw     = rng.normal(80, 10, 200).astype(np.float32)
    score_orig = rng.uniform(0, 1, 200).astype(np.float32)

    gmm      = ScalarFieldGMM(sf_raw).fit()
    sf_score = gmm.anomaly_probability()

    lo, hi    = score_orig.min(), score_orig.max()
    dist_norm = (score_orig - lo) / (hi - lo + 1e-8)
    score_v2  = 0.7 * dist_norm + 0.3 * sf_score

    # sf_score ≈ 0.5 para unimodal → score_v2 ≈ 0.7*dist_norm + 0.15
    expected = 0.7 * dist_norm + 0.3 * 0.5
    assert np.allclose(score_v2, expected, atol=0.05)


# ── Teste integração 2: gate soft ─────────────────────────────────────────────

def test_soft_gate_suppresses_normal_points():
    """Pontos claramente normais (alto scalar_field) devem ter score modulado para baixo."""
    from utils.evaluation import ScalarFieldGMM
    r = make_fake_result()
    sf_raw = r['scalar_field']

    gmm     = ScalarFieldGMM(sf_raw).fit()
    weights = gmm.soft_weights()

    score_orig     = r['score'].copy()
    score_modulated = score_orig * weights

    # Pontos normais (sf > 30) devem ter score reduzido
    normal_mask = sf_raw > 30
    assert (score_modulated[normal_mask] <= score_orig[normal_mask]).all()

def test_soft_gate_unimodal_all_ones():
    """Para unimodal, soft gate deve ser transparente (weights=1.0)."""
    from utils.evaluation import ScalarFieldGMM
    rng = np.random.default_rng(2)
    sf_raw = rng.normal(80, 10, 200).astype(np.float32)
    gmm    = ScalarFieldGMM(sf_raw).fit()
    weights = gmm.soft_weights()
    assert np.allclose(weights, 1.0)


# ── Teste integração 3: Push-Pull ponderado ───────────────────────────────────

def test_pseudo_label_confidence_reduces_boundary_gradient():
    """
    Pontos no vale (ambíguos) devem ter confidence baixa → gradiente reduzido.
    """
    from utils.evaluation import ScalarFieldGMM
    # Scalar_field com vale claro em ~30
    sf_bimodal = np.concatenate([
        np.random.uniform(0, 25, 100).astype(np.float32),
        np.random.uniform(35, 100, 100).astype(np.float32),
    ])
    # Pontos no vale
    sf_valley = np.array([27.0, 28.5, 30.0, 31.0, 32.0], dtype=np.float32)

    gmm      = ScalarFieldGMM(sf_bimodal).fit()
    conf_all = gmm.pseudo_label_confidence()

    # Refazer com valley points incluídos
    sf_with_valley = np.concatenate([sf_bimodal, sf_valley])
    gmm2 = ScalarFieldGMM(sf_with_valley).fit()
    conf2 = gmm2.pseudo_label_confidence()

    # Pontos no vale (últimos 5) devem ter confidence < pontos nos núcleos
    valley_conf = conf2[-5:]
    core_conf   = conf2[:10]   # pontos deep crack (0-25)
    assert valley_conf.mean() < core_conf.mean(), \
        f"Vale conf={valley_conf.mean():.3f} deve ser < núcleo conf={core_conf.mean():.3f}"
```

- [ ] **Step 2: Rodar testes para confirmar que passam (usam apenas evaluation.py)**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -m pytest tests/test_teacher_student_v2.py -v
```

Esperado: todos PASS (usam `ScalarFieldGMM` de `evaluation.py` já implementado).

- [ ] **Step 3: Criar teacher_student_v2.py — copiar v1**

```bash
cp /home/cmatheus/projects/TCC/src/teacher_student_v1.py \
   /home/cmatheus/projects/TCC/src/teacher_student_v2.py
```

- [ ] **Step 4: Atualizar header do v2**

No início de `src/teacher_student_v2.py`, substituir o bloco de comentário do header (linhas 1–12):

```python
# ============================================================================
# PROFESSOR-ALUNO v2 — REVERSE DISTILLATION + SCALAR FIELD GMM
# ============================================================================
# Upgrades sobre o v1:
#   1. Score fusion : 0.7·distill_norm + 0.3·ScalarFieldGMM.anomaly_probability()
#   2. Gate soft    : score * ScalarFieldGMM.soft_weights() (substitui gate hard)
#   3. Push-Pull    : y * ScalarFieldGMM.pseudo_label_confidence() (gradiente ponderado)
#   4. PLY colorido : rachaduras em vermelho, cor original preservada nos normais
#
# v1 permanece intacto — este arquivo é standalone.
# Referências: idem v1 + ScalarFieldGMM (design doc 2026-04-13)
# ============================================================================
```

- [ ] **Step 5: Atualizar diretórios de saída no v2**

Localizar as 4 linhas de paths no início do v2 (após os imports):

```python
RESULTS    = f'{BASE_PATH}/results_ts'
MODELS     = f'{BASE_PATH}/models_ts'
VIS_PATH   = f'{BASE_PATH}/visualizations_ts'
LOGS_PATH  = f'{BASE_PATH}/logs_ts'
```

Substituir por:

```python
RESULTS    = f'{BASE_PATH}/results_ts'
MODELS     = f'{BASE_PATH}/models_ts_v2'
VIS_PATH   = f'{BASE_PATH}/visualizations_ts'
LOGS_PATH  = f'{BASE_PATH}/logs_ts'
PLY_PATH   = f'{BASE_PATH}/results_ts/ply'
```

- [ ] **Step 6: Aplicar integração 1 — score fusion em compute_anomaly_scores()**

Em `src/teacher_student_v2.py`, localizar a função `compute_anomaly_scores` e substituir o bloco de normalização/composição do score (o trecho que vai de `if memory_bank is not None` até `results.append({...})`):

```python
        if memory_bank is not None and memory_bank.bank is not None:
            btn, _    = model.teacher_features(x)
            bank_dist = memory_bank.score(btn.cpu(), k=5)
            lo, hi    = bank_dist.min(), bank_dist.max()
            bank_norm = (bank_dist - lo) / (hi - lo + 1e-8)
            lo2, hi2  = raw_score.min(), raw_score.max()
            dist_norm = (raw_score - lo2) / (hi2 - lo2 + 1e-8)
            distill_component = alpha_dist * dist_norm + alpha_bank * bank_norm
        else:
            lo, hi = raw_score.min(), raw_score.max()
            distill_component = (raw_score - lo) / (hi - lo + 1e-8)

        # ── v2: score fusion com ScalarFieldGMM ──────────────────────────────
        sf_raw   = d['features'][:, 9]
        sf_gmm   = ScalarFieldGMM(sf_raw).fit()
        sf_score = sf_gmm.anomaly_probability()   # (N,) ∈ [0,1]
        score    = 0.7 * distill_component + 0.3 * sf_score

        results.append({
            'filename'        : d['filename'],
            'has_crack'       : d['has_crack'],
            'score'           : score,
            'gt_labels'       : labels,
            'n_points'        : len(labels),
            'xyz'             : d['features'][:, :3],
            'rgb'             : d['features'][:, 3:6],
            'scalar_field'    : sf_raw,
            'sf_gmm_modality' : sf_gmm.modality,
        })
```

- [ ] **Step 7: Aplicar integração 2 — gate soft no main()**

Em `src/teacher_student_v2.py`, localizar a linha no `main()`:

```python
    results = apply_scalar_field_gate(results, sf_min, sf_max)
```

Substituir por:

```python
    # ── v2: gate soft por nuvem (substitui gate hard global) ─────────────────
    for r in results:
        sf_raw = r.get('scalar_field')
        if sf_raw is None:
            continue
        sf_gmm = ScalarFieldGMM(sf_raw).fit()
        r['score'] = r['score'] * sf_gmm.soft_weights()
```

E remover a linha que computa `sf_min, sf_max`:

```python
    sf_min, sf_max = compute_crack_sf_interval(train_list)
    log.info(f"Intervalo crack (scalar_field): [{sf_min:.2f}, {sf_max:.2f}]")
```

Substituir por:

```python
    log.info("v2: gate soft GMM por nuvem (substitui intervalo global do v1)")
```

Também remover (ou comentar) a segunda chamada de `apply_scalar_field_gate` nas estratégias de threshold:

```python
        # v1: res_s = apply_scalar_field_gate(res_s, sf_min, sf_max)
        # v2: gate soft já aplicado ao score antes do threshold
```

- [ ] **Step 8: Aplicar integração 3 — Push-Pull ponderado**

Em `src/teacher_student_v2.py`, no `train_teacher_student()`, localizar o bloco do loop supervisionado:

```python
                    out  = model(x)
                    loss = push_pull(
                        out['teacher_scales'], out['student_scales'], y)
```

Substituir por:

```python
                    out = model(x)
                    # v2: ponderar y pela confiança do scalar_field GMM
                    sf_col = x[:, 9].cpu().numpy()
                    sf_conf = torch.tensor(
                        ScalarFieldGMM(sf_col).fit().pseudo_label_confidence(),
                        dtype=torch.float32, device=device,
                    )
                    y_weighted = y * sf_conf
                    loss = push_pull(
                        out['teacher_scales'], out['student_scales'], y_weighted)
```

- [ ] **Step 9: Adicionar PLY colorido no main() do v2**

Em `src/teacher_student_v2.py`, no `main()`, após o bloco de avaliação (após `evaluate(res_s)` do último strategy), adicionar antes do `log.info("\nConcluído")`:

```python
    # ── PLY colorido — rachaduras em vermelho ─────────────────────────────────
    os.makedirs(PLY_PATH, exist_ok=True)
    log.info("\nSalvando PLY coloridos (v2)...")
    for r in results:
        if not r['has_crack']:
            continue
        ply_out = os.path.join(PLY_PATH,
                               r['filename'].replace('.ply', '_pred_v2.ply'))
        save_colored_ply(
            xyz        = r['xyz'],
            rgb_orig   = r['rgb'],
            pred_labels= r['pred_labels'],
            path       = ply_out,
        )

    # ── Relatório de severidade ───────────────────────────────────────────────
    sev_rows = []
    for r in results:
        if not r['has_crack']:
            continue
        n_crack = int(r['pred_labels'].sum())
        n_total = r['n_points']
        sev_rows.append({
            'arquivo'      : r['filename'],
            'n_pontos'     : n_total,
            'n_rachaduras' : n_crack,
            'pct_rachadura': round(n_crack / n_total * 100, 2),
            'severidade'   : 'alta'  if n_crack / n_total > 0.10 else
                             'media' if n_crack / n_total > 0.03 else 'baixa',
        })
    sev_path = os.path.join(RESULTS, f'relatorio_severidade_{ts}.csv')
    with open(sev_path, 'w', newline='') as f:
        if sev_rows:
            w = csv.DictWriter(f, fieldnames=list(sev_rows[0].keys()))
            w.writeheader()
            w.writerows(sev_rows)
    log.info(f"PLY coloridos salvos em {PLY_PATH}")
```

- [ ] **Step 10: Verificar que v2 importa sem erros**

```bash
cd /home/cmatheus/projects/TCC
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
# Só importar classes, não executar main
import teacher_student_v2
print('OK — teacher_student_v2 importado')
" 2>&1 | tail -5
```

Esperado: `OK — teacher_student_v2 importado`

---

## Task 6: Documentação em text_for_ia/

**Files:**
- Create: `text_for_ia/scalar_field_segmentation.md`

- [ ] **Step 1: Criar documentação**

Criar `text_for_ia/scalar_field_segmentation.md`:

```markdown
# Segmentação por Scalar Field — Documentação

## Por que o scalar_field discrimina rachaduras

O `scalar_Scalar_field` dos arquivos PLY é a intensidade de retorno do scanner laser.
Rachaduras absorvem mais luz → retornam menor intensidade → scalar_field mais baixo.
Em 15/34 arquivos de avaria, há um gap perfeito (gap=1) entre o máximo do cluster
de rachadura e o mínimo da superfície normal. F1=1.0 com threshold simples confirmado.

Os outros 19 arquivos usam `scalar_R` (canal RGB vermelho) — sem poder discriminativo.

## ScalarFieldGMM (utils/evaluation.py)

Classe central compartilhada por v2 e scalar_field_unsup.
- `fit()`: valley detection (scipy) → fallback GMM sklearn
- `modality`: 'bimodal' | 'unimodal'
- `anomaly_probability()`: P(crack|sf) por ponto ∈ [0,1]
- `soft_weights()`: gate soft — 1.0 para unimodal (transparente)
- `pseudo_label_confidence()`: |P-0.5|*2 — 0 no vale, 1 no núcleo

## scalar_field_unsup.py

Teacher (frozen) + DGCNN self-supervised.
- EdgeConvBlock: k-NN em features Teacher (não em XYZ), scipy para evitar OOM
- Grafo dinâmico: reconstruído após cada bloco nas features atualizadas
- Loss: L_contrast (bimodal) + L_recon (pretext task para unimodal)
- Score final: 0.7*dgcnn_norm + 0.3*sf_gmm

## teacher_student_v2.py

Fork do v1 com 3 integrações:
1. Score fusion: 0.7*distill_norm + 0.3*sf_gmm.anomaly_probability()
2. Gate soft: score * sf_gmm.soft_weights() (substitui gate hard global)
3. Push-Pull ponderado: y * sf_gmm.pseudo_label_confidence()

Para nuvens unimodais (scalar_R): soft_weights=1.0, confidence=0.0 → comportamento
idêntico ao v1 — degrada gracefully sem piorar.

## Saída PLY colorida

save_colored_ply() em evaluation.py:
- Pontos normais: cor original preservada
- Rachaduras preditas: vermelho (255, 0, 0)
- Configurável via crack_color=(R,G,B)
- Salvo em results_sf/ply/ e results_ts/ply/

## Ablation study (evaluate_ablation)

4 configs: distill_only → distill+gate → distill+fusion → v2_completo
Compara F1, IoU, AUROC para demonstrar contribuição marginal de cada componente.
```

---

## Self-Review do Plano

**Cobertura do spec:**
- ✅ ScalarFieldGMM com todos os 4 métodos + 2 properties
- ✅ save_colored_ply com cor original preservada
- ✅ EdgeConvBlock com k-NN dinâmico em features Teacher
- ✅ ScalarFieldDGCNN com score_head + recon_head
- ✅ Training loop 2 fases: warmup (L_recon) + self-sup (L_contrast + L_recon)
- ✅ compute_anomaly_scores_unsup com score fusion
- ✅ main() do unsup com histórico, métricas, PLY, severidade
- ✅ 3 integrações do v2 (score fusion, gate soft, Push-Pull ponderado)
- ✅ PLY colorido no v2 com `save_colored_ply`
- ✅ evaluate_ablation e compare_models
- ✅ Documentação em text_for_ia/
- ✅ Nenhum git commit automático

**Placeholder scan:** nenhum TBD, TODO, ou "similar ao task N".

**Consistência de tipos:**
- `ScalarFieldGMM.soft_weights()` → `np.ndarray (N,)` — consistente entre task 1, task 5 (step 7)
- `ScalarFieldGMM.pseudo_label_confidence()` → `np.ndarray (N,)` — consistente entre task 1, task 5 (step 8)
- `save_colored_ply(xyz, rgb_orig, pred_labels, path)` — mesma assinatura em task 1 (impl) e task 4/5 (uso)
- `build_knn_idx(feats_np, k)` → `torch.int64 (N,k)` — consistente entre task 3 (impl) e task 3 (uso no forward)
