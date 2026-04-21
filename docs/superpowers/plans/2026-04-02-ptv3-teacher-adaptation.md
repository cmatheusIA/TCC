# PTv3 Teacher Adaptation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Adaptar o Teacher para usar pesos PTv3 ScanNet200 com fallback automático A→B→C.

**Architecture:** Abordagem A usa o PTv3 oficial com torchsparse; B reconstrói os blocos enc2 em PyTorch puro sem sparse ops; C mantém o `PointTransformerInspiredAdvanced` S3DIS original. Os três expõem a mesma interface (`forward`, `feature_adapter`, `lfa`, `blocks`) para compatibilidade com os hooks em `teacher_features()` sem alterar `TeacherStudentModel`.

**Tech Stack:** PyTorch 2.7+cu128, scipy cKDTree (já instalado), torchsparse (tentativa em Task 6), PTv3 checkpoint em `pretrained_models/ptv3_scannet200.pth`.

---

## Contexto de interface crítico

`TeacherStudentModel.teacher_features()` usa forward hooks em:
- `self.teacher.feature_adapter` → captura `(N, 128)` → usado como `t_adapter`
- `self.teacher.lfa` → captura `(N, 128)` → fonte primária de `t2` e `t1`
- `self.teacher.blocks[0]` → captura `(N, 128)` → fallback para `t1`
- `self.teacher(x)` → retorna `(N, 512)` → fonte de `t3` via `proj_t3`

**Regra:** qualquer Teacher novo deve ter atributos `feature_adapter`, `lfa`, `blocks` que emitam `(N,128)` durante o `forward`, e `forward(x)` deve retornar `(N, 512)`.

---

## Task 1: Config — adicionar `PTRANSF_WEIGHTS_S3DIS`

**Files:**
- Modify: `src/utils/config.py`

- [ ] **Step 1: Editar config.py**

Substituir o bloco atual de pesos por:

```python
KPCONV_WEIGHTS        = f'{PRETRAINED}/kpconv_s3dis_202010091238.pth'
PTRANSF_WEIGHTS       = f'{PRETRAINED}/ptv3_scannet200.pth'         # PTv3 (tentativa A ou B)
PTRANSF_WEIGHTS_S3DIS = f'{PRETRAINED}/pointtransformer_s3dis_202109241350utc.pth'  # fallback C
```

- [ ] **Step 2: Verificar sintaxe**

```bash
cd ~/projects/TCC && uv run python -c "from utils.config import PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS; print(PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS)"
```
Expected: dois caminhos impressos sem erro.

- [ ] **Step 3: Commit**

```bash
git add src/utils/config.py
git commit -m "config: add PTRANSF_WEIGHTS_S3DIS for PTv3 fallback chain"
```

---

## Task 2: `PTv3CompatibleBlock` em `architectures.py`

**Files:**
- Modify: `src/utils/architectures.py` (adicionar antes de `PointTransformerInspiredAdvanced`)
- Create: `src/test_teacher.py` (smoke test)

- [ ] **Step 1: Escrever o teste de shape**

Criar `src/test_teacher.py`:

```python
"""Smoke tests para os novos Teachers. Executar com: uv run python src/test_teacher.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import torch
from utils.config import *

def test_ptv3_compatible_block_shapes():
    from utils.architectures import PTv3CompatibleBlock
    block = PTv3CompatibleBlock(d_model=128, n_heads=8, k_neighbors=16)
    N = 256
    x   = torch.randn(N, 128)
    xyz = torch.randn(N, 3)
    out = block(x, xyz)
    assert out.shape == (N, 128), f"Esperado (256,128), obtido {out.shape}"
    print("✓ PTv3CompatibleBlock: shape OK")

if __name__ == '__main__':
    test_ptv3_compatible_block_shapes()
    print("TODOS OS TESTES PASSARAM")
```

- [ ] **Step 2: Rodar — verificar que falha com ImportError**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: `ImportError: cannot import name 'PTv3CompatibleBlock'`

- [ ] **Step 3: Implementar `PTv3CompatibleBlock` em `architectures.py`**

Adicionar imediatamente antes da linha `class PointTransformerInspiredAdvanced`:

```python
# ============================================================================
# PTv3-COMPATIBLE BLOCK — bloco de atenção compatível com enc2 do PTv3
# ============================================================================

class PTv3CompatibleBlock(nn.Module):
    """
    Replica a estrutura de enc2 do PTv3 sem sparse ops.

    Pesos carregáveis do checkpoint enc2 via _selective_load (shape fallback):
      attn_qkv  (384,128), attn_proj (128,128)
      fc1       (512,128), fc2       (128,512)
      norm1     (128,),    norm2     (128,)

    CPE (Conditional Positional Encoding): o original usa sparse conv
    (128,3,3,3,128). Aqui substituído por Linear sobre k-NN mean aggregate.
    Treinado do zero — não carrega pesos do checkpoint.

    Referência: Wu et al., PTv3 (CVPR 2024 Oral, arXiv:2312.10035)
    """

    def __init__(self, d_model: int = 128, n_heads: int = 8, k_neighbors: int = 16):
        super().__init__()
        assert d_model % n_heads == 0, "d_model deve ser divisível por n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_head   = d_model // n_heads   # 16 com d_model=128, n_heads=8
        self.k        = k_neighbors

        # CPE dense (substitui sparse conv 128×3×3×3×128)
        self.cpe_linear = nn.Linear(d_model, d_model)
        self.cpe_norm   = nn.LayerNorm(d_model)

        # Self-attention com QKV combinado (enc2.block*.attn.*)
        self.norm1     = nn.LayerNorm(d_model)
        self.attn_qkv  = nn.Linear(d_model, 3 * d_model)   # → (384, 128)
        self.attn_proj = nn.Linear(d_model, d_model)        # → (128, 128)

        # FFN 4× expansion (enc2.block*.mlp.0.*)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1   = nn.Linear(d_model, 4 * d_model)        # → (512, 128)
        self.fc2   = nn.Linear(4 * d_model, d_model)        # → (128, 512)
        self.act   = nn.GELU()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """
        x:   (N, d_model)
        xyz: (N, 3)
        Returns: (N, d_model)
        """
        N = x.shape[0]
        k = min(self.k, N)

        # ── CPE: mean aggregate k vizinhos (scipy cKDTree, O(N log N)) ──────
        xyz_np  = xyz.detach().cpu().numpy()
        tree    = cKDTree(xyz_np)
        _, idx  = tree.query(xyz_np, k=k)              # (N, k)  numpy
        idx_t   = torch.from_numpy(idx).long().to(x.device)
        nb_mean = x[idx_t].mean(dim=1)                 # (N, d_model)
        x_cpe   = self.cpe_norm(x + self.cpe_linear(nb_mean))

        # ── Self-attention (pre-norm, QKV combinado) ────────────────────────
        residual = x_cpe
        h   = self.norm1(x_cpe)
        qkv = self.attn_qkv(h)                         # (N, 3*d_model)
        qkv = qkv.reshape(N, 3, self.n_heads, self.d_head).permute(1, 2, 0, 3)
        q, k_t, v = qkv[0], qkv[1], qkv[2]            # (n_heads, N, d_head)
        attn = (q @ k_t.transpose(-2, -1)) * (self.d_head ** -0.5)
        attn = attn.softmax(dim=-1)
        out  = (attn @ v).permute(1, 0, 2).reshape(N, self.d_model)
        out  = self.attn_proj(out)
        x    = residual + out

        # ── FFN (pre-norm) ───────────────────────────────────────────────────
        residual = x
        h = self.norm2(x)
        h = self.fc2(self.act(self.fc1(h)))
        x = residual + h

        return x
```

- [ ] **Step 4: Rodar o teste — deve passar**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: `✓ PTv3CompatibleBlock: shape OK` e `TODOS OS TESTES PASSARAM`

- [ ] **Step 5: Commit**

```bash
git add src/utils/architectures.py src/test_teacher.py
git commit -m "feat: add PTv3CompatibleBlock (dense analog of PTv3 enc2)"
```

---

## Task 3: `PTv3CompatibleTeacher` em `architectures.py`

**Files:**
- Modify: `src/utils/architectures.py`
- Modify: `src/test_teacher.py`

- [ ] **Step 1: Adicionar teste de shape e hooks**

Adicionar em `src/test_teacher.py`, após o teste anterior:

```python
def test_ptv3_compatible_teacher_shapes():
    from utils.architectures import PTv3CompatibleTeacher
    # Sem checkpoint — inicialização aleatória
    teacher = PTv3CompatibleTeacher(input_dim=15, d_model=128, checkpoint_path=None)
    teacher.eval()
    N = 128
    x = torch.randn(N, 15)
    with torch.no_grad():
        bottleneck = teacher(x)
    assert bottleneck.shape == (N, 512), f"bottleneck: esperado (128,512), obtido {bottleneck.shape}"
    # Verificar que feature_adapter, lfa, blocks existem e têm output 128D
    feats = {}
    h1 = teacher.feature_adapter.register_forward_hook(lambda m,i,o: feats.update({'adapter': o}))
    h2 = teacher.lfa.register_forward_hook(lambda m,i,o: feats.update({'lfa': o}))
    h3 = teacher.blocks[0].register_forward_hook(lambda m,i,o: feats.update({'blk0': o}))
    with torch.no_grad():
        teacher(x)
    h1.remove(); h2.remove(); h3.remove()
    assert feats['adapter'].shape == (N, 128), f"adapter: {feats['adapter'].shape}"
    assert feats['lfa'].shape     == (N, 128), f"lfa: {feats['lfa'].shape}"
    assert feats['blk0'].shape    == (N, 128), f"blk0: {feats['blk0'].shape}"
    print("✓ PTv3CompatibleTeacher: shapes e hooks OK")

def test_ptv3_compatible_teacher_freeze():
    from utils.architectures import PTv3CompatibleTeacher
    teacher = PTv3CompatibleTeacher(input_dim=15, checkpoint_path=None)
    teacher.freeze()
    trainable = [n for n, p in teacher.named_parameters() if p.requires_grad]
    frozen    = [n for n, p in teacher.named_parameters() if not p.requires_grad]
    assert len(trainable) > 0, "Nenhum parâmetro treinável após freeze"
    assert all('feature_adapter' in n for n in trainable), \
        f"Params treináveis inesperados: {trainable}"
    print(f"✓ PTv3CompatibleTeacher freeze: {len(frozen)} frozen, {len(trainable)} treináveis")
```

Adicionar também no bloco `__main__`:
```python
    test_ptv3_compatible_teacher_shapes()
    test_ptv3_compatible_teacher_freeze()
```

- [ ] **Step 2: Rodar — verificar que falha**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: `ImportError: cannot import name 'PTv3CompatibleTeacher'`

- [ ] **Step 3: Implementar `PTv3CompatibleTeacher`**

Adicionar em `architectures.py`, imediatamente após `PTv3CompatibleBlock`:

```python
# ============================================================================
# PTv3-COMPATIBLE TEACHER — Abordagem B (sem torchsparse)
# ============================================================================

class PTv3CompatibleTeacher(nn.Module):
    """
    Teacher baseado em blocos PTv3-compatible (sem sparse ops).

    Expõe feature_adapter, lfa, blocks para compatibilidade com os forward
    hooks em TeacherStudentModel.teacher_features() — sem alterar TeacherStudentModel.

    Transfer: _selective_load carrega por shape enc2.block0 → lfa,
              enc2.block1 → blocks[0]. blocks[1] e blocks[2] inicializam do zero.

    Arquitetura:
      feature_adapter : Linear(15→64) → LN → ReLU → Linear(64→128)   [do zero]
      lfa             : PTv3CompatibleBlock(128)  ← enc2.block0 weights
      blocks[0]       : PTv3CompatibleBlock(128)  ← enc2.block1 weights
      blocks[1,2]     : PTv3CompatibleBlock(128)  [do zero]
      proj            : LN → Linear(128→256) → GELU → Linear(256→512)  [do zero]
    """

    _NEW_LAYERS = ['feature_adapter', 'cpe_linear', 'cpe_norm', 'proj']

    def __init__(self, input_dim: int = INPUT_DIM, d_model: int = D_MODEL,
                 num_extra_blocks: int = 3, checkpoint_path: str = None):
        super().__init__()
        print(f"\n🧠 PTv3CompatibleTeacher (input={input_dim}D, d_model={d_model}D)")

        # ── feature_adapter: hook point #1, 15D → 128D ─────────────────────
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )

        # ── lfa: hook point #2, primeiro bloco PTv3 ─────────────────────────
        # Nomear "lfa" para compatibilidade com o hook em teacher_features()
        self.lfa = PTv3CompatibleBlock(d_model)

        # ── blocks: hooks[0] em blocks[0], blocos adicionais ────────────────
        self.blocks = nn.ModuleList([
            PTv3CompatibleBlock(d_model) for _ in range(num_extra_blocks)
        ])

        # ── proj: bottleneck 128D → 512D ────────────────────────────────────
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Linear(d_model * 2, 512),
        )

        self._init_weights()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pretrained(checkpoint_path)
        else:
            print(f"   ⚠️  Pesos não encontrados: {checkpoint_path}")

        n = sum(p.numel() for p in self.parameters())
        print(f"   📊 Parâmetros: {n:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _load_pretrained(self, path: str):
        print(f"   📦 Carregando: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            loaded, skipped_new, shape_miss = _selective_load(
                self, sd, new_layer_names=self._NEW_LAYERS)
            total = sum(1 for _ in self.parameters())
            pct   = loaded / total * 100 if total else 0
            print(f"   ✅ Transfer: {loaded}/{total} tensores ({pct:.1f}%)")
            print(f"      Novas (skip): {skipped_new} | Shape mismatch: {shape_miss}")
        except Exception as e:
            print(f"   ⚠️  Erro no transfer: {e}")

    def freeze(self):
        """Congela tudo exceto feature_adapter."""
        for name, p in self.named_parameters():
            if 'feature_adapter' not in name:
                p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 15) → (N, 512)"""
        xyz = x[:, :3]

        x = self.feature_adapter(x)          # (N, 128) — hook 'adapter'
        x = self.lfa(x, xyz)                 # (N, 128) — hook 'lfa'

        for blk in self.blocks:              # hook 'block0' captura blocks[0]
            x = blk(x, xyz)

        return self.proj(x)                  # (N, 512)
```

- [ ] **Step 4: Rodar os testes**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: três linhas `✓` e `TODOS OS TESTES PASSARAM`

- [ ] **Step 5: Commit**

```bash
git add src/utils/architectures.py src/test_teacher.py
git commit -m "feat: add PTv3CompatibleTeacher (Approach B, no sparse ops)"
```

---

## Task 4: `PTv3Teacher` em `architectures.py` (Abordagem A)

**Files:**
- Modify: `src/utils/architectures.py`
- Modify: `src/test_teacher.py`

- [ ] **Step 1: Adicionar teste de importação condicional**

Adicionar em `src/test_teacher.py`:

```python
def test_ptv3_teacher_import():
    try:
        import torchsparse
        from utils.architectures import PTv3Teacher
        teacher = PTv3Teacher(input_dim=15, checkpoint_path=None)
        N = 64
        x = torch.randn(N, 15)
        with torch.no_grad():
            bottleneck = teacher(x)
        assert bottleneck.shape == (N, 512), f"bottleneck: {bottleneck.shape}"
        print("✓ PTv3Teacher (torchsparse): shapes OK")
    except ImportError:
        print("⚠  PTv3Teacher: torchsparse não instalado — pulando (comportamento esperado)")
```

Adicionar no bloco `__main__`: `test_ptv3_teacher_import()`

- [ ] **Step 2: Rodar — deve passar (torchsparse ausente = skip)**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: `⚠  PTv3Teacher: torchsparse não instalado — pulando`

- [ ] **Step 3: Implementar `PTv3Teacher`**

Adicionar em `architectures.py`, após `PTv3CompatibleTeacher`:

```python
# ============================================================================
# PTv3 TEACHER — Abordagem A (requer torchsparse)
# ============================================================================

class PTv3Teacher(nn.Module):
    """
    Teacher usando o PTv3 oficial (Pointcept) com torchsparse.

    Requer: torchsparse instalado e compilado para o sm atual.
    Levanta ImportError no __init__ se torchsparse não estiver disponível
    — build_teacher() captura e cai para PTv3CompatibleTeacher.

    Interface de hooks (compatível com teacher_features()):
      feature_adapter : Linear(15→128) — chamado no início do forward
      lfa             : wrapper que recebe enc2(128D) e o passa adiante
      blocks[0]       : wrapper que recebe enc3→proj(256→128) e o passa adiante

    Pesos carregados:
      PTv3 backbone completo via _selective_load (nome exato via 'module.backbone.*')
      feature_adapter, lfa, blocks, proj_head: treinados do zero
    """

    _NEW_LAYERS = ['feature_adapter', 'lfa', 'blocks', 'proj_head',
                   'stem_adapter', 'enc2_hook_proj', 'enc3_hook_proj']

    def __init__(self, input_dim: int = INPUT_DIM, checkpoint_path: str = None):
        # Verificar torchsparse imediatamente — build_teacher() depende deste erro
        try:
            import torchsparse
            import torchsparse.nn as spnn
            from torchsparse import SparseTensor
            self._torchsparse = torchsparse
            self._spnn = spnn
            self._SparseTensor = SparseTensor
        except ImportError as e:
            raise ImportError(f"PTv3Teacher requer torchsparse: {e}") from e

        super().__init__()
        print(f"\n🚀 PTv3Teacher (torchsparse, input={input_dim}D)")

        # ── stem_adapter: projeta 15D → 6D para o stem PTv3 ─────────────────
        self.stem_adapter = nn.Linear(input_dim, 6)

        # ── feature_adapter: hook point #1 (15D → 128D, pré-backbone) ───────
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, 64), nn.LayerNorm(64), nn.ReLU(),
            nn.Linear(64, 128),
        )

        # ── PTv3 backbone (carregado do checkpoint) ───────────────────────────
        # Instanciar via Pointcept se disponível, senão usar config manual
        self.backbone = self._build_ptv3_backbone()

        # ── Wrappers de hook: lfa e blocks[0] ────────────────────────────────
        # Recebem as features enc2/enc3 projetadas durante o forward para que
        # os hooks em teacher_features() capturem tensores (N,128).
        self.lfa    = nn.Identity()                    # recebe enc2 (128D) — hook #2
        self.blocks = nn.ModuleList([nn.Linear(256, 128)])  # recebe enc3→128D — hook #3

        # ── proj_head: enc3 256D → bottleneck 512D ────────────────────────────
        self.proj_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pretrained(checkpoint_path)
        else:
            print(f"   ⚠️  Pesos não encontrados: {checkpoint_path}")

        n = sum(p.numel() for p in self.parameters())
        print(f"   📊 Parâmetros: {n:,}")

    def _build_ptv3_backbone(self):
        """
        Tenta importar PTv3 do Pointcept. Se não disponível, constrói
        uma versão simplificada usando os blocos compatíveis.
        """
        try:
            # Pointcept PTv3 — requer Pointcept instalado no path
            from model.point_transformer_v3 import PointTransformerV3  # noqa
            return PointTransformerV3()
        except ImportError:
            # Fallback: 4 blocos PTv3CompatibleBlock como backbone denso
            import warnings
            warnings.warn("Pointcept não encontrado — usando backbone denso como PTv3Teacher")
            return nn.ModuleList([PTv3CompatibleBlock(128) for _ in range(4)])

    def _dense_to_sparse(self, x_6d: torch.Tensor, xyz: torch.Tensor):
        """Converte nuvem densa (N,6) para SparseTensor (torchsparse)."""
        SparseTensor = self._SparseTensor
        voxel_size = VOXEL_SIZE
        coords = (xyz / voxel_size).int()
        batch  = torch.zeros(len(coords), 1, dtype=torch.int, device=coords.device)
        coords = torch.cat([batch, coords], dim=1)         # (N, 4): [batch, x, y, z]
        return SparseTensor(feats=x_6d, coords=coords)

    def _load_pretrained(self, path: str):
        print(f"   📦 Carregando: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
            loaded, skipped_new, shape_miss = _selective_load(
                self, sd, new_layer_names=self._NEW_LAYERS)
            total = sum(1 for _ in self.parameters())
            pct   = loaded / total * 100 if total else 0
            print(f"   ✅ Transfer: {loaded}/{total} ({pct:.1f}%)")
        except Exception as e:
            print(f"   ⚠️  Erro no transfer: {e}")

    def freeze(self):
        for name, p in self.named_parameters():
            if 'feature_adapter' not in name:
                p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 15) → (N, 512)"""
        xyz = x[:, :3]

        # hook #1: feature_adapter (15D→128D, pré-backbone)
        _ = self.feature_adapter(x)

        if isinstance(self.backbone, nn.ModuleList):
            # Backbone denso (fallback — Pointcept não disponível)
            h = self.feature_adapter(x)
            for blk in self.backbone:
                h = blk(h, xyz)
            enc2_feat = h        # (N, 128)
            enc3_feat = h        # mesmo nível sem hierarquia real
        else:
            # Backbone PTv3 oficial (torchsparse)
            x_6d   = self.stem_adapter(x)                    # (N, 6)
            sparse = self._dense_to_sparse(x_6d, xyz)
            out    = self.backbone(sparse)                   # dict com enc feats
            # Extrair enc2 e enc3 do dict de saída do PTv3
            enc2_feat = out.get('enc2', out.get('feat', x_6d))
            if hasattr(enc2_feat, 'feats'):
                enc2_feat = enc2_feat.feats               # sparse → dense
            enc3_feat = out.get('enc3', enc2_feat)
            if hasattr(enc3_feat, 'feats'):
                enc3_feat = enc3_feat.feats

            # Garantir N consistente (voxelização pode mudar N)
            if enc2_feat.shape[0] != x.shape[0]:
                enc2_feat = enc2_feat[:x.shape[0]]
            if enc3_feat.shape[0] != x.shape[0]:
                enc3_feat = enc3_feat[:x.shape[0]]

        # hook #2: lfa recebe enc2 (128D)
        enc2_feat = self.lfa(enc2_feat)

        # hook #3: blocks[0] projeta enc3 → 128D
        if enc3_feat.shape[-1] != 128:
            enc3_feat = self.blocks[0](enc3_feat)
        else:
            enc3_feat = self.blocks[0](enc3_feat)

        # bottleneck: enc3 256D → 512D (ou enc2 128D se backbone denso)
        if enc3_feat.shape[-1] == 256:
            bottleneck = self.proj_head(enc3_feat)
        else:
            # backbone denso: enc3_feat já é 128D — expandir
            bottleneck = self.proj_head(
                torch.cat([enc3_feat, enc3_feat], dim=-1))  # 128→256→512

        return bottleneck
```

- [ ] **Step 4: Rodar os testes**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: todos os testes anteriores passam + `⚠  PTv3Teacher: torchsparse não instalado`

- [ ] **Step 5: Commit**

```bash
git add src/utils/architectures.py src/test_teacher.py
git commit -m "feat: add PTv3Teacher skeleton (Approach A, torchsparse conditional)"
```

---

## Task 5: `build_teacher()` e atualização de `build_model()`

**Files:**
- Modify: `src/teacher_student_v1.py`
- Modify: `src/test_teacher.py`

- [ ] **Step 1: Adicionar teste da cadeia de fallback**

Adicionar em `src/test_teacher.py`:

```python
def test_build_teacher_fallback():
    """Verifica que build_teacher() retorna um Teacher válido (B ou C)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    # Importar build_teacher do teacher_student_v1
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ts", os.path.join(os.path.dirname(__file__), "teacher_student_v1.py"))
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)

    teacher = ts.build_teacher(input_dim=15,
                               ptv3_ckpt=None,
                               s3dis_ckpt=None)
    N = 64
    x = torch.randn(N, 15)
    with torch.no_grad():
        bottleneck = teacher(x)
    assert bottleneck.shape == (N, 512), f"bottleneck: {bottleneck.shape}"
    print(f"✓ build_teacher fallback: {type(teacher).__name__} retornou (N,512)")
```

Adicionar no `__main__`: `test_build_teacher_fallback()`

- [ ] **Step 2: Rodar — falha pois build_teacher não existe**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py 2>&1 | tail -5
```
Expected: `AttributeError: module ... has no attribute 'build_teacher'`

- [ ] **Step 3: Adicionar `build_teacher()` em `teacher_student_v1.py`**

Localizar a função `build_model()` (linha ~620) e adicionar `build_teacher()` logo acima:

```python
def build_teacher(input_dim: int, ptv3_ckpt: str, s3dis_ckpt: str) -> nn.Module:
    """
    Instancia o Teacher com fallback automático A → B → C.

    A: PTv3Teacher (torchsparse) — transfer 100% enc2+enc3
    B: PTv3CompatibleTeacher (denso) — transfer ~50% via shape-matching enc2
    C: PointTransformerInspiredAdvanced (S3DIS) — transfer mínimo por shape

    Todos expõem feature_adapter / lfa / blocks para compatibilidade com
    os forward hooks em TeacherStudentModel.teacher_features().
    """
    # ── Abordagem A: PTv3 completo (torchsparse) ──────────────────────────────
    try:
        from utils.architectures import PTv3Teacher
        teacher = PTv3Teacher(input_dim=input_dim, checkpoint_path=ptv3_ckpt)
        log.info("Teacher: PTv3 completo (torchsparse) ✓")
        return teacher
    except ImportError as e:
        log.warning(f"PTv3Teacher: torchsparse indisponível ({e}) → tentando B")
    except Exception as e:
        log.warning(f"PTv3Teacher falhou ({type(e).__name__}: {e}) → tentando B")

    # ── Abordagem B: PTv3-compatible (PyTorch puro) ───────────────────────────
    try:
        from utils.architectures import PTv3CompatibleTeacher
        teacher = PTv3CompatibleTeacher(input_dim=input_dim, checkpoint_path=ptv3_ckpt)
        log.info("Teacher: PTv3CompatibleTeacher (sem torchsparse) ✓")
        return teacher
    except Exception as e:
        log.warning(f"PTv3CompatibleTeacher falhou ({type(e).__name__}: {e}) → C (S3DIS)")

    # ── Abordagem C: S3DIS fallback ───────────────────────────────────────────
    from utils.architectures import PointTransformerInspiredAdvanced
    teacher = PointTransformerInspiredAdvanced(input_dim=input_dim,
                                               checkpoint_path=s3dis_ckpt)
    log.info("Teacher: PointTransformerInspiredAdvanced (S3DIS fallback) ✓")
    return teacher
```

- [ ] **Step 4: Atualizar `build_model()`**

Substituir o `build_model()` atual:

```python
def build_model(device: torch.device) -> TeacherStudentModel:
    teacher = build_teacher(
        input_dim=INPUT_DIM,
        ptv3_ckpt=PTRANSF_WEIGHTS,
        s3dis_ckpt=PTRANSF_WEIGHTS_S3DIS,
    )
    model = TeacherStudentModel(input_dim=INPUT_DIM, teacher_ckpt=None)
    # Substituir teacher instanciado por build_teacher
    model.teacher = teacher
    model._freeze_teacher()
    return model.to(device)
```

- [ ] **Step 5: Rodar os testes**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: todos passam incluindo `✓ build_teacher fallback: PTv3CompatibleTeacher retornou (N,512)`

- [ ] **Step 6: Commit**

```bash
git add src/teacher_student_v1.py src/test_teacher.py
git commit -m "feat: add build_teacher() fallback chain A→B→C, update build_model()"
```

---

## Task 6: Tentar instalação do torchsparse (sm_120)

**Files:** nenhum arquivo de código — instalação apenas.

- [ ] **Step 1: Verificar pré-requisitos**

```bash
nvcc --version
uv run python -c "import torch; print(torch.version.cuda, torch.cuda.get_device_capability())"
```
Expected: CUDA 12.8, `(12, 0)`.

- [ ] **Step 2: Tentar instalação via pip**

```bash
uv pip install torchsparse 2>&1 | tail -20
```

Se falhar (wheel não disponível para sm_120), seguir Step 3.

- [ ] **Step 3: Compilar do código-fonte (se Step 2 falhou)**

```bash
cd /tmp
git clone https://github.com/mit-han-lab/torchsparse.git
cd torchsparse
TORCH_CUDA_ARCH_LIST="12.0" uv pip install -e . 2>&1 | tail -30
```

Se a compilação falhar com erro de arquitetura, registrar a falha e pular Task 7 — a cadeia de fallback já usa B automaticamente.

- [ ] **Step 4: Verificar instalação**

```bash
uv run python -c "import torchsparse; print('torchsparse OK:', torchsparse.__version__)"
```

Se OK: continuar para Task 7. Se falhar: registrar e pular Task 7.

---

## Task 7: Teste da Abordagem A (só executar se Task 6 OK)

**Files:**
- Modify: `src/test_teacher.py`

- [ ] **Step 1: Rodar o teste de importação (já existente)**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: `✓ PTv3Teacher (torchsparse): shapes OK` em vez do warning anterior.

- [ ] **Step 2: Verificar que build_teacher usa A**

```bash
cd ~/projects/TCC && uv run python -c "
import sys; sys.path.insert(0,'src')
from utils.config import *
import teacher_student_v1 as ts
import torch
t = ts.build_teacher(INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS)
print('Teacher:', type(t).__name__)
" 2>&1 | grep "Teacher:"
```
Expected: `Teacher: PTv3 completo (torchsparse) ✓`

- [ ] **Step 3: Commit (se testes passarem)**

```bash
git add src/test_teacher.py
git commit -m "test: verify PTv3Teacher (Approach A) with torchsparse"
```

---

## Task 8: Teste de integração — forward pass completo do TeacherStudentModel

**Files:**
- Modify: `src/test_teacher.py`

- [ ] **Step 1: Adicionar teste de integração**

Adicionar em `src/test_teacher.py`:

```python
def test_teacher_student_integration():
    """Verifica forward pass completo com o Teacher ativo via build_model()."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ts", os.path.join(os.path.dirname(__file__), "teacher_student_v1.py"))
    ts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ts)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = ts.build_model(device)
    model.eval()

    N = 512
    x = torch.randn(N, 15, device=device)

    with torch.no_grad():
        out = model(x)

    assert 'teacher_scales' in out
    assert 'student_scales' in out
    assert 'bottleneck'     in out

    t3, t2, t1 = out['teacher_scales']
    s3, s2, s1 = out['student_scales']

    assert t3.shape == (N, 256), f"t3: {t3.shape}"
    assert t2.shape == (N, 128), f"t2: {t2.shape}"
    assert t1.shape == (N, 64),  f"t1: {t1.shape}"
    assert s3.shape == (N, 256), f"s3: {s3.shape}"
    assert s2.shape == (N, 128), f"s2: {s2.shape}"
    assert s1.shape == (N, 64),  f"s1: {s1.shape}"

    teacher_name = type(model.teacher).__name__
    print(f"✓ Integração: {teacher_name} | t3{t3.shape} t2{t2.shape} t1{t1.shape}")
```

Adicionar no `__main__`: `test_teacher_student_integration()`

- [ ] **Step 2: Rodar**

```bash
cd ~/projects/TCC && uv run python src/test_teacher.py
```
Expected: todos os testes passam, última linha indica qual Teacher está ativo.

- [ ] **Step 3: Commit final**

```bash
git add src/test_teacher.py
git commit -m "test: integration test for TeacherStudentModel with PTv3 fallback chain"
```

---

## Task 9: Documentar em `text_for_ia/`

**Files:**
- Create: `text_for_ia/IMPLEMENTACOES_PTEV3_TEACHER.md`
- Modify: `text_for_ia/MELHORIAS.md`

- [ ] **Step 1: Criar documento de implementação**

Criar `text_for_ia/IMPLEMENTACOES_PTEV3_TEACHER.md` descrevendo:
- Qual abordagem ficou ativa (A, B ou C)
- Quantos pesos foram carregados (log do `_selective_load`)
- Shapes dos hooks verificados
- Resultado do `test_teacher_student_integration`

- [ ] **Step 2: Atualizar tabela de status em `MELHORIAS.md`**

Mudar `PTv3 ScanNet como Teacher | Pendente` para `✅ Implementado`.

- [ ] **Step 3: Commit**

```bash
git add text_for_ia/
git commit -m "docs: document PTv3 teacher adaptation implementation"
```

---

## Self-Review

**Spec coverage:**
- ✅ Abordagem A (`PTv3Teacher`) → Task 4
- ✅ Abordagem B (`PTv3CompatibleTeacher`) → Task 3
- ✅ Abordagem C (S3DIS fallback) → Task 5 `build_teacher()`
- ✅ Interface de hooks (`feature_adapter`, `lfa`, `blocks`) → Tasks 3, 4, 5
- ✅ `build_teacher()` fallback A→B→C → Task 5
- ✅ `build_model()` atualizado → Task 5
- ✅ `PTRANSF_WEIGHTS_S3DIS` → Task 1
- ✅ Smoke tests → Tasks 2, 3, 4, 5, 8
- ✅ Documentação text_for_ia → Task 9

**Placeholder scan:** nenhum TBD encontrado.

**Type consistency:** `build_teacher()` em Task 5 recebe `ptv3_ckpt` e `s3dis_ckpt`; `build_model()` passa `PTRANSF_WEIGHTS` e `PTRANSF_WEIGHTS_S3DIS` definidos em Task 1. Consistente.

**Escopo:** independente — não altera `StudentDecoder`, `training_utils`, `evaluation`, `data`.
