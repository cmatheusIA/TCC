# Design: Adaptação do Teacher para PTv3 com Fallback Chain

**Data:** 2026-04-02  
**Status:** Aprovado  
**Contexto:** TCC — Detecção de Avarias em Nuvens de Pontos

---

## Problema

O Teacher atual (`PointTransformerInspiredAdvanced`) é uma arquitetura custom
PT-inspirada que carrega pesos S3DIS (paredes de gesso/vidro/cerâmica) via shape
matching, com transferência mínima. O PTv3 ScanNet200 (46M params, 529MB) foi
baixado mas é incompatível com a arquitetura atual: usa sparse voxel convolutions
(torchsparse/spconv) e estrutura de atenção diferente (QKV combinado).

Objetivo: adaptar o Teacher para usar PTv3 com transferência real de pesos,
mantendo fallback robusto para o caso de falha na instalação de dependências.

---

## Ambiente

- RTX 5060 Ti 16GB, sm_120 (Blackwell), CUDA 12.8
- WSL2 Ubuntu 24.04, PyTorch 2.7+cu128, uv
- torchsparse/spconv/MinkowskiEngine: **não instalados**
- PTv3 parâmetros: 46M | checkpoint: 529MB

---

## Interface do Teacher (contrato imutável)

Todos os Teachers devem implementar:

```python
def forward(x: Tensor[N, 15]) -> Tuple[Tensor[N, 512], List[Tensor]]:
    # Retorna: bottleneck (N, 512), scales=[t3(N,256), t2(N,128), t1(N,64)]

def freeze() -> None   # congela transformer blocks, mantém adapter treinável
def unfreeze() -> None
```

`StudentDecoder`, `TeacherStudentModel`, `training_utils.py`, `evaluation.py`,
`data.py` — **sem nenhuma mudança**.

---

## Abordagens (A → B → C em cascata)

### Abordagem A — PTv3 completo (torchsparse)

**Pré-requisito:** `torchsparse` compilado para sm_120.

**Arquitetura `PTv3Teacher`:**
```
x (N,15)
  → stem_adapter: Linear(15, 6)           ← projeta para dimensão do stem PTv3
  → dense_to_sparse()                     ← converte nuvem densa para SparseTensor
  → PTv3 backbone oficial (Pointcept)
        enc0 (32D) → enc1 (64D) → enc2 (128D) → enc3 (256D)
  → sparse_to_dense()
  → scale_proj:
        enc2 (128D) → t2 (128D), t1 via Linear(128,64)
        enc3 (256D) → t3 (256D)
  → proj_head: Linear(256, 512)           ← bottleneck
```

**Pesos carregados:** 100% dos blocos enc2 + enc3 (nome exato via `_selective_load`).
`stem_adapter` treinado do zero.

**Fallback:** se `import torchsparse` falhar ou qualquer `RuntimeError` na
inicialização, cai silenciosamente para B.

---

### Abordagem B — PTv3-compatible (PyTorch puro)

**Sem dependências novas.**

**`PTv3CompatibleBlock`** — replica enc2 sem sparse ops:

```python
class PTv3CompatibleBlock(nn.Module):
    # CPE (Conditional Positional Encoding):
    #   PTv3 original: sparse conv (128, 3,3,3, 128)
    #   Aqui: Linear(128, 128) sobre k-NN neighborhood (denso)
    #   → treinado do zero; sem pesos do checkpoint
    self.cpe_linear = nn.Linear(128, 128)
    self.cpe_norm   = nn.LayerNorm(128)

    # Attention (pesos enc2 carregados):
    self.attn_qkv  = nn.Linear(128, 384)   # (384,128) ← checkpoint
    self.attn_proj = nn.Linear(128, 128)   # (128,128) ← checkpoint
    self.norm1     = nn.LayerNorm(128)     # (128,)    ← checkpoint

    # FFN (pesos enc2 carregados):
    self.fc1 = nn.Linear(128, 512)         # (512,128) ← checkpoint
    self.fc2 = nn.Linear(512, 128)         # (128,512) ← checkpoint
    self.norm2 = nn.LayerNorm(128)         # (128,)    ← checkpoint
```

Cobertura de pesos do checkpoint: ~70% por bloco (CPE treinado do zero).

**`PTv3CompatibleTeacher`:**
```
x (N,15)
  → feature_adapter: Linear(15,64) → LayerNorm → ReLU → Linear(64,128)
  → pos_enc: SpatialPositionalEncoding3D(128)   ← reutilizado
  → 4× PTv3CompatibleBlock
        hook em block[0].output → Linear(128, 64) → t1 (N,64)
        hook em block[1].output → t2 (N,128)
        hook em block[2].output → Linear(128, 256) → t3 (N,256)
  → proj: Linear(128,256) → GELU → Linear(256,512) → bottleneck (N,512)
```

`_NEW_LAYERS = ['feature_adapter', 'cpe_linear', 'cpe_norm', 'proj', 'proj_t1', 'proj_t3']`

**Fallback:** se qualquer exceção na inicialização/carregamento, cai para C.

---

### Abordagem C — S3DIS (fallback)

`PointTransformerInspiredAdvanced` sem mudança, carrega `pointtransformer_s3dis_202109241350utc.pth`.

---

## Cadeia de fallback — `build_teacher()` em `teacher_student_v1.py`

```python
def build_teacher(input_dim: int, ptv3_ckpt: str, s3dis_ckpt: str) -> nn.Module:
    # A: torchsparse
    try:
        import torchsparse
        teacher = PTv3Teacher(input_dim=input_dim, checkpoint_path=ptv3_ckpt)
        log.info("Teacher: PTv3 completo (torchsparse) ✓")
        return teacher
    except (ImportError, Exception) as e:
        log.warning(f"PTv3Teacher indisponível ({type(e).__name__}: {e}) → B")

    # B: PTv3-compatible (denso)
    try:
        teacher = PTv3CompatibleTeacher(input_dim=input_dim, checkpoint_path=ptv3_ckpt)
        log.info("Teacher: PTv3-compatible (sem torchsparse) ✓")
        return teacher
    except Exception as e:
        log.warning(f"PTv3CompatibleTeacher falhou ({e}) → C (S3DIS)")

    # C: S3DIS fallback
    teacher = PointTransformerInspiredAdvanced(input_dim=input_dim, checkpoint_path=s3dis_ckpt)
    log.info("Teacher: PTv1 S3DIS (fallback) ✓")
    return teacher
```

`build_model()` chama `build_teacher()` em vez de instanciar o Teacher diretamente.

---

## Mudanças por arquivo

| Arquivo | Mudança |
|---|---|
| `src/utils/architectures.py` | + `PTv3CompatibleBlock`, `PTv3CompatibleTeacher`, `PTv3Teacher` |
| `src/teacher_student_v1.py` | + `build_teacher()`, `build_model()` usa `build_teacher()` |
| `src/utils/config.py` | + `PTRANSF_WEIGHTS_S3DIS` (fallback C nomeado explicitamente) |

**Sem mudança:** `StudentDecoder`, `TeacherStudentModel`, `training_utils.py`,
`evaluation.py`, `data.py`, `building_blocks.py`, `_selective_load`.

---

## Critério de sucesso

- [ ] Abordagem A: Teacher inicializa sem erro quando torchsparse disponível
- [ ] Abordagem B: Teacher inicializa, `_selective_load` carrega ≥60% dos pesos enc2
- [ ] Abordagem C: fallback ativo quando B falha
- [ ] `TeacherStudentModel.forward()` funciona com os 3 Teachers sem mudança
- [ ] Treino de 1 epoch sem erro de shape em qualquer abordagem ativa
