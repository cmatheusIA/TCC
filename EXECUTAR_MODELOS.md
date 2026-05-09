# Como Executar os Modelos — Pipeline Binário de Rachaduras

Dataset: 35 nuvens avaria + 29 normais em `data/train/`
Protocolo: LOCO (Leave-One-Cloud-Out) por nuvem avaria

---

## 1. GMMScalar — Não-supervisionado (~2 min)

```bash
cd /home/cmatheus/projects/TCC/src
uv run python gmm_scalar.py
```

Resultado em: `results_gmm_scalar/loco_<timestamp>.csv`

---

## 2. ScalarGAT — Supervisionado (~30–60 min)

```bash
cd /home/cmatheus/projects/TCC/src
uv run python scalar_gat.py 
```

Resultado em: `results_scalar_gat/loco_<timestamp>.csv`

> Roda 35 folds em paralelo com 16 workers. Acompanhe o progresso no terminal.

---

## 3. ScalarMAE — Semi-supervisionado (~2–4 h)

```bash
cd /home/cmatheus/projects/TCC/src
uv run python scalar_mae.py
```

Resultado em: `results_scalar_mae/loco_<timestamp>.csv`

> Pode rodar em background. Monitorar os 2 primeiros folds para confirmar que FixMatch não diverge.

---

## 4. PTv3 Binary — Supervisionado (~3–6 h)

**Antes de rodar o LOCO completo, validar o backbone:**

```bash
cd /home/cmatheus/projects/TCC/src
uv run python -c "
from ptv3_binary import build_binary_model
teacher, head = build_binary_model()
print('Teacher OK:', type(teacher).__name__)
import torch
x = torch.randn(100, 16)
out = teacher(x)
print('Teacher output:', out.shape)
"
```

Esperado: `Teacher output: torch.Size([100, 128])` (ou dimensão diferente de 128 — ajustar `BinarySegHead(d_model=...)` se necessário).

**Se OK, rodar LOCO:**

```bash
cd /home/cmatheus/projects/TCC/src
uv run python ptv3_binary.py
```

Resultado em: `results_ptv3_binary/loco_<timestamp>.csv`

---

## 5. DGCNN-BiGAN — Não-supervisionado (~8–20 h)

**Rodar overnight com nohup:**

```bash
cd /home/cmatheus/projects/TCC/src
nohup uv run python crack_gan.py > ../logs_crack_gan/stdout.log 2>&1 &
echo "PID: $!"
```

Acompanhar progresso:

```bash
tail -f /home/cmatheus/projects/TCC/logs_crack_gan/stdout.log
```

Resultado em: `results_crack_gan/loco_<timestamp>.csv`

> Se a loss do discriminador divergir (valores > 1000), interromper e reduzir `LR_D` e `LR_G` de `1e-4` para `5e-5` em `src/crack_gan.py`.

---

## 6. Benchmark — Comparação Final (< 5 s)

Pode ser rodado a qualquer momento. Modelos sem resultado aparecem como `(sem resultados)`.

```bash
cd /home/cmatheus/projects/TCC/src
uv run python benchmark_crack.py
```

Resultado em: `results_benchmark/summary_binary.csv`

Exemplo de saída esperada:
```
========================================================================
  BENCHMARK BINÁRIO — Detecção de Rachaduras (LOCO, n=35)
  Baseline XGBoost: AUROC=0.930  F1=0.595
========================================================================
  Modelo             AUROC     ±       F1     ±       AP  Folds
  -----------------------------------------------------------------
  xgboost (baseline)  0.9300     —   0.5950     —        —      —
  gmm_scalar          0.????  0.???  0.????  0.???  0.????     35
  scalar_gat          0.????  0.???  0.????  0.???  0.????     35
  ...
  ★ = supera baseline XGBoost
```

---

## Checklist Mínimo para o TCC

- [ ] GMMScalar — baseline não-supervisionado
- [ ] ScalarGAT — supervisionado leve
- [ ] ScalarMAE — semi-supervisionado
- [ ] PTv3 Binary — backbone pré-treinado
- [ ] DGCNN-BiGAN — opcional, mais custoso
