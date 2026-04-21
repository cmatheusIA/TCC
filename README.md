# Detecção de Rachaduras em Nuvens de Pontos 3D

Trabalho de Conclusão de Curso — Detecção automática de rachaduras em superfícies de alvenaria a partir de nuvens de pontos 3D capturadas por fotogrametria.

---

## O que o projeto faz

Dada uma nuvem de pontos 3D de uma parede ou estrutura de alvenaria, o sistema identifica automaticamente **quais pontos pertencem a uma rachadura** — sem precisar de exemplos de rachaduras durante o treinamento.

O sistema aprende o que é uma superfície **normal** (sem rachadura) e considera qualquer desvio significativo como anomalia.

---

## Por que isso é difícil

- **Rachaduras são raras e caras de anotar por ponto.** Um único scan pode ter centenas de milhares de pontos; anotar manualmente quais pertencem à rachadura é inviável.
- **Nuvens de pontos não têm estrutura regular** como imagens — não é possível aplicar CNNs diretamente.
- **A variabilidade de captura é alta** — distância do scanner, iluminação e ângulo afetam diretamente a geometria medida.

---

## Abordagem

Dois modelos são implementados e comparados:

| Modelo | Princípio | Arquivo |
|--------|-----------|---------|
| **Professor-Aluno** | Um professor pré-treinado extrai features de superfície; um aluno tenta imitá-lo. Em regiões normais o aluno consegue imitar bem — em rachaduras, falha. | `src/teacher_student_v1.py` |
| **GAN** | Um gerador aprende a reconstruir superfícies normais. Superfícies com rachadura são reconstruídas com erro alto. | `src/gan_rachaduras_v5.py` |

O Professor usa pesos pré-treinados do **Point Transformer V3** (Wu et al., CVPR 2024) em dados do ScanNet200 — dataset com 200 categorias de superfícies internas de concreto.

---

## Resultado esperado

- **Mapa de anomalia por ponto** — cada ponto recebe um score de 0 a 1
- **Classificação binária** — normal ou rachadura, com threshold calibrado automaticamente via GMM
- **Severidade ABNT NBR 6118:2014** — nível A/B/C/D com recomendação de intervenção
- **Visualizações** — arquivo PLY colorido (verde=TP, vermelho=FP, azul=FN) para uso no CloudCompare

---

## Estrutura de dados esperada

```
data/
  train/
    n_avaria_1.ply    ← superfície normal (sem rachadura)
    n_avaria_2.ply
    ...
  test/
    n_avaria_1.ply    ← superfície normal (avaliação)
    avaria_1.ply      ← superfície com rachadura (avaliação)
    ...
```

Os arquivos PLY devem conter: `x, y, z, red, green, blue, nx, ny, nz, scalar_R, scalar_labels`  
(gerados pelo CloudCompare com cálculo de normais e campos escalares).

---

## Como rodar

```bash
# Instalar dependências
uv sync

# Modelo Professor-Aluno
uv run python src/teacher_student_v1.py

# GAN
uv run python src/gan_rachaduras_v5.py
```

**Requisitos:** Python 3.11, PyTorch 2.7+cu128, CUDA 12.8, GPU com ≥ 8GB VRAM  
Testado em: RTX 5060 Ti 16GB, WSL2 Ubuntu 24.04

---

## Pesos pré-treinados

Colocar em `pretrained_models/`:

| Arquivo | Uso | Download |
|---------|-----|---------|
| `ptv3_scannet200.pth` | Teacher (Professor-Aluno) | HuggingFace Pointcept/PointTransformerV3 |
| `kpconv_s3dis_*.pth` | Encoder do GAN | HuguesTHOMAS/KPConv-PyTorch releases |

---

## Referências principais

- Deng & Li, *Decoupled Anomaly Detection*, CVPR 2022 — base do modelo Professor-Aluno
- Wu et al., *Point Transformer V3*, CVPR 2024 (Oral) — backbone do Professor
- Roth et al., *PatchCore*, CVPR 2022 — memory bank de features normais
- Yao et al., *Semi-Push-Pull Contrastive Learning*, CVPR 2023 — loss semi-supervisionada
