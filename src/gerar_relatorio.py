"""
Gera relatorio_progresso_tcc.html — relatório de progresso do TCC.
Uso: uv run python src/gerar_relatorio.py
Depois: abra no browser → Ctrl+P → Salvar como PDF
"""
import base64
import pathlib
from datetime import date

BASE = pathlib.Path(__file__).parent.parent

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; color: #2c3e50;
       font-size: 11pt; line-height: 1.65; background: #fff; }
.page { max-width: 820px; margin: 0 auto; padding: 40px 55px; }
h1 { font-size: 22pt; color: #1a1a2e; margin-bottom: 6px; }
h2 { font-size: 14pt; color: #1a1a2e; border-bottom: 2px solid #3498db;
     padding-bottom: 5px; margin: 36px 0 14px; }
h3 { font-size: 11.5pt; color: #34495e; margin: 18px 0 8px; font-weight: 600; }
p  { margin-bottom: 10px; text-align: justify; }
ul { margin: 6px 0 12px 24px; }
li { margin-bottom: 5px; }
.capa { text-align: center; padding: 70px 0 50px; border-bottom: 3px solid #3498db; margin-bottom: 36px; }
.capa .subtitle  { font-size: 13pt; color: #555; margin-top: 8px; }
.capa .subtitlesm { font-size: 11pt; color: #777; margin-top: 4px; }
.capa .meta { margin-top: 44px; color: #666; font-size: 10pt; line-height: 2.2; }
table { width: 100%; border-collapse: collapse; margin: 14px 0 20px; font-size: 10pt; }
thead tr { background: #2c3e50; color: #fff; }
th { padding: 8px 10px; text-align: left; font-weight: 600; }
td { padding: 7px 10px; border-bottom: 1px solid #e4e8ed; vertical-align: top; }
tr:nth-child(even) td { background: #f7f9fc; }
.badge-ok   { background:#27ae60; color:#fff; padding:2px 8px; border-radius:3px; font-size:8.5pt; white-space:nowrap; }
.badge-no   { background:#e74c3c; color:#fff; padding:2px 8px; border-radius:3px; font-size:8.5pt; white-space:nowrap; }
.badge-wip  { background:#e67e22; color:#fff; padding:2px 8px; border-radius:3px; font-size:8.5pt; white-space:nowrap; }
.badge-plan { background:#3498db; color:#fff; padding:2px 8px; border-radius:3px; font-size:8.5pt; white-space:nowrap; }
figure { margin: 22px 0; text-align: center; }
figure img { max-width: 100%; border: 1px solid #dde3ea; border-radius: 5px; }
figcaption { font-size: 9pt; color: #666; margin-top: 7px; font-style: italic; }
.highlight { background: #eaf4fb; border-left: 4px solid #3498db;
             padding: 12px 16px; margin: 16px 0; border-radius: 0 4px 4px 0; font-size: 10.5pt; }
.warn      { background: #fef9e7; border-left: 4px solid #f39c12;
             padding: 12px 16px; margin: 16px 0; border-radius: 0 4px 4px 0; font-size: 10.5pt; }
.ok-row td { background: #eafaf1 !important; }
@media print {
  body { font-size: 10pt; }
  .page { padding: 15px 30px; max-width: 100%; }
  h2 { page-break-before: always; margin-top: 10px; }
  .capa { page-break-after: always; }
  figure img { max-width: 88%; }
  .highlight, .warn { page-break-inside: avoid; }
  table { page-break-inside: avoid; font-size: 9pt; }
}
"""


def img_b64(rel_path: str) -> str:
    p = BASE / rel_path
    if not p.exists():
        return ""
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:image/png;base64,{data}"


def build_sections() -> list[str]:
    sf_img  = img_b64("visualizations_sf/supervised_feature_analysis.png")
    gmm_img = img_b64("visualizations_sf/gmm_interval_analysis.png")
    today   = date.today().strftime("%d de %B de %Y").replace(
        "January","Janeiro").replace("February","Fevereiro").replace(
        "March","Março").replace("April","Abril").replace(
        "May","Maio").replace("June","Junho").replace(
        "July","Julho").replace("August","Agosto").replace(
        "September","Setembro").replace("October","Outubro").replace(
        "November","Novembro").replace("December","Dezembro")

    s = []

    # ── CAPA ──────────────────────────────────────────────────────────────────
    s.append(f"""
<div class="capa">
  <h1>Relatório de Progresso</h1>
  <div class="subtitle">Detecção de Avarias em Nuvens de Pontos 3D</div>
  <div class="subtitlesm">Igreja dos Homens Pretos — São Luís, MA</div>
  <div class="meta">
    Aluno: Carlos Matheus<br>
    Orientadora: Prof.ª [Nome da Orientadora]<br>
    Curso: [Curso] — [Universidade]<br>
    {today}
  </div>
</div>""")

    # ── 1. CONTEXTO ───────────────────────────────────────────────────────────
    s.append("""
<h2>1. Contexto</h2>
<p>Este trabalho tem como objetivo desenvolver um sistema automático de
<strong>detecção e segmentação de rachaduras</strong> em nuvens de pontos 3D
obtidas por varredura a laser terrestre (TLS — <em>Terrestrial Laser Scanner</em>)
da Igreja dos Homens Pretos, em São Luís, MA. A edificação é tombada pelo
patrimônio histórico e exige inspeção periódica de integridade estrutural.</p>

<p>Nuvens de pontos TLS capturam a geometria e a reflectância da superfície
com alta resolução espacial. Cada ponto carrega coordenadas (x, y, z),
cor RGB, normais estimadas e um campo escalar de intensidade de retorno do
laser (<em>scalar field</em>). O desafio central é identificar automaticamente
os pontos que correspondem a rachaduras sem depender de inspeção manual exaustiva.</p>

<div class="highlight">
<strong>Objetivo:</strong> Comparar abordagens supervisionadas e
não-supervisionadas para segmentação de rachaduras em nuvens de pontos 3D,
avaliando precisão, recall e AUROC em um dataset real de patrimônio histórico.
</div>""")

    # ── 2. OS DADOS ───────────────────────────────────────────────────────────
    s.append("""
<h2>2. Os Dados</h2>
<p>O dataset foi coletado com CloudCompare v2.14 entre dezembro de 2024 e
fevereiro de 2025, abrangendo pilastras e paredes internas da Igreja.</p>

<h3>Estrutura dos arquivos PLY</h3>
<table>
  <thead><tr><th>Colunas</th><th>Feature</th><th>Descrição</th></tr></thead>
  <tbody>
    <tr><td>0–2</td><td>x, y, z</td><td>Coordenadas 3D, normalizadas para esfera unitária por nuvem</td></tr>
    <tr><td>3–5</td><td>r, g, b</td><td>Cor RGB normalizada [0, 1]</td></tr>
    <tr><td>6–8</td><td>nx, ny, nz</td><td>Normal estimada por vizinhança local (Open3D)</td></tr>
    <tr><td>9</td><td><strong>scalar_field</strong></td><td>Intensidade de retorno do laser, normalizada por nuvem [0, 1]</td></tr>
    <tr><td>10–13</td><td>curv, dens, var, sv</td><td>Geometria local: curvatura, densidade, variância, variação de superfície</td></tr>
    <tr><td>14–15</td><td>lum, sat</td><td>Luminosidade (R+G+B)/3 e saturação max(RGB)−min(RGB)</td></tr>
    <tr><td>—</td><td><strong>scalar_labels</strong></td><td>Ground truth: 0 = normal, 1 = rachadura (apenas avaria_*.ply)</td></tr>
  </tbody>
</table>

<h3>Estatísticas do dataset</h3>
<table>
  <thead><tr><th>Conjunto</th><th>Arquivos</th><th>Pontos (aprox.)</th><th>SF médio</th><th>Lum. média</th></tr></thead>
  <tbody>
    <tr><td>avaria_*.ply (com rachadura)</td><td>35 PLY</td><td>~800 K</td><td>—</td><td>—</td></tr>
    <tr><td>n_avaria_*.ply (superfície normal)</td><td>29 PLY</td><td>~500 K</td><td>—</td><td>—</td></tr>
    <tr><td><strong>Pontos de rachadura</strong></td><td>—</td><td>~6,62% do total</td><td><strong>~0.21</strong></td><td><strong>~0.21</strong></td></tr>
    <tr><td><strong>Pontos normais</strong></td><td>—</td><td>~93,38%</td><td><strong>~0.43–0.64</strong></td><td><strong>~0.64</strong></td></tr>
  </tbody>
</table>

<div class="highlight">
<strong>Propriedade-chave dos dados:</strong> Os <em>scalar_labels</em> foram
derivados do <em>scalar_field</em> no CloudCompare — rachaduras correspondem a
regiões de baixo retorno laser (superfície escura e irregular). O scalar_field
já contém a informação dos labels de forma contínua, o que explica sua
dominância em todas as análises.
</div>""")

    # ── 3. DESCOBERTA CENTRAL ─────────────────────────────────────────────────
    s.append(f"""
<h2>3. Descoberta Central — Scalar Field como Único Discriminador Linear</h2>
<p>Uma análise supervisionada completa avaliou a capacidade discriminativa
individual de cada feature em relação ao label de rachadura. Para cada feature,
calculou-se o AUROC dentro de cada nuvem separadamente (<em>per-cloud</em>) — métrica
mais justa que o AUROC pooled global, que sofre influência das diferenças de
escala entre nuvens.</p>

<h3>AUROC individual por feature</h3>
<table>
  <thead><tr><th>Feature</th><th>AUROC pooled</th><th>AUROC per-cloud (med.)</th><th>per-cloud std</th><th>|Spearman|</th></tr></thead>
  <tbody>
    <tr class="ok-row"><td><strong>sf (scalar field)</strong></td><td><strong>0.8898</strong></td><td><strong>0.9888</strong></td><td>0.1236</td><td><strong>0.3322</strong></td></tr>
    <tr><td>lum</td><td>0.5800</td><td>0.5082</td><td>0.0069</td><td>0.0682</td></tr>
    <tr><td>b (azul)</td><td>0.5814</td><td>0.5073</td><td>0.0062</td><td>0.0694</td></tr>
    <tr><td>sat</td><td>0.5443</td><td>0.5084</td><td>0.0109</td><td>0.0378</td></tr>
    <tr><td>r (vermelho)</td><td>0.5467</td><td>0.5075</td><td>0.0087</td><td>0.0398</td></tr>
    <tr><td>sv (surface variation)</td><td>0.5276</td><td>0.5060</td><td>0.0088</td><td>0.0235</td></tr>
    <tr><td>curv</td><td>0.5078</td><td>0.5086</td><td>0.0085</td><td>0.0067</td></tr>
    <tr><td colspan="5" style="color:#888;font-style:italic;text-align:center;padding:6px">
      x, y, z, nx, ny, nz, dens, var, g: AUROC per-cloud ≈ 0.505–0.509 (essencialmente aleatório)</td></tr>
  </tbody>
</table>

<div class="highlight">
<strong>Conclusão:</strong> O vetor 16D colapsa efetivamente em 1D para este
problema. O scalar_field domina com AUROC per-cloud de 0.9888; todas as outras
features ficam em ≈ 0.507 — indistinguíveis do acaso quando avaliadas por nuvem.
</div>

<p>Adicionalmente, testaram-se 12 combinações de features (sf×lum, sf×sat,
sf/(dens+ε), sf×(1−curv), etc.). <strong>Nenhuma superou o SF puro</strong> —
a melhor combinação (sf−lum) obteve AUROC=0.963 contra 0.989 do SF sozinho,
confirmando que combinações lineares introduzem ruído sem acrescentar sinal.</p>

{ f'<figure><img src="{sf_img}" alt="Análise supervisionada de features"><figcaption>Figura 1 — Esquerda: AUROC per-cloud por feature (sf domina com 0.989). Centro: heatmap |Spearman| entre features e label (apenas sf e seus proxies de cor têm correlação visível). Direita: AUROC das combinações (todas abaixo do SF puro — linha azul).</figcaption></figure>' if sf_img else '' }""")

    # ── 4. ANÁLISE GMM ────────────────────────────────────────────────────────
    s.append(f"""
<h2>4. Análise GMM — Separabilidade por Nuvem</h2>
<p>Para cada nuvem de avaria, ajustou-se um Gaussian Mixture Model (GMM) ao
scalar_field. O componente minoritário é identificado como o cluster de
rachadura, extraindo-se três parâmetros: <strong>μ_crack</strong> (centro),
<strong>σ_crack</strong> (spread) e <strong>overlap_ratio</strong> — fração de
pontos normais que caem dentro do intervalo [μ_crack ± 2σ_crack]. Overlap &gt; 0.40
indica que as distribuições se sobrepõem demais e o SF sozinho não consegue
separar os pontos com confiança.</p>

<h3>Clouds por dificuldade (overlap_ratio descendente)</h3>
<table>
  <thead><tr><th>Arquivo</th><th>Bimodal</th><th>μ crack</th><th>σ crack</th><th>Overlap</th><th>AUROC SF</th><th>n crack</th></tr></thead>
  <tbody>
    <tr><td>avaria_35.ply</td><td>Sim</td><td>0.285</td><td>0.094</td><td style="color:#e74c3c"><strong>0.628 ⚠</strong></td><td>1.000</td><td>3.379</td></tr>
    <tr><td>avaria_34.ply</td><td>Sim</td><td>0.481</td><td>0.141</td><td style="color:#e74c3c"><strong>0.592 ⚠</strong></td><td>1.000</td><td>1.306</td></tr>
    <tr><td>avaria_4.ply</td> <td>Sim</td><td>0.230</td><td>0.090</td><td style="color:#e74c3c"><strong>0.572 ⚠</strong></td><td>1.000</td><td>1.826</td></tr>
    <tr><td>avaria_38.ply</td><td>Sim</td><td>0.233</td><td>0.098</td><td style="color:#e74c3c"><strong>0.558 ⚠</strong></td><td>0.968</td><td>588</td></tr>
    <tr><td>avaria_3.ply</td> <td>Sim</td><td>0.393</td><td>0.177</td><td style="color:#e74c3c"><strong>0.533 ⚠</strong></td><td>1.000</td><td>985</td></tr>
    <tr><td>avaria_9.ply</td> <td>Sim</td><td>0.290</td><td>0.125</td><td style="color:#e67e22"><strong>0.463 ⚠</strong></td><td>0.986</td><td>957</td></tr>
    <tr><td>avaria_32.ply</td><td>Sim</td><td>0.264</td><td>0.137</td><td style="color:#e67e22"><strong>0.422 ⚠</strong></td><td>1.000</td><td>355</td></tr>
    <tr><td>avaria_26.ply</td><td>Sim</td><td>0.324</td><td>0.087</td><td>0.181</td><td>0.803</td><td>736</td></tr>
    <tr><td>avaria_17.ply</td><td>Sim</td><td>0.170</td><td>0.032</td><td>0.000</td><td style="color:#e74c3c"><strong>0.534</strong></td><td>1.020</td></tr>
    <tr><td>avaria_33.ply</td><td>Sim</td><td>0.616</td><td>0.109</td><td>0.000</td><td style="color:#e74c3c"><strong>0.595</strong></td><td>1.723</td></tr>
    <tr><td>avaria_21.ply</td><td>Sim</td><td>0.132</td><td>0.104</td><td>0.012</td><td style="color:#e67e22"><strong>0.611</strong></td><td>94</td></tr>
    <tr><td>avaria_10.ply</td><td><em>Não</em></td><td>—</td><td>—</td><td>—</td><td>1.000</td><td>1.056</td></tr>
    <tr><td>avaria_2.ply</td> <td><em>Não</em></td><td>—</td><td>—</td><td>—</td><td>1.000</td><td>802</td></tr>
    <tr><td>avaria_39.ply</td><td><em>Não</em></td><td>—</td><td>—</td><td>—</td><td>1.000</td><td>395</td></tr>
  </tbody>
</table>

<div class="highlight">
32 nuvens avaria &nbsp;|&nbsp; 3 unimodais (GMM não encontrou clusters distintos) &nbsp;|&nbsp;
7 com overlap &gt; 0.40 &nbsp;|&nbsp;
AUROC SF médio: <strong>0.9350</strong> (std: 0.1256 &nbsp;|&nbsp; mín: 0.5335)
</div>

{ f'<figure><img src="{gmm_img}" alt="Análise GMM por nuvem"><figcaption>Figura 2 — Esquerda: scatter overlap_ratio × AUROC SF. Nuvens com overlap alto mas AUROC=1.0 ainda são separáveis na prática (SF fica na cauda extrema do intervalo de crack). avaria_17 e avaria_33 (AUROC&lt;0.62) são genuinamente difíceis — SF médio de crack ≈ SF médio de normal. Direita: distribuição do overlap_ratio — maioria das nuvens tem overlap baixo.</figcaption></figure>' if gmm_img else '' }""")

    # ── 5. NUVENS DIFÍCEIS ────────────────────────────────────────────────────
    s.append("""
<h2>5. Nuvens Difíceis</h2>
<p>Três nuvens apresentam AUROC &lt; 0.65 no SF puro — o scalar field não
discrimina efetivamente os pontos de rachadura nelas:</p>

<table>
  <thead><tr><th>Arquivo</th><th>AUROC SF</th><th>n crack</th><th>SF médio crack</th><th>SF médio normal</th><th>Razão da dificuldade</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>avaria_17.ply</strong></td>
      <td style="color:#e74c3c"><strong>0.534</strong></td>
      <td>1.020</td><td>0.429</td><td>0.423</td>
      <td>SF médio de crack ≈ SF médio de normal. Rachaduras com SF heterogêneo (0.017–0.851) — crack visível mas com reflectância variada</td>
    </tr>
    <tr>
      <td><strong>avaria_33.ply</strong></td>
      <td style="color:#e74c3c"><strong>0.595</strong></td>
      <td>1.723</td><td>0.119</td><td>0.225</td>
      <td>GMM identificou o componente errado (μ=0.62 em vez de 0.12). Distribuição de SF atípica nesta nuvem</td>
    </tr>
    <tr>
      <td><strong>avaria_21.ply</strong></td>
      <td style="color:#e67e22"><strong>0.611</strong></td>
      <td>94</td><td>0.540</td><td>0.546</td>
      <td>Poucos pontos de crack (94) com SF alto para rachadura (média 0.54) — caso atípico</td>
    </tr>
  </tbody>
</table>

<div class="warn">
<strong>Implicação:</strong> Para essas três nuvens, qualquer abordagem baseada
puramente em SF vai falhar independentemente do modelo. A feature
<em>surface variation</em> (<strong>sv</strong>) — com sinal residual de AUROC=0.69
após remover o efeito do SF — é a principal candidata a ajudar nesses casos,
pois captura a variação abrupta de normais nas bordas de rachaduras.
</div>""")

    # ── 6. JORNADA DAS ABORDAGENS ─────────────────────────────────────────────
    s.append("""
<h2>6. Jornada das Abordagens</h2>

<table>
  <thead><tr><th>Data</th><th>Abordagem</th><th>Motivação</th><th>Resultado</th><th>Status</th></tr></thead>
  <tbody>
    <tr class="ok-row">
      <td>Jan–Mar&nbsp;2026</td>
      <td><strong>teacher_student_v1</strong></td>
      <td>Reverse Distillation com PTv3 frozen + Push-Pull Loss semi-supervisionada</td>
      <td>AUROC=0.952 &nbsp; F1=0.903 &nbsp; P=0.989</td>
      <td><span class="badge-wip">Melhor agregado ⚠ std=0.182</span></td>
    </tr>
    <tr>
      <td>Abr&nbsp;2026</td>
      <td>SF-GMM</td>
      <td>Threshold GMM no scalar field — abordagem sem deep learning</td>
      <td>AUROC=0.828 &nbsp; F1=0.274 &nbsp; P=0.167</td>
      <td><span class="badge-no">Precisão ruim</span></td>
    </tr>
    <tr>
      <td>14&nbsp;Abr</td>
      <td>teacher_student_v3 (pretraining geométrico)</td>
      <td>Pre-treino do backbone em geometria antes da distilação</td>
      <td>AUROC=0.773 — catastrophic forgetting</td>
      <td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>20&nbsp;Abr</td>
      <td>DomainAdapter (Caminho A)</td>
      <td>Adapter entre teacher e student para domínio S3DIS→crack</td>
      <td>AUROC=0.686 — adapter suaviza sinal de crack</td>
      <td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>21&nbsp;Abr</td>
      <td>SF MAE v1 (XYZ+RGB+normais)</td>
      <td>MAE: modelo prevê SF a partir de outras features; erro = anomalia</td>
      <td>AUROC=0.30 invertido — atalho RGB→SF</td>
      <td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>22&nbsp;Abr</td>
      <td>Spatial GNN</td>
      <td>Refinar scores SF-GMM com GNN espacial para remover FP isolados</td>
      <td>AUROC=0.79 &nbsp; P=0.363 — pseudo-labels ruidosos demais</td>
      <td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>22&nbsp;Abr</td>
      <td>scalar_field_unsup (DGCNN)</td>
      <td>DGCNN self-supervised com pseudo-labels do SF-GMM</td>
      <td>AUROC=0.830 &nbsp; F1=0.270 — L_contrast estagnada</td>
      <td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>23&nbsp;Abr</td>
      <td><strong>XGBoost + sv + sat</strong></td>
      <td>Supervisionado: 15 features (SF + geométricas + GMM stats), LOCO-CV</td>
      <td>—</td>
      <td><span class="badge-wip">A rodar (~30 min)</span></td>
    </tr>
    <tr>
      <td>—</td>
      <td><strong>PTv3 Linear Probe</strong></td>
      <td>MLP(512→128→2) sobre backbone PTv3 frozen com Focal Loss</td>
      <td>Estimado: AUROC &gt; 0.80</td>
      <td><span class="badge-plan">Planejado (~1 h)</span></td>
    </tr>
    <tr>
      <td>—</td>
      <td><strong>PTv3 Partial Fine-tuning</strong></td>
      <td>Últimos 50% blocos PTv3 + cabeça treináveis, LLRD + Focal + Dice</td>
      <td>Estimado: AUROC &gt; 0.95</td>
      <td><span class="badge-plan">Planejado (~4–6 h)</span></td>
    </tr>
  </tbody>
</table>""")

    # ── 7. MELHOR RESULTADO ───────────────────────────────────────────────────
    s.append("""
<h2>7. teacher_student_v1 — Análise Detalhada</h2>
<p>O teacher_student_v1 implementa Reverse Distillation (Deng &amp; Li, CVPR 2022)
adaptada para nuvens de pontos. O Teacher é o PointTransformer V3 (Wu et al.,
CVPR 2024) pré-treinado em ScanNet200, mantido completamente congelado. O Student
é um decoder treinável do zero, supervisionado pela distância cosseno entre suas
ativações e as do Teacher em 3 escalas espaciais.</p>

<h3>Métricas agregadas — Run 13/04/2026</h3>
<table>
  <thead><tr><th>Estratégia</th><th>Precisão</th><th>Recall</th><th>F1</th><th>IoU</th><th>AUROC</th><th>AP</th></tr></thead>
  <tbody>
    <tr><td>G-mean</td><td>0.889</td><td>0.918</td><td>0.903</td><td>0.824</td><td>0.952</td><td>0.931</td></tr>
    <tr><td>F1-ótimo</td><td>0.989</td><td>0.899</td><td>0.942</td><td>0.890</td><td>0.952</td><td>0.931</td></tr>
    <tr><td>F-beta(0.5)</td><td>0.994</td><td>0.889</td><td>0.939</td><td>0.885</td><td>0.952</td><td>0.931</td></tr>
  </tbody>
</table>

<div class="warn">
<strong>Limitação crítica das métricas agregadas:</strong> Os números acima somam
todos os pontos de todas as 31 nuvens. As 22 nuvens com F1 ≥ 0.90 dominam e
escondem falhas graves em nuvens específicas. O std por nuvem é 0.182 —
variância muito alta para um método confiável.
</div>

<h3>Distribuição real de F1 por nuvem — computado a partir dos PLY de predição</h3>
<table>
  <thead><tr><th>Arquivo</th><th>n crack</th><th>Precisão</th><th>Recall</th><th>F1</th><th>Problema</th></tr></thead>
  <tbody>
    <tr style="background:#fde8e8"><td><strong>avaria_21.ply</strong></td><td>94</td><td>0.222</td><td>0.021</td>
      <td style="color:#c0392b"><strong>0.039</strong></td><td>Recall=2% — modelo praticamente não detecta</td></tr>
    <tr style="background:#fdf0e0"><td>avaria_40.ply</td><td>627</td><td>0.523</td><td>1.000</td>
      <td style="color:#e67e22"><strong>0.687</strong></td><td>47% falsos positivos</td></tr>
    <tr style="background:#fdf0e0"><td>avaria_26.ply</td><td>736</td><td>0.716</td><td>0.779</td>
      <td style="color:#e67e22"><strong>0.746</strong></td><td>Performance moderada</td></tr>
    <tr style="background:#fdf0e0"><td>avaria_17.ply</td><td>1.020</td><td>0.810</td><td>0.739</td>
      <td style="color:#e67e22"><strong>0.773</strong></td><td>SF não discrimina nesta nuvem — esperado</td></tr>
    <tr style="background:#fdf0e0"><td>avaria_36.ply</td><td>1.251</td><td>0.677</td><td>0.973</td>
      <td style="color:#e67e22"><strong>0.798</strong></td><td>Recall alto, precisão baixa</td></tr>
    <tr style="background:#fdf0e0"><td>avaria_41.ply</td><td>3.765</td><td>1.000</td><td>0.682</td>
      <td style="color:#e67e22"><strong>0.811</strong></td><td>Perde 32% dos cracks — score map inflado na nuvem inteira</td></tr>
    <tr style="background:#fdf0e0"><td>avaria_33.ply</td><td>1.723</td><td>0.964</td><td>0.706</td>
      <td style="color:#e67e22"><strong>0.815</strong></td><td>SF atípico — recall comprometido</td></tr>
    <tr><td colspan="6" style="text-align:center;color:#666;font-style:italic;padding:8px">
      avaria_22, 30, 24: F1 = 0.90–0.92 &nbsp;&nbsp;|&nbsp;&nbsp;
      avaria_13, 19: F1 = 0.95–0.97 &nbsp;&nbsp;|&nbsp;&nbsp;
      22/31 nuvens: F1 ≥ 0.99</td></tr>
  </tbody>
</table>

<table style="margin-top:8px">
  <thead><tr><th>Estatística per-nuvem (F1)</th><th>Valor</th></tr></thead>
  <tbody>
    <tr><td>Média</td><td>0.900</td></tr>
    <tr style="background:#fde8e8"><td><strong>Desvio padrão</strong></td><td><strong>0.182</strong> — alta variância</td></tr>
    <tr style="background:#fde8e8"><td><strong>Mínimo</strong></td><td><strong>0.039</strong> (avaria_21)</td></tr>
    <tr><td>Máximo</td><td>1.000</td></tr>
    <tr style="background:#fde8e8"><td>F1 &lt; 0.50</td><td><strong>1 nuvem</strong></td></tr>
    <tr style="background:#fdf0e0"><td>F1 &lt; 0.70</td><td><strong>2 nuvens</strong></td></tr>
    <tr><td>F1 ≥ 0.90</td><td>22/31 nuvens (71%)</td></tr>
  </tbody>
</table>

<div class="warn">
<strong>Problema de localização visual confirmado:</strong> A inspeção das
predições em CloudCompare revela que em avaria_41 o score map está globalmente
elevado na nuvem inteira — o modelo não localiza a rachadura, apenas aplica um
threshold num score ruidoso. Em avaria_07 os pontos vermelhos aparecem distribuídos
sem estrutura linear, inconsistentes com o padrão esperado de uma rachadura.
</div>

<div class="highlight">
<strong>Conclusão revisada:</strong> O teacher_student_v1 apresenta a melhor
métrica <em>agregada</em> entre as abordagens testadas, mas com variância alta
por nuvem (std=0.182) e problemas de localização espacial. Não pode ser apresentado
como solução definitiva sem demonstrar consistência por nuvem e qualidade visual.
</div>

<div class="warn">
<strong>Run invalidado:</strong> Run de 05/04/2026 com F1=99.8% descartado por
data leakage — scalar_labels incluído como feature de entrada.
</div>""")

    # ── 8. ABORDAGENS DESCARTADAS ─────────────────────────────────────────────
    s.append("""
<h2>8. Abordagens Descartadas</h2>
<table>
  <thead><tr><th>Abordagem</th><th>P</th><th>R</th><th>F1</th><th>AUROC</th><th>Razão da falha</th></tr></thead>
  <tbody>
    <tr>
      <td>SF-GMM (G-mean)</td><td>0.167</td><td>0.768</td><td>0.274</td><td>0.828</td>
      <td>Threshold no SF bruto marca qualquer superfície escura como rachadura — 83% de falsos positivos. AUROC razoável mas precisão inaceitável para uso prático</td>
    </tr>
    <tr>
      <td>teacher_student_v3 (pretraining geométrico)</td><td>0.191</td><td>0.636</td><td>0.294</td><td>0.773</td>
      <td>Pre-treino geométrico completo do backbone causou catastrophic forgetting — features PTv3 ficaram menos discriminativas para rachaduras</td>
    </tr>
    <tr>
      <td>DomainAdapter (Caminho A)</td><td>—</td><td>—</td><td>—</td><td>0.686</td>
      <td>Adapter suaviza o sinal de crack ao mapear features PTv3 para espaço intermediário, perdendo a discriminação que o Teacher já codificava</td>
    </tr>
    <tr>
      <td>SF MAE v1 (XYZ+RGB+normais, 9D)</td><td>—</td><td>—</td><td>—</td><td>~0.30 inv.</td>
      <td>RGB é proxy direto de SF (cracks escuros → SF baixo). Modelo aprende atalho: prediz SF baixo corretamente em cracks → erro baixo onde deveria ser alto → AUROC invertido</td>
    </tr>
    <tr>
      <td>Spatial GNN</td><td>0.363</td><td>—</td><td>—</td><td>0.790</td>
      <td>Pseudo-labels do SF-GMM têm precision=0.363 — GNN treina em dados muito ruidosos e propaga os erros em vez de corrigi-los</td>
    </tr>
    <tr>
      <td>scalar_field_unsup (DGCNN self-sup)</td><td>—</td><td>—</td><td>0.270</td><td>0.830</td>
      <td>kNN em espaço de features 512D é inútil para esta tarefa; L_contrast estagnada; Teacher PTv3 é cego para crack (trata crack e parede como mesma classe)</td>
    </tr>
  </tbody>
</table>""")

    # ── 9. TABELA COMPARATIVA GERAL ───────────────────────────────────────────
    s.append("""
<h2>9. Tabela Comparativa Geral</h2>
<table>
  <thead>
    <tr><th>Abordagem</th><th>Precisão</th><th>Recall</th><th>F1</th><th>AUROC</th><th>AP</th><th>Supervisão</th><th>Status</th></tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>teacher_student_v1</strong> (G-mean)</td>
      <td>0.889</td><td>0.918</td><td>0.903</td>
      <td>0.952</td><td>0.931</td>
      <td>Semi-sup</td><td><span class="badge-wip">Melhor agregado ⚠ std=0.182</span></td>
    </tr>
    <tr>
      <td><strong>teacher_student_v1</strong> (F1-ótimo)</td>
      <td>0.989</td><td>0.899</td><td>0.942</td>
      <td>0.952</td><td>0.931</td>
      <td>Semi-sup</td><td><span class="badge-wip">Melhor agregado ⚠ F1 min=0.039</span></td>
    </tr>
    <tr>
      <td>SF puro (threshold ótimo)</td><td>—</td><td>—</td><td>—</td><td>0.935</td><td>—</td>
      <td>Nenhuma</td><td><span class="badge-ok">Teto linear SF</span></td>
    </tr>
    <tr>
      <td>SF-GMM (G-mean)</td><td>0.167</td><td>0.768</td><td>0.274</td><td>0.828</td><td>—</td>
      <td>Nenhuma</td><td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>SF-GMM (F1-ótimo)</td><td>0.244</td><td>0.517</td><td>0.331</td><td>0.828</td><td>—</td>
      <td>Nenhuma</td><td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>scalar_field_unsup (DGCNN)</td><td>—</td><td>—</td><td>0.270</td><td>0.830</td><td>—</td>
      <td>Auto-sup</td><td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>teacher_student_v3</td><td>0.191</td><td>0.636</td><td>0.294</td><td>0.773</td><td>0.297</td>
      <td>Semi-sup</td><td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>Spatial GNN</td><td>0.363</td><td>—</td><td>—</td><td>0.790</td><td>—</td>
      <td>Semi-sup</td><td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>DomainAdapter</td><td>—</td><td>—</td><td>—</td><td>0.686</td><td>—</td>
      <td>Semi-sup</td><td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td>SF MAE v1 (com RGB)</td><td>—</td><td>—</td><td>—</td><td>~0.30 inv.</td><td>—</td>
      <td>Auto-sup</td><td><span class="badge-no">Encerrado</span></td>
    </tr>
    <tr>
      <td><strong>XGBoost + sv + sat</strong></td><td>TBD</td><td>TBD</td><td>TBD</td><td>TBD</td><td>TBD</td>
      <td>Supervisionado</td><td><span class="badge-wip">A rodar (~30 min)</span></td>
    </tr>
    <tr>
      <td><strong>PTv3 Linear Probe</strong></td><td>TBD</td><td>TBD</td><td>TBD</td><td>&gt;0.80 est.</td><td>TBD</td>
      <td>Supervisionado</td><td><span class="badge-plan">Planejado (~1 h)</span></td>
    </tr>
    <tr>
      <td><strong>PTv3 Partial Fine-tuning</strong></td><td>TBD</td><td>TBD</td><td>TBD</td><td>&gt;0.95 est.</td><td>TBD</td>
      <td>Supervisionado</td><td><span class="badge-plan">Planejado (~4–6 h)</span></td>
    </tr>
  </tbody>
</table>""")

    # ── 10. PARTIAL CORRELATION ───────────────────────────────────────────────
    s.append(f"""
<h2>10. Partial Correlation e Combinações de Features</h2>
<p>Para identificar features com sinal <em>além</em> do scalar field,
calculou-se a correlação de cada feature com o label após remover linearmente
o efeito do SF (resíduo da regressão feature ~ SF). Um AUROC residual &gt; 0.55
indica que a feature carrega informação ortogonal ao SF — potencialmente útil
para um modelo não-linear.</p>

<h3>AUROC do resíduo (sinal independente do SF)</h3>
<table>
  <thead><tr><th>Feature</th><th>|Spearman| parcial</th><th>AUROC resíduo</th><th>Interpretação física</th></tr></thead>
  <tbody>
    <tr class="ok-row"><td><strong>sv (surface variation)</strong></td><td><strong>0.1599</strong></td><td><strong>0.6876</strong></td><td>Variação abrupta de normais nas bordas de rachaduras</td></tr>
    <tr class="ok-row"><td><strong>sat (saturação)</strong></td><td><strong>0.0974</strong></td><td><strong>0.6143</strong></td><td>Cracks acinzentados vs superfície com cor</td></tr>
    <tr><td>nx (normal x)</td><td>0.0827</td><td>0.5970</td><td>Orientação da parede — proxy de localização</td></tr>
    <tr><td>r (vermelho)</td><td>0.0787</td><td>0.5923</td><td>Proxy residual de cor</td></tr>
    <tr><td>lum (luminosidade)</td><td>0.0593</td><td>0.5696</td><td>Altamente correlacionado com SF — sinal marginal</td></tr>
    <tr><td colspan="4" style="color:#888;font-style:italic;text-align:center;padding:6px">
      Demais features (y, z, ny, nz, g, b, curv, dens, var): AUROC residual &lt; 0.54</td></tr>
  </tbody>
</table>

<div class="highlight">
<strong>Ação:</strong> <em>sv</em> e <em>sat</em> foram adicionados ao XGBoost
como features diretas (cols 13 e 15 do vetor 16D), juntamente com
<em>sv_contrast</em> (sv[i] / mean(sv[vizinhos])). Combinações lineares simples
de sf com outros features <strong>sempre degradam</strong> o AUROC — é necessário
um modelo não-linear para extrair o sinal residual de sv e sat.
</div>""")

    # ── 11. PRÓXIMOS PASSOS ───────────────────────────────────────────────────
    s.append("""
<h2>11. Próximos Passos</h2>
<table>
  <thead><tr><th>#</th><th>Abordagem</th><th>Descrição técnica</th><th>AUROC estimado</th><th>Tempo</th></tr></thead>
  <tbody>
    <tr>
      <td><strong>1</strong></td>
      <td><strong>XGBoost + sv + sat</strong></td>
      <td>15 features por ponto: SF, SF_rank, sf_contrast, z_crack, GMM stats (μ, σ, weight, overlap), percentis SF, <strong>sv</strong>, <strong>sat</strong>, <strong>sv_contrast</strong>.
          Leave-one-cloud-out CV em 35 nuvens avaria. Pós-processamento DBSCAN para remover FP isolados.</td>
      <td>0.93–0.96</td>
      <td>~30 min</td>
    </tr>
    <tr>
      <td><strong>2</strong></td>
      <td><strong>PTv3 Linear Probe</strong></td>
      <td>Backbone PTv3 (ScanNet200) 100% congelado. Cabeça MLP(512→128→2) treinada com Focal Loss (γ=2).
          Valida se o PTv3 raw já codifica sinal de rachadura em suas representações internas.</td>
      <td>&gt; 0.80</td>
      <td>~1 h</td>
    </tr>
    <tr>
      <td><strong>3</strong></td>
      <td><strong>PTv3 Partial Fine-tuning</strong></td>
      <td>Primeiros 50% dos blocos congelados. Últimos blocos + cabeça treináveis com LLRD
          (lr decai × 0.7 por bloco). Loss: 0.7 × Focal(γ=2) + 0.3 × Dice. Protocolo PTv3 paper (CVPR 2024).
          <em>Executar apenas se Linear Probe &lt; 0.90.</em></td>
      <td>&gt; 0.95</td>
      <td>4–6 h</td>
    </tr>
  </tbody>
</table>""")

    # ── 12. CONCLUSÃO PARCIAL ─────────────────────────────────────────────────
    s.append("""
<h2>12. Conclusão Parcial</h2>

<h3>O que está provado</h3>
<ul>
  <li>O <strong>scalar field</strong> é o único discriminador linear efetivo: AUROC per-cloud mediana
      de 0.989 contra ≈ 0.51 para todas as outras features individualmente.</li>
  <li>O <strong>teacher_student_v1</strong> tem a melhor métrica <em>agregada</em>:
      F1=0.942 (F1-ótimo), AUROC=0.952 — mas com std=0.182 por nuvem e F1 mínimo de 0.039.
      Apresenta problemas de localização visual em nuvens com score map globalmente inflado.</li>
  <li>Abordagens não-supervisionadas (SF-GMM, DGCNN, SF MAE) têm precisão muito baixa (0.17–0.36)
      — inadequadas para uso operacional.</li>
  <li>A <strong>surface variation (sv)</strong> contém sinal ortogonal ao SF:
      AUROC residual=0.69 — captura geometria de borda de crack.</li>
  <li>Nenhuma combinação linear de features supera o SF puro — é necessário
      modelo não-linear para aproveitar o sinal residual.</li>
  <li>Três nuvens (avaria_17, 33, 21) são genuinamente difíceis: SF médio de crack ≈ SF médio de normal.</li>
</ul>

<h3>O que falta provar</h3>
<ul>
  <li>Se o XGBoost supervisionado com sv+sat consegue <strong>precisão ≥ 0.85 com recall ≥ 0.85</strong>,
      atingindo ou superando o teacher_student_v1.</li>
  <li>Se o backbone PTv3 já codifica sinal de crack nas suas representações
      (PTv3 Linear Probe).</li>
  <li>Se o fine-tuning supervisionado do PTv3 supera o teacher_student_v1 em AUROC e F1.</li>
  <li>Qual abordagem performa melhor especificamente nas <strong>três nuvens difíceis</strong>
      (avaria_17, 33, 21) — o verdadeiro teste de generalização do método.</li>
</ul>""")

    return s


def main():
    sections = build_sections()
    body = "\n".join(sections)

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Relatório de Progresso TCC — Detecção de Avarias</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">
{body}
</div>
</body>
</html>"""

    out = BASE / "relatorio_progresso_tcc.html"
    out.write_text(html, encoding="utf-8")
    size_kb = out.stat().st_size // 1024
    print(f"Relatório gerado: {out}  ({size_kb} KB)")
    print("Abra no browser → Ctrl+P → Salvar como PDF")


if __name__ == "__main__":
    main()
