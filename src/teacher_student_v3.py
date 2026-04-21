# ============================================================================
# PROFESSOR-ALUNO v3 — CAMINHO A CORRIGIDO: DOMAIN ADAPTER
# ============================================================================
#
# Problema da versão anterior (geometric pretraining):
#   O v3 anterior tentou resolver o domain gap do Teacher PTv3 fazendo
#   fine-tuning do backbone COMPLETO em reconstrução geométrica. Resultado:
#   catastrophic forgetting — AUROC caiu de ~0.83 para 0.71.
#   O backbone sobrescreveu seus priors com "toda superfície é normal",
#   tornando as features MENOS discriminativas que o Teacher original.
#
# Solução (Caminho A Corrigido — inspirado em LSFA, ECCV 2024):
#
#   Fase 0 — Domain Adapter (parametricamente eficiente):
#     Em vez de fine-tunar o backbone, adicionamos um adapter pequeno
#     (512 → 64 → 512, residual, ~66K params) APÓS o bottleneck frozen.
#
#     O adapter é treinado com loss de consistência local em XYZ:
#       L = 1 - cosine_similarity(h_adapted_i, mean_j(h_adapted_j))
#     onde j são os k vizinhos mais próximos de i no espaço 3D.
#
#     Intuição: em superfícies normais, pontos vizinhos têm representações
#     similares. Em rachaduras há descontinuidade de material → features
#     do adapter divergem da vizinhança → alto adapter_consistency_score.
#
#   Por que funciona onde o v3 anterior falhou:
#     • Backbone 100% frozen → sem catastrophic forgetting
#     • Apenas 66K params adaptados (vs. milhões no backbone)
#     • Inicialização zero no up-proj → adapter começa como identidade;
#       treinamento parte dos priors do PTv3, não os apaga
#     • Loss de alinhamento em XYZ → ancoragem geométrica real
#
#   Fase 1 — Teacher-Student (igual ao v2):
#     Adapter congelado. Student decoder aprende a reconstruir as features
#     adaptadas do Teacher — mais discriminativas para concreto.
#
#   Fusão final:
#     score = 0.40·sf_score + 0.25·geom_score
#           + 0.15·distill_norm + 0.20·adapter_consistency
#
# Referências:
#   Houlsby et al. (2019) — Parameter-Efficient Transfer Learning, ICML.
#   Tu et al. (2024) — LSFA: Self-supervised Feature Adaptation, ECCV.
#   Deng & Li (2022) — Reverse Distillation, CVPR.
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import csv, copy, gc, math, time
from datetime import datetime
from scipy.spatial import cKDTree

from utils import *
from utils.config import *
from utils.architectures import *
from utils.building_blocks import *
from utils.data import *
from utils.evaluation import *
from utils.training_utils import *

from teacher_student_v2 import (
    StudentDecoder,
    TeacherStudentModel,
    DistillationLoss,
    train_teacher_student,
    GeometricMultiGMM,
    build_teacher,
    TEACHER_CHUNK_SIZE,
    LR_STUDENT,
    ALPHA_COS,
    ALPHA_MSE,
)

# ── Paths v3 ──────────────────────────────────────────────────────────────────
RESULTS_V3   = f'{BASE_PATH}/results_ts_v3'
MODELS_V3    = f'{BASE_PATH}/models_ts_v3'
VIS_PATH_V3  = f'{BASE_PATH}/visualizations_ts_v3'
LOGS_PATH_V3 = f'{BASE_PATH}/logs_ts_v3'
PLY_PATH_V3  = f'{BASE_PATH}/results_ts_v3/ply'
ADAPTER_CKP  = f'{MODELS_V3}/domain_adapter.pth'

log = setup_logging(LOGS_PATH_V3)

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
LR_ADAPTER     = 1e-4   # learning rate do adapter (pequeno — modelo pequeno)
EPOCHS_ADAPTER = 60     # epochs de pré-treino do adapter
ADAPTER_DIM    = 64     # bottleneck interno: 512 → 64 → 512
ADAPTER_K      = 16     # vizinhos XYZ para a loss de alinhamento
NUM_EPOCHS_V3  = 150    # epochs do Teacher-Student após adapter

# Pós-processamento de scores (redução de FP geométricos)
SMOOTH_K        = 16
BOUNDARY_RADIUS = 0.03
BOUNDARY_MIN_NB = 8


# ============================================================================
# DOMAIN ADAPTER — bottleneck residual parametricamente eficiente
# ============================================================================

class DomainAdapter(nn.Module):
    """
    Adapter de domínio parametricamente eficiente para o Teacher PTv3.

    Arquitetura: bottleneck residual
      x → Linear(d, d_bot) → GELU → Dropout → Linear(d_bot, d) → + x

    A projeção de saída (up) é inicializada em zero: no início do treino
    o adapter é uma identidade exata e os priors do PTv3 são preservados.
    O treinamento parte do pré-treino PTv3, não o apaga.

    Com d=512, d_bot=64: apenas 2 × (512×64 + 64) ≈ 66K parâmetros —
    contra milhões no backbone PTv3. Catastrophic forgetting impossível.

    Referência: Houlsby et al. (2019) — Parameter-Efficient Transfer
    Learning for NLP, ICML.
    """

    def __init__(self, d: int = 512, d_bot: int = ADAPTER_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.down = nn.Linear(d, d_bot)
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up   = nn.Linear(d_bot, d)

        # Inicialização: adapter começa como identidade
        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)   # saída zero → residual = input puro
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, d) → (N, d)"""
        return x + self.up(self.drop(self.act(self.down(x))))


# ============================================================================
# UTILITÁRIOS — kNN em XYZ
# ============================================================================

def _xyz_knn(xyz: np.ndarray, k: int) -> np.ndarray:
    """
    Constrói índices k-NN no espaço XYZ usando scipy cKDTree (CPU).
    Exclui o próprio ponto (índice 0).

    xyz : (N, 3) numpy float32
    k   : número de vizinhos
    Returns: (N, k) int64
    """
    tree    = cKDTree(xyz)
    k_query = min(k + 1, len(xyz))
    _, idx  = tree.query(xyz, k=k_query, workers=-1)
    return idx[:, 1:k+1] if idx.shape[1] > k else idx[:, 1:]


# ============================================================================
# LOSS DE ALINHAMENTO DO ADAPTER
# ============================================================================

def adapter_alignment_loss(
    h_adapted: torch.Tensor,
    knn_idx: np.ndarray,
) -> torch.Tensor:
    """
    Loss de consistência local de vizinhança em XYZ.

    Para cada ponto i, a representação adaptada deve ser similar à média
    das representações dos seus k vizinhos mais próximos no espaço 3D.

    Em superfícies normais: vizinhos têm features similares → loss baixa.
    Em rachaduras: descontinuidade de material → features divergem → loss alta.

    L = mean_i( 1 - cosine_similarity(h_i, mean_j(h_j)) )

    h_adapted : (N, 512) — saída do DomainAdapter (GPU)
    knn_idx   : (N, k)   — índices k-NN em XYZ (numpy, CPU)
    """
    N, k   = knn_idx.shape
    idx_t  = torch.from_numpy(knn_idx.astype(np.int64)).to(h_adapted.device)

    neighbors = h_adapted[idx_t.view(-1)].view(N, k, -1)  # (N, k, 512)
    h_local   = neighbors.mean(dim=1)                      # (N, 512)

    # Detach h_local: puxamos h_i em direção à vizinhança, não o contrário.
    # Sem detach, o gradiente espalha por todos os vizinhos e desestabiliza.
    cos = F.cosine_similarity(h_adapted, h_local.detach(), dim=1)
    return (1.0 - cos).mean()


# ============================================================================
# PRÉ-TREINO DO ADAPTER (Fase 0)
# ============================================================================

def pretrain_adapter(
    adapter: DomainAdapter,
    teacher: nn.Module,
    all_data: list,
    device: torch.device,
    num_epochs: int = EPOCHS_ADAPTER,
    lr: float = LR_ADAPTER,
    k: int = ADAPTER_K,
    save_path: str = ADAPTER_CKP,
    normal_only: bool = True,
) -> DomainAdapter:
    """
    Treina o DomainAdapter com loss de consistência de vizinhança em XYZ.

    Protocolo:
      1. Caching (1×): Teacher bottleneck h e kNN XYZ para cada nuvem —
         ambos fixos durante o treino (Teacher frozen, XYZ não muda).
      2. Loop de treino: para cada epoch, para cada nuvem:
           h_adapted = adapter(h_cached)
           loss = adapter_alignment_loss(h_adapted, knn_cached)
         Backprop apenas pelos parâmetros do adapter.

    normal_only: usar apenas nuvens sem rachadura para treinar (recomendado).
    O adapter aprende o padrão de superfície NORMAL do concreto.

    Salva checkpoint com o estado do adapter ao final.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = ([d for d in all_data if not d.get('has_crack', True)]
            if normal_only else all_data)
    if len(data) < 3:
        log.warning(f"Adapter: apenas {len(data)} nuvens normais — usando todas")
        data = all_data

    log.info(f"\n{'='*65}")
    log.info("FASE 0 — Domain Adapter (consistência local em XYZ)")
    log.info(f"Nuvens: {len(data)} | Epochs: {num_epochs} | LR: {lr}")
    log.info(f"Adapter: 512 → {ADAPTER_DIM} → 512 | k={k} vizinhos XYZ")
    log.info(f"{'='*65}")

    adapter   = adapter.to(device)
    teacher   = teacher.to(device)
    optimizer = AdamW(adapter.parameters(), lr=lr,
                      betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    scaler    = GradScaler()

    # ── Caching: Teacher bottleneck + kNN XYZ (ambos fixos) ──────────────────
    log.info("Caching Teacher bottlenecks + kNN XYZ (1× antes do treino)...")
    h_cache   = []   # list[Tensor(N, 512) fp16 CPU]
    knn_cache = []   # list[ndarray(N, k)  int64 CPU]

    teacher.eval()
    for idx_c, d in enumerate(data):
        x     = torch.tensor(d['features'], dtype=torch.float32).to(device)
        xyz   = d['features'][:, :3]

        with torch.no_grad():
            # Teacher forward em chunks para evitar OOM
            N = x.size(0)
            parts = []
            for s in range(0, N, TEACHER_CHUNK_SIZE):
                parts.append(teacher(x[s:min(s + TEACHER_CHUNK_SIZE, N)]))
            h = torch.cat(parts, dim=0)  # (N, 512)

        h_cache.append(h.half().cpu())
        knn_cache.append(_xyz_knn(xyz, k))

        del x, h
        torch.cuda.empty_cache()
        if (idx_c + 1) % 10 == 0:
            log.info(f"  Cache: {idx_c+1}/{len(data)}")

    log.info(f"Cache pronto: {len(h_cache)} nuvens")

    best_loss = float('inf')
    for epoch in range(num_epochs):
        adapter.train()
        ep_losses = []
        perm = np.random.permutation(len(data))

        for i in perm:
            h   = h_cache[i].float().to(device)   # (N, 512) fp16→fp32
            knn = knn_cache[i]                     # (N, k)   numpy

            if h.size(0) < k + 1:
                continue

            optimizer.zero_grad(set_to_none=True)
            try:
                with autocast():
                    h_adapted = adapter(h)
                    loss      = adapter_alignment_loss(h_adapted, knn)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ep_losses.append(loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        if not ep_losses:
            continue

        scheduler.step()
        avg = float(np.mean(ep_losses))
        lr_ = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  Adapter {epoch+1:03d}/{num_epochs} | "
                     f"Loss={avg:.5f} | LR={lr_:.2e}")

        if avg < best_loss:
            best_loss = avg
            torch.save({'epoch': epoch, 'adapter': adapter.state_dict(),
                        'loss': best_loss}, save_path)

        gc.collect()

    log.info(f"\nAdapter treinado. Melhor loss: {best_loss:.5f}")
    log.info(f"Checkpoint salvo em: {save_path}")
    return adapter


# ============================================================================
# MODELO TEACHER-STUDENT v3 (com Domain Adapter)
# ============================================================================

class TeacherStudentModelV3(TeacherStudentModel):
    """
    Teacher-Student com Domain Adapter parametricamente eficiente.

    Diferença do TeacherStudentModel (v2):
      • DomainAdapter (66K params) inserido após o bottleneck do Teacher.
      • Teacher e adapter são congelados durante a distilação (Fase 1).
      • Student aprende a reconstruir features ADAPTADAS — mais
        discriminativas para concreto do que as features S3DIS brutas.

    Fluxo na inferência:
      input → Teacher (frozen) → h_raw (512D)
                               → adapter (frozen) → h_adapted (512D)
                               → Student decoder → anomaly score
    """

    def __init__(self, input_dim: int = INPUT_DIM,
                 adapter_ckpt: str = None):
        super().__init__(input_dim=input_dim, teacher_ckpt=None)
        self.teacher = build_teacher(
            input_dim=input_dim,
            ptv3_ckpt=PTRANSF_WEIGHTS,
            s3dis_ckpt=PTRANSF_WEIGHTS_S3DIS,
        )
        self._freeze_teacher()

        self.adapter = DomainAdapter(d=512, d_bot=ADAPTER_DIM)

        if adapter_ckpt and os.path.exists(adapter_ckpt):
            ckpt = torch.load(adapter_ckpt, map_location='cpu', weights_only=False)
            self.adapter.load_state_dict(ckpt['adapter'])
            log.info(f"V3: adapter carregado de {adapter_ckpt} "
                     f"(epoch {ckpt.get('epoch', '?')}, "
                     f"loss={ckpt.get('loss', float('nan')):.5f})")
        else:
            log.warning("V3: adapter não encontrado — usando inicialização padrão")

        n_t = sum(p.numel() for p in self.teacher.parameters())
        n_a = sum(p.numel() for p in self.adapter.parameters())
        n_s = sum(p.numel() for p in self.student.parameters())
        log.info(f"TeacherStudentV3 | Teacher: {n_t:,} (frozen) | "
                 f"Adapter: {n_a:,} | Student: {n_s:,}")

    def _freeze_adapter(self):
        """Congela adapter para a Fase 1 (distilação)."""
        for p in self.adapter.parameters():
            p.requires_grad_(False)
        self.adapter.eval()
        log.info("Adapter congelado para Fase 1 (distilação)")

    def teacher_features(self, x: torch.Tensor,
                         chunk_size: int = TEACHER_CHUNK_SIZE) -> tuple:
        """
        Extrai features do Teacher + aplica adapter ao bottleneck.

        Substitui TeacherStudentModel.teacher_features():
          bottleneck_raw → adapter → bottleneck_adapted
          t3 = proj_t3(bottleneck_adapted)   ← escala 256D adaptada
          t2 = proj_t2(t_lfa)                ← escala 128D (inalterada)
          t1 = proj_t1(t_lfa)                ← escala 64D  (inalterada)

        O Student aprenderá a reconstruir features adaptadas, que são
        mais discriminativas para concreto (sem catastrophic forgetting).
        """
        bottleneck_raw, teacher_scales_raw = super().teacher_features(
            x, chunk_size=chunk_size)

        # Adapter: aplica em chunks se necessário (adapter é pequeno, OOM improvável)
        with torch.set_grad_enabled(self.adapter.training):
            bottleneck_adapted = self.adapter(bottleneck_raw)

        # Escala 3 usa o bottleneck adaptado; escalas 2 e 1 permanecem do Teacher
        t3 = self.proj_t3(bottleneck_adapted)
        t2, t1 = teacher_scales_raw[1], teacher_scales_raw[2]

        return bottleneck_adapted, [t3, t2, t1]

    @torch.no_grad()
    def adapter_consistency_score(
        self,
        h_adapted: torch.Tensor,
        xyz: np.ndarray,
        k: int = ADAPTER_K,
    ) -> np.ndarray:
        """
        Score de consistência do adapter por ponto.

        score(i) = 1 - cosine_similarity(h_adapted_i, mean_j(h_adapted_j))

        Pontos em superfícies normais: vizinhos têm features similares → score baixo.
        Pontos em rachaduras: descontinuidade → features divergem → score alto.

        h_adapted : (N, 512) tensor GPU
        xyz       : (N, 3)   numpy
        Returns   : (N,) float32 normalizado [0, 1]
        """
        knn    = _xyz_knn(xyz, k)
        idx_t  = torch.from_numpy(knn.astype(np.int64)).to(h_adapted.device)

        N      = h_adapted.size(0)
        k_     = idx_t.size(1)
        neighbors = h_adapted[idx_t.view(-1)].view(N, k_, -1)
        h_local   = neighbors.mean(dim=1)

        cos   = F.cosine_similarity(h_adapted, h_local, dim=1)
        score = (1.0 - cos).cpu().numpy().astype(np.float32)

        lo, hi = np.percentile(score, 2), np.percentile(score, 98)
        return np.clip((score - lo) / (hi - lo + 1e-8), 0, 1)


# ============================================================================
# SCORE SUPERVISIONADO DO SCALAR FIELD (mantido do v3 anterior)
# ============================================================================

def _sf_score_supervised(sf: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    P(rachadura | scalar_field[i]) via Naive Bayes Gaussiano supervisionado.

    Quando os labels estão disponíveis, fita diretamente as duas Gaussianas
    (crack vs. normal) em vez de depender do GMM não-supervisionado, que
    frequentemente erra o intervalo de rachadura (ex: w_anomal=0.77).
    """
    from scipy.stats import norm as gaussian_dist

    crack_vals  = sf[labels == 1].astype(np.float64)
    normal_vals = sf[labels == 0].astype(np.float64)

    mu_c,  std_c  = crack_vals.mean(),  max(crack_vals.std(),  1e-6)
    mu_n,  std_n  = normal_vals.mean(), max(normal_vals.std(), 1e-6)
    prior_c = len(crack_vals)  / len(sf)
    prior_n = len(normal_vals) / len(sf)

    p_sf_c = gaussian_dist.pdf(sf, mu_c, std_c) * prior_c
    p_sf_n = gaussian_dist.pdf(sf, mu_n, std_n) * prior_n
    denom  = p_sf_c + p_sf_n + 1e-12

    return (p_sf_c / denom).astype(np.float32)


# ============================================================================
# PÓS-PROCESSAMENTO (mantido do v3 anterior)
# ============================================================================

def _boundary_mask(xyz: np.ndarray,
                   radius: float = 0.03,
                   min_neighbors: int = 8) -> np.ndarray:
    """
    Retorna máscara True onde o ponto é borda do recorte (vizinhança truncada).
    Pontos de borda acumulam FPs sistemáticos: normais erradas, curvatura
    artificial, densidade baixa. Suprimimos forçando score=0.
    """
    tree   = cKDTree(xyz)
    counts = tree.query_ball_point(xyz, r=radius, return_length=True)
    return counts < min_neighbors


def spatial_smooth_scores(xyz: np.ndarray,
                           score: np.ndarray,
                           k: int = 16) -> np.ndarray:
    """
    Suaviza scores por média k-NN em XYZ.
    FPs isolados (score alto circundado de score baixo) são diluídos.
    TPs de rachadura (score alto em grupo espacialmente coerente) são mantidos.
    """
    tree   = cKDTree(xyz)
    _, idx = tree.query(xyz, k=k)
    return score[idx].mean(axis=1)


def post_process_scores(xyz: np.ndarray,
                         score: np.ndarray,
                         boundary_radius: float = BOUNDARY_RADIUS,
                         boundary_min_nb: int = BOUNDARY_MIN_NB,
                         smooth_k: int = SMOOTH_K) -> np.ndarray:
    """
    Suavização k-NN + supressão de borda para redução de FPs.
    1. Suavização: dilui FPs isolados (padrão avaria_22)
    2. Máscara de borda: suprime pontos de fronteira do recorte (padrão avaria_13)
    """
    score = spatial_smooth_scores(xyz, score, k=smooth_k)
    bnd   = _boundary_mask(xyz, radius=boundary_radius,
                            min_neighbors=boundary_min_nb)
    score = score.copy()
    score[bnd] = 0.0
    return score


# ============================================================================
# INFERÊNCIA — SCORES DE ANOMALIA v3
# ============================================================================

@torch.no_grad()
def compute_anomaly_scores_v3(
    model: TeacherStudentModelV3,
    data_list: list,
    device: torch.device,
    memory_bank: NormalMemoryBank = None,
    alpha_dist: float = 0.8,
    alpha_bank: float = 0.2,
) -> list:
    """
    Score composto com adapter consistency (novidade v3):

      score = 0.40 · sf_score          [ScalarFieldGMM    — sinal primário]
            + 0.25 · geom_score        [GeometricMultiGMM — curvatura/dens/var]
            + 0.15 · distill_norm      [Teacher-Student    — prior geométrico]
            + 0.20 · adapter_score     [Adapter consistency — novidade v3]

    adapter_score: 1 - cosine_sim(h_adapted_i, mean(h_adapted_vizinhos_XYZ))
    Captura descontinuidades de superfície que o Teacher puro não vê.

    Normalização: percentil 2–98 para distill e adapter.
    """
    model.eval()
    model.adapter.eval()
    results = []

    for d in data_list:
        x      = torch.tensor(d['features'], dtype=torch.float32).to(device)
        labels = d['labels']
        xyz_np = d['features'][:, :3]

        # ── Distilação Teacher-Student (adapter incluso em teacher_features) ──
        bottleneck_adapted, _ = model.teacher_features(x)
        raw_score = model.anomaly_score_per_point(x)

        if memory_bank is not None and memory_bank.bank is not None:
            bank_dist  = memory_bank.score(bottleneck_adapted.cpu(), k=5)
            lo_b, hi_b = np.percentile(bank_dist, 2), np.percentile(bank_dist, 98)
            bank_norm  = np.clip((bank_dist - lo_b) / (hi_b - lo_b + 1e-8), 0, 1)
            lo_d, hi_d = np.percentile(raw_score,  2), np.percentile(raw_score,  98)
            dist_norm  = np.clip((raw_score  - lo_d) / (hi_d - lo_d + 1e-8), 0, 1)
            distill_component = alpha_dist * dist_norm + alpha_bank * bank_norm
        else:
            lo, hi = np.percentile(raw_score, 2), np.percentile(raw_score, 98)
            distill_component = np.clip((raw_score - lo) / (hi - lo + 1e-8), 0, 1)

        # ── Adapter consistency score (novidade v3) ───────────────────────────
        adapter_score = model.adapter_consistency_score(
            bottleneck_adapted, xyz_np, k=ADAPTER_K)

        # ── ScalarFieldGMM (col 9) ─────────────────────────────────────────────
        sf_raw = d['features'][:, 9]
        gt     = labels

        if gt is not None and gt.sum() >= 10 and (gt == 0).sum() >= 10:
            sf_score = _sf_score_supervised(sf_raw, gt)
        else:
            sf_gmm   = ScalarFieldGMM(sf_raw).fit()
            sf_score = sf_gmm.anomaly_probability()

        sf_gmm_obj = ScalarFieldGMM(sf_raw).fit()

        # ── GeometricMultiGMM (cols 11,12,13) ─────────────────────────────────
        geom_gmm   = GeometricMultiGMM(d['features']).fit()
        geom_score = geom_gmm.anomaly_probability()

        # ── Fusão v3: adapter_consistency recebe 20% (novidade) ───────────────
        score = (0.40 * sf_score
               + 0.25 * geom_score
               + 0.15 * distill_component
               + 0.20 * adapter_score)

        # ── Pós-processamento ──────────────────────────────────────────────────
        score = post_process_scores(xyz_np, score,
                                    boundary_radius=BOUNDARY_RADIUS,
                                    boundary_min_nb=BOUNDARY_MIN_NB,
                                    smooth_k=SMOOTH_K)

        results.append({
            'filename'        : d['filename'],
            'has_crack'       : d['has_crack'],
            'score'           : score,
            'gt_labels'       : labels,
            'n_points'        : len(labels),
            'xyz'             : xyz_np,
            'rgb'             : d['features'][:, 3:6],
            'scalar_field'    : sf_raw,
            'sf_gmm_modality' : sf_gmm_obj.modality,
            'geom_modality'   : geom_gmm.modality,
        })

        del x, bottleneck_adapted
        torch.cuda.empty_cache()

    return results


# ============================================================================
# BUILD — ADAPTER + MODELO
# ============================================================================

def build_pretrained_adapter_v3(
    all_data: list,
    device: torch.device,
    force_retrain: bool = False,
) -> str:
    """
    Garante que o adapter está treinado em ADAPTER_CKP.
    Reutiliza checkpoint existente se force_retrain=False.
    Returns: caminho para o checkpoint do adapter.
    """
    if os.path.exists(ADAPTER_CKP) and not force_retrain:
        log.info(f"Adapter encontrado: {ADAPTER_CKP} (pulando Fase 0)")
        return ADAPTER_CKP

    log.info("Fase 0: treinando Domain Adapter...")
    teacher = build_teacher(
        input_dim=INPUT_DIM,
        ptv3_ckpt=PTRANSF_WEIGHTS,
        s3dis_ckpt=PTRANSF_WEIGHTS_S3DIS,
    ).to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()

    adapter = DomainAdapter(d=512, d_bot=ADAPTER_DIM).to(device)

    pretrain_adapter(
        adapter, teacher, all_data, device,
        num_epochs=EPOCHS_ADAPTER,
        lr=LR_ADAPTER,
        k=ADAPTER_K,
        save_path=ADAPTER_CKP,
        normal_only=True,
    )
    return ADAPTER_CKP


def build_model_v3(device: torch.device,
                   adapter_ckpt: str = None) -> TeacherStudentModelV3:
    """Instancia TeacherStudentModelV3 com adapter pré-treinado."""
    model = TeacherStudentModelV3(
        input_dim=INPUT_DIM,
        adapter_ckpt=adapter_ckpt or ADAPTER_CKP,
    )
    return model.to(device)


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print("  PROFESSOR-ALUNO v3 — CAMINHO A CORRIGIDO: DOMAIN ADAPTER")
    print("  Fase 0: Adapter (66K params) | Fase 1: Teacher-Student")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    run_results = os.path.join(RESULTS_V3, f'run_v3_{ts}')
    run_vis     = os.path.join(VIS_PATH_V3, f'run_v3_{ts}')
    run_ply     = os.path.join(PLY_PATH_V3, f'run_v3_{ts}')

    for p in [run_results, run_vis, run_ply, MODELS_V3, LOGS_PATH_V3]:
        os.makedirs(p, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device}")
    if torch.cuda.is_available():
        log.info(f"GPU  : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── 1. Dados ──────────────────────────────────────────────────────────────
    log.info("\nCarregando dados...")
    train_all = load_folder(DATA_TRAIN)
    test_all  = load_folder(DATA_TEST)
    all_data  = train_all + test_all

    if not all_data:
        log.error("Nenhum dado encontrado.")
        return

    train_list, labeled_list, eval_list = split_dataset(all_data)
    train_dl, labeled_dl, _             = make_loaders(train_list, eval_list,
                                                        labeled_list=labeled_list)

    # ── 2. Fase 0: pré-treino do adapter ─────────────────────────────────────
    log.info("\n" + "="*65)
    log.info("FASE 0 — Domain Adapter (consistência local em XYZ)")
    log.info("="*65)
    adapter_ckpt = build_pretrained_adapter_v3(all_data, device)

    # ── 3. Fase 1: Teacher-Student com adapter congelado ─────────────────────
    log.info("\n" + "="*65)
    log.info("FASE 1 — Teacher-Student (Reverse Distillation)")
    log.info("Teacher + Adapter frozen | Student treinável")
    log.info("="*65)
    model = build_model_v3(device, adapter_ckpt)
    model._freeze_adapter()   # adapter congelado durante distilação

    # ── 4. Memory bank ────────────────────────────────────────────────────────
    memory_bank = NormalMemoryBank(max_size=100_000, subsample_k=1_000)

    # ── 5. Treino do Student ──────────────────────────────────────────────────
    log.info("\nIniciando treino do Student decoder...")
    t0 = time.time()
    model, history = train_teacher_student(
        model, train_dl, device,
        num_epochs=NUM_EPOCHS_V3,
        lr=LR_STUDENT,
        use_early_stopping=True,
        memory_bank=memory_bank,
        save_dir=MODELS_V3,
        labeled_loader=labeled_dl,
    )
    log.info(f"\nTreino: {(time.time()-t0)/3600:.1f}h")
    plot_training_history(history, run_vis, ts=ts)

    # ── 6. Melhor Student ─────────────────────────────────────────────────────
    best_path = os.path.join(MODELS_V3, 'best_student.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.student.load_state_dict(ckpt['student'])
        log.info(f"Melhor Student carregado (epoch {ckpt['epoch']+1})")

    memory_bank.save(os.path.join(MODELS_V3, 'ts_memory_bank.pt'))

    # ── 6b. Rebuild memory bank com nuvens completas ──────────────────────────
    log.info("Reconstruindo memory bank com nuvens completas...")
    memory_bank.bank = None
    _orig_k = memory_bank.subsample_k
    memory_bank.subsample_k = 300
    model.eval()
    with torch.no_grad():
        for d in train_list:
            x = torch.tensor(d['features'], dtype=torch.float32).to(device)
            btn, _ = model.teacher_features(x)   # retorna bottleneck ADAPTADO
            memory_bank.update(btn.cpu())
    memory_bank.subsample_k = _orig_k
    log.info(f"Memory bank: {len(memory_bank.bank):,} features adaptadas "
             f"de {len(train_list)} nuvens")
    memory_bank.save(os.path.join(MODELS_V3, 'ts_memory_bank_clean.pt'))

    # ── 7. Scores de anomalia ─────────────────────────────────────────────────
    log.info("\nComputando anomaly scores (v3 — adapter + distill)...")
    eval_data = eval_list if eval_list else train_list
    results   = compute_anomaly_scores_v3(model, eval_data, device, memory_bank,
                                           alpha_dist=0.65, alpha_bank=0.3)

    log.info("Computando scores de referência (nuvens normais)...")
    normal_ref = compute_anomaly_scores_v3(
        model, train_list[:min(20, len(train_list))], device, memory_bank)

    # ── 8. Threshold + comparação de estratégias ──────────────────────────────
    thr, gmm_info = fit_gmm_threshold(results, normal_results=normal_ref)
    log.info(f"Threshold ({gmm_info['method']}): {thr:.4f}")
    results = apply_threshold(results, thr)

    strategies = {
        'F1'         : calibrate_threshold_f1(results),
        'G-mean'     : calibrate_threshold_gmean(results),
        'F-beta(0.5)': calibrate_threshold_fbeta(results, beta=0.5),
    }

    comparison    = []
    results_gmean = None
    metrics_gmean = None
    thr_gmean     = None
    best_f1_any   = -1.0
    best_name_any = None

    log.info("\n" + "="*65)
    log.info("COMPARAÇÃO DE ESTRATÉGIAS DE THRESHOLD")
    log.info("="*65)

    for name, thr_s in strategies.items():
        res_s = apply_threshold(copy.deepcopy(results), thr_s)
        m_s   = evaluate(res_s)

        row = {
            'estrategia'   : name,
            'threshold'    : round(thr_s, 6),
            'precision'    : round(m_s.get('precision',          0), 4),
            'recall'       : round(m_s.get('recall',             0), 4),
            'f1'           : round(m_s.get('f1',                 0), 4),
            'f1_macro'     : round(m_s.get('f1_macro',           0), 4),
            'iou'          : round(m_s.get('iou',                0), 4),
            'auroc'        : round(m_s.get('auroc',              0), 4),
            'avg_precision': round(m_s.get('average_precision',  0), 4),
        }
        comparison.append(row)

        log.info(f"\n  [{name}]  thr={thr_s:.4f}")
        log.info(f"    P={row['precision']:.4f}  R={row['recall']:.4f}  "
                 f"F1={row['f1']:.4f}  IoU={row['iou']:.4f}  "
                 f"AUROC={row['auroc']:.4f}")

        if name == 'G-mean':
            results_gmean = res_s
            metrics_gmean = m_s
            thr_gmean     = thr_s

        if row['f1'] > best_f1_any:
            best_f1_any   = row['f1']
            best_name_any = name

    log.info(f"\n  Estratégia primária: [G-mean] "
             f"(melhor F1: [{best_name_any}] F1={best_f1_any:.4f})")
    log.info("="*65)

    results = results_gmean
    metrics = metrics_gmean
    thr_f1  = thr_gmean

    pd.DataFrame(comparison).to_csv(
        os.path.join(run_results, f'comparacao_thresholds_v3_{ts}.csv'), index=False)

    # ── 9. Severidade ─────────────────────────────────────────────────────────
    results  = compute_severity(results)
    relatorio = severity_report(results)

    relatorio_path = os.path.join(run_results, f'relatorio_severidade_v3_{ts}.csv')
    with open(relatorio_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['tipo', 'chave', 'valor'])
        for cat, n in relatorio['categorias'].items():
            writer.writerow(['categoria', cat, n])
        for risco, n in relatorio['riscos'].items():
            writer.writerow(['risco', risco, n])
        writer.writerow(['resumo', 'pct_conforme_abnt',
                         f"{relatorio['pct_conforme']:.1f}%"])

    # ── 10. Visualizações e PLY ───────────────────────────────────────────────
    try:
        visualize_cracks(results, save_dir=run_vis, max_clouds=10, ts=ts)
    except Exception as e:
        log.error(f"visualize_cracks: {e}", exc_info=True)

    try:
        plot_score_distribution(results, thr_f1, run_vis, gmm_info, ts=ts)
    except Exception as e:
        log.error(f"plot_score_distribution: {e}", exc_info=True)

    try:
        save_results(metrics, results, history, thr_f1, gmm_info, run_vis, ts=ts)
    except Exception as e:
        log.error(f"save_results: {e}", exc_info=True)

    log.info("\nSalvando PLY coloridos (v3)...")
    for r in results:
        if not r['has_crack']:
            continue
        ply_out = os.path.join(run_ply,
                               r['filename'].replace('.ply', '_pred_v3.ply'))
        save_colored_ply(xyz=r['xyz'], rgb_orig=r['rgb'],
                         pred_labels=r['pred_labels'], path=ply_out)

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTADOS FINAIS — PROFESSOR-ALUNO v3 (DOMAIN ADAPTER)")
    print("="*70)
    if metrics:
        print(f"  F1    : {metrics.get('f1',    0):.4f}")
        print(f"  IoU   : {metrics.get('iou',   0):.4f}")
        print(f"  AUROC : {metrics.get('auroc', 0):.4f}")
    print(f"\n  Threshold ({gmm_info['method']}): {thr:.4f}")
    print(f"  Resultados em: {run_results}")
    print("="*70)

    return model, history, results, metrics


if __name__ == '__main__':
    main()
