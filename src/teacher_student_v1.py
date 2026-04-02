# ============================================================================
# PROFESSOR-ALUNO v1 — REVERSE DISTILLATION
# ============================================================================
# Teacher : PointTransformerInspiredAdvanced (pré-treinado S3DIS, congelado)
# Student : StudentDecoder (decoder treinável, Reverse Distillation)
# Score   : distância coseno Teacher-Student em 3 escalas + memory bank
# Referências:
#   Deng & Li (CVPR 2022) — Reverse Distillation from One-Class Embedding
#   Wang et al. (WACV 2023) — Asymmetric Student-Teacher Networks
#   Zhou et al. (2024)      — R3D-AD
#   Roth et al. (2022)      — PatchCore
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
# ── Caminhos e hiperparâmetros específicos do Professor-Aluno ──────────────────
RESULTS    = f'{BASE_PATH}/results_ts'
MODELS     = f'{BASE_PATH}/models_ts'
VIS_PATH   = f'{BASE_PATH}/visualizations_ts'
LOGS_PATH  = f'{BASE_PATH}/logs_ts'

log = setup_logging(LOGS_PATH)

LR_STUDENT    = 1e-4     # learning rate do Student decoder
NUM_EPOCHS    = 150      # epochs totais (early stopping costuma parar antes)
ALPHA_COS     = 0.85     # cosine é o sinal primário do anomaly score — aumentar peso
ALPHA_MSE     = 0.15     # magnitude normalizada é secundária
TEMP_DISTILL  = 4.0      # temperatura de distilação [Hinton et al., 2015]
# ─────────────────────────────────────────────────────────────────────────────


# ============================================================================
# STUDENT DECODER — Reverse Distillation
# ============================================================================

class StudentDecoder(nn.Module):
    """
    Decoder que inverte o bottleneck do Teacher para reconstruir suas
    features intermediárias em 3 escalas.

    Input : bottleneck (N, 512) — saída do Teacher frozen
    Output: [s1 (N, 256), s2 (N, 128), s3 (N, 64)]
             correspondendo às escalas [l4→l3, l3→l2, l2→l1] do Teacher

    Cada escala usa um bloco MLP com BatchNorm + skip connection residual,
    seguido de um attention gate que modula a contribuição de cada feature.

    Reverse Distillation (Deng & Li, CVPR 2022):
      "The student decoder learns to reconstruct teacher features in reverse
       order, from the most abstract representation to the most local one."
    """

    def __init__(self, d_bottleneck: int = 512):
        super().__init__()

        # ── Escala 3: bottleneck → 256D (mais abstrato) ───────────────────────
        self.up3 = nn.Sequential(
            nn.Linear(d_bottleneck, 384), nn.BatchNorm1d(384), nn.GELU(),
            nn.Linear(384, 256),          nn.BatchNorm1d(256), nn.GELU(),
        )
        self.res3 = nn.Linear(d_bottleneck, 256)   # skip para estabilizar
        self.gate3 = nn.Sequential(
            nn.Linear(256, 256), nn.Sigmoid()
        )

        # ── Escala 2: 256D → 128D (escala média) ─────────────────────────────
        self.up2 = nn.Sequential(
            nn.Linear(256, 192), nn.BatchNorm1d(192), nn.GELU(),
            nn.Linear(192, 128), nn.BatchNorm1d(128), nn.GELU(),
        )
        self.res2 = nn.Linear(256, 128)
        self.gate2 = nn.Sequential(
            nn.Linear(128, 128), nn.Sigmoid()
        )

        # ── Escala 1: 128D → 64D (mais local) ────────────────────────────────
        self.up1 = nn.Sequential(
            nn.Linear(128, 96), nn.BatchNorm1d(96), nn.GELU(),
            nn.Linear(96, 64),  nn.BatchNorm1d(64), nn.GELU(),
        )
        self.res1 = nn.Linear(128, 64)
        self.gate1 = nn.Sequential(
            nn.Linear(64, 64), nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bottleneck: torch.Tensor) -> list:
        """
        bottleneck: (N, 512)
        Returns: [s3 (N,256), s2 (N,128), s1 (N,64)]
        """
        s3 = self.up3(bottleneck) + self.res3(bottleneck)
        s3 = s3 * self.gate3(s3)

        s2 = self.up2(s3) + self.res2(s3)
        s2 = s2 * self.gate2(s2)

        s1 = self.up1(s2) + self.res1(s2)
        s1 = s1 * self.gate1(s1)

        return [s3, s2, s1]   # ordem inversa à do Teacher: [256, 128, 64]


# ============================================================================
# MODELO PROFESSOR-ALUNO COMPLETO
# ============================================================================

class TeacherStudentModel(nn.Module):
    """
    Encapsula Teacher (frozen) + Student (trainável) em um único módulo.

    Teacher  : PointTransformerInspiredAdvanced pré-treinado em S3DIS
               → extrai features em 3 escalas + bottleneck
               → parâmetros NUNCA atualizados

    Student  : StudentDecoder
               → recebe bottleneck do Teacher
               → reconstrói features nas 3 escalas

    Anomaly score por ponto (inferência):
      score(p) = mean_scales( 1 - cosine_similarity(T_feat(p), S_feat(p)) )

    As 3 escalas têm pesos diferentes: escala mais local (64D) recebe
    peso maior pois rachaduras são fenômenos locais finos.
    Pesos: [0.2, 0.3, 0.5] para [256D, 128D, 64D].
    """

    # Pesos das escalas para o anomaly score (soma = 1.0)
    # [256D, 128D, 64D] — pesos equilibrados reduzem falsos positivos de ruído
    # de superfície que afetam excessivamente a escala fina (64D)
    SCALE_WEIGHTS = [0.2, 0.3, 0.5]

    def __init__(self, input_dim: int = INPUT_DIM,
                 teacher_ckpt: str = None):
        super().__init__()

        # ── Teacher: PointTransformer congelado ───────────────────────────────
        self.teacher = PointTransformerInspiredAdvanced(
            input_dim=input_dim, checkpoint_path=teacher_ckpt)
        self._freeze_teacher()

        # ── Student: decoder treinável ────────────────────────────────────────
        self.student = StudentDecoder(d_bottleneck=512)

        # ── Projeções treinadas para escalas intermediárias do Teacher ─────────
        # Em vez de slicing arbitrário (bottleneck[:,:256], lfa[:,:64]),
        # projeções lineares aprendem quais dimensões são relevantes por escala.
        self.proj_t3 = nn.Linear(512, 256)   # bottleneck   → escala 256D
        self.proj_t2 = nn.Linear(128, 128)   # t_lfa        → escala 128D
        self.proj_t1 = nn.Linear(128, 64)    # t_lfa        → escala  64D

        n_t = sum(p.numel() for p in self.teacher.parameters())
        n_s = sum(p.numel() for p in self.student.parameters())
        log.info(f"TeacherStudent | Teacher: {n_t:,} params (frozen) | "
                 f"Student: {n_s:,} params (treinável)")

    def _freeze_teacher(self):
        """Teacher sempre congelado — nunca recebe gradientes."""
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.teacher.eval()   # BN e Dropout em modo eval permanentemente

    def teacher_features(self, x: torch.Tensor) -> tuple:
        """
        Extrai features do Teacher SEM gradientes.
        x: (N, input_dim)
        Returns: (bottleneck (N,512), [t3 (N,256), t2 (N,128), t1 (N,64)])

        Nota: PointTransformerInspiredAdvanced retorna (N,512) diretamente.
        As escalas intermediárias são extraídas com hooks temporários.
        """
        self.teacher.eval()

        # Coletar ativações intermediárias com forward hooks
        feats = {}

        def _hook_adapter(m, inp, out):
            feats['adapter'] = out          # (N, 128) — pós feature_adapter

        def _hook_lfa(m, inp, out):
            feats['lfa'] = out              # (N, 128) — pós LFA

        def _hook_block0(m, inp, out):
            feats['block0'] = out           # (N, 128) — saída do 1º blk

        hooks = [
            self.teacher.feature_adapter.register_forward_hook(_hook_adapter),
            self.teacher.lfa.register_forward_hook(_hook_lfa),
        ]
        if len(self.teacher.blocks) > 0:
            hooks.append(
                self.teacher.blocks[0].register_forward_hook(_hook_block0))

        with torch.no_grad():
            bottleneck = self.teacher(x)    # (N, 512) — pós proj

        for h in hooks:
            h.remove()

        # Escalas intermediárias disponíveis
        # Mapear para os 3 tamanhos esperados pelo Student: 256, 128, 64
        # A arquitetura PointTransformer opera em 128D internamente.
        # Projetamos para as dimensões do Student com projeções lineares leves.
        t_adapter = feats.get('adapter', None)   # (N, 128)
        t_lfa     = feats.get('lfa',     None)   # (N, 128)
        t_block0  = feats.get('block0',  None)   # (N, 128)

        # Projeções fixas (não treinadas — mantidas no eval) para alinhar dims
        # São aplicadas apenas na inferência das features intermediárias
        teacher_scales = self._project_teacher_scales(
            bottleneck, t_adapter, t_lfa, t_block0)

        return bottleneck, teacher_scales

    def _project_teacher_scales(self, bottleneck, t_adapter, t_lfa, t_block0):
        """
        Projeta features intermediárias do Teacher para as 3 escalas [256, 128, 64].
        Usa projeções lineares treinadas (proj_t3, proj_t2, proj_t1) em vez de
        slicing arbitrário — as projeções aprendem quais dimensões são relevantes
        para cada escala durante o treinamento.
        """
        # Escala 3: 256D — projeção treinada do bottleneck 512D
        t3 = self.proj_t3(bottleneck)

        # Escala 2: 128D — projeção treinada do LFA (já em 128D)
        if t_lfa is not None:
            t2 = self.proj_t2(t_lfa)
        elif t_adapter is not None:
            t2 = self.proj_t2(t_adapter)
        else:
            t2 = self.proj_t2(bottleneck[:, :128])

        # Escala 1: 64D — projeção treinada do LFA para 64D
        if t_lfa is not None:
            t1 = self.proj_t1(t_lfa)
        elif t_block0 is not None:
            t1 = self.proj_t1(t_block0)
        else:
            t1 = self.proj_t1(bottleneck[:, :128])

        return [t3, t2, t1]   # [256D, 128D, 64D]

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (N, input_dim)
        Returns dict com:
          'teacher_scales' : [t3, t2, t1] — features do Teacher (detached)
          'student_scales' : [s3, s2, s1] — reconstruções do Student
          'bottleneck'     : (N, 512)
        """
        # Teacher: sempre sem gradientes
        bottleneck, teacher_scales = self.teacher_features(x)

        # Student: recebe bottleneck, reconstrói escalas
        student_scales = self.student(bottleneck.detach())

        return {
            'teacher_scales': [t.detach() for t in teacher_scales],
            'student_scales' : student_scales,
            'bottleneck'     : bottleneck.detach(),
        }

    @torch.no_grad()
    def anomaly_score_per_point(self, x: torch.Tensor) -> np.ndarray:
        """
        Calcula score de anomalia por ponto (inferência).
        x: (N, input_dim)
        Returns: (N,) float32 — score [0, 2], maior = mais anômalo
        """
        self.eval()
        out = self.forward(x)
        T   = out['teacher_scales']
        S   = out['student_scales']
        W   = self.SCALE_WEIGHTS
        N   = x.size(0)

        score = torch.zeros(N, device=x.device)
        for t, s, w in zip(T, S, W):
            # Distância coseno por ponto: 1 - cos_sim ∈ [0, 2]
            cos = nn.functional.cosine_similarity(t, s, dim=1)
            score += w * (1.0 - cos)

        return score.cpu().numpy().astype(np.float32)


# ============================================================================
# LOSS DE DISTILAÇÃO
# ============================================================================

class DistillationLoss(nn.Module):
    """
    Loss combinada para Reverse Distillation:

    L = α_cos · (1 - cosine_similarity(T, S))
      + α_mse · MSE(T, S) / T²    (normalizado pela magnitude do Teacher)

    A normalização pelo Teacher evita que escalas com features de maior
    magnitude dominem o gradiente [Hinton et al., 2015 — Knowledge Distillation].

    Aplicada nas 3 escalas com pesos [0.2, 0.3, 0.5] — maior peso na
    escala local (64D) onde rachaduras são mais discriminativas.
    """

    def __init__(self, alpha_cos: float = ALPHA_COS,
                 alpha_mse: float = ALPHA_MSE,
                 scale_weights: list = None):
        super().__init__()
        self.alpha_cos = alpha_cos
        self.alpha_mse = alpha_mse
        self.w = scale_weights or TeacherStudentModel.SCALE_WEIGHTS

    def forward(self, teacher_scales: list, student_scales: list) -> torch.Tensor:
        total = torch.tensor(0.0, device=teacher_scales[0].device)

        for t, s, w in zip(teacher_scales, student_scales, self.w):
            # ── Loss cosseno: penaliza direção errada ─────────────────────────
            cos_sim = nn.functional.cosine_similarity(t, s, dim=1)
            l_cos   = (1.0 - cos_sim).mean()

            # ── Loss MSE normalizada: penaliza magnitude errada ───────────────
            t_norm  = nn.functional.normalize(t, dim=1)
            s_norm  = nn.functional.normalize(s, dim=1)
            l_mse   = nn.functional.mse_loss(s_norm, t_norm)

            total = total + w * (self.alpha_cos * l_cos + self.alpha_mse * l_mse)

        return total


# ============================================================================
# TREINAMENTO PROFESSOR-ALUNO
# ============================================================================

def train_teacher_student(model: TeacherStudentModel,
                          train_loader: DataLoader,
                          device: torch.device,
                          num_epochs: int = NUM_EPOCHS,
                          lr: float = LR_STUDENT,
                          use_early_stopping: bool = True,
                          memory_bank: NormalMemoryBank = None,
                          save_dir: str = MODELS,
                          labeled_loader: DataLoader = None) -> tuple:
    """
    Treina apenas o Student decoder. Teacher permanece 100% congelado.

    Fases:
      Fase 1 (0 … num_epochs//3): LR normal — Student aprende representação básica
      Fase 2 (num_epochs//3 … fim): LR × 0.3 — refinamento sutil

    Loss: DistillationLoss (cosine + MSE normalizado em 3 escalas)

    Não há dinâmica adversarial, N_CRITIC, gradient penalty ou colapso de modo.
    Treinamento determinístico e estável por design.

    Referência: Deng & Li (2022) — Reverse Distillation, CVPR.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Parâmetros treináveis: Student decoder + projeções de escala do Teacher
    trainable_params = list(model.student.parameters()) + \
                       list(model.proj_t3.parameters()) + \
                       list(model.proj_t2.parameters()) + \
                       list(model.proj_t1.parameters())
    optimizer = AdamW(
        trainable_params,
        lr=lr, betas=(0.9, 0.999), weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6)

    distill_loss = DistillationLoss()
    scaler       = GradScaler()

    push_pull    = PushPullLoss(alpha=0.75,
                               scale_weights=TeacherStudentModel.SCALE_WEIGHTS)
    history = {'loss': [], 'lr': [], 'sup_loss': []}
    best_loss  = float('inf')
    best_epoch = 0
    patience_c = 0
    patience   = 20 if use_early_stopping else num_epochs
    phase1_end = max(10, num_epochs // 2)   # mais epochs para memory bank (fase 2)
    phase2_done = False

    log.info(f"\n{'='*65}")
    log.info(f"PROFESSOR-ALUNO — Treinamento do Student Decoder")
    log.info(f"Teacher: PointTransformer (frozen) | Student: Decoder treinável")
    log.info(f"{'='*65}")

    for epoch in range(num_epochs):

        # Fase 2: reduzir LR para refinamento
        if epoch == phase1_end and not phase2_done:
            for pg in optimizer.param_groups:
                pg['lr'] *= 0.3
            log.info(f"\nFase 2 (epoch {epoch+1}): LR × 0.3 → refinamento")
            phase2_done = True

        model.student.train()
        model.teacher.eval()   # Teacher SEMPRE em eval

        ep_losses = []
        n_skip    = 0

        for batch in train_loader:
            try:
                x = batch['features'][0].to(device, non_blocking=True)
                # x: (N, 15)

                if x.size(0) < 32:
                    n_skip += 1
                    continue

                def _step(use_ckpt=False):
                    if use_ckpt:
                        model.teacher.use_checkpoint = True
                    optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        out  = model(x) 
                        loss = distill_loss(
                            out['teacher_scales'], out['student_scales'])
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.student.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if use_ckpt:
                        model.teacher.use_checkpoint = False
                    return loss.item()

                try:
                    lv = _step(use_ckpt=False)
                except RuntimeError as oom:
                    if 'out of memory' not in str(oom).lower():
                        raise
                    torch.cuda.empty_cache()
                    log.warning(f"OOM ({x.size(0):,} pts) → grad checkpointing...")
                    try:
                        lv = _step(use_ckpt=True)
                    except RuntimeError:
                        torch.cuda.empty_cache()
                        n_skip += 1
                        continue

                ep_losses.append(lv)

                # Memory bank: coletar bottlenecks normais (Fase 2)
                if memory_bank is not None and epoch >= phase1_end:
                    with torch.no_grad():
                        btn = model.teacher(x)   # (N, 512)
                    memory_bank.update(btn.cpu())

            except RuntimeError as e:
                log.warning(f"Batch erro: {e}")
                torch.cuda.empty_cache()
                n_skip += 1

        if not ep_losses:
            log.warning(f"Epoch {epoch+1}: sem batches válidos")
            continue

        # ── Fase 2b: loop supervisionado (Push-Pull) ──────────────────────────
        avg_sup = float('nan')
        if labeled_loader is not None and epoch >= phase1_end:
            sup_losses = []
            for batch in labeled_loader:
                x = batch['features'][0].to(device)
                y = batch['labels'][0].float().to(device)   # (N,) scalar_labels
                if x.size(0) < 32 or y.sum() == 0:
                    # pula nuvens sem pontos de rachadura anotados
                    continue
                try:
                    optimizer.zero_grad(set_to_none=True)
                    with autocast():
                        out  = model(x)
                        loss = push_pull(
                            out['teacher_scales'], out['student_scales'], y)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    sup_losses.append(loss.item())
                except RuntimeError as e:
                    err_str = str(e).lower()
                    log.warning(f"Supervised batch erro: {e}")
                    if any(k in err_str for k in ('cublas', 'cuda error',
                                                   'device-side', 'illegal memory')):
                        raise
            if sup_losses:
                avg_sup = float(np.mean(sup_losses))
                log.info(f"   Semi-sup loss: {avg_sup:.5f} ({len(sup_losses)} batches)")

        avg_loss = float(np.mean(ep_losses))
        cur_lr   = optimizer.param_groups[0]['lr']
        vram     = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        history['loss'].append(avg_loss)
        history['lr'].append(cur_lr)
        history['sup_loss'].append(avg_sup)

        log.info(
            f"Epoch {epoch+1:03d}/{num_epochs} | Loss={avg_loss:.5f} | "
            f"LR={cur_lr:.2e} | VRAM={vram:.1f}GB | skip={n_skip}"
        )

        scheduler.step()

        # Salvar melhor modelo
        if avg_loss < best_loss:
            best_loss, best_epoch = avg_loss, epoch
            patience_c = 0
            torch.save({
                'epoch'       : epoch,
                'student'     : model.student.state_dict(),
                'optimizer'   : optimizer.state_dict(),
                'history'     : history,
                'best_loss'   : best_loss,
                'timestamp'   : datetime.now().isoformat(),
            }, os.path.join(save_dir, 'best_student.pth'))
            log.info(f"   Melhor Student salvo (epoch {epoch+1}, loss={best_loss:.5f})")
        else:
            patience_c += 1

        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'student': model.student.state_dict(),
                        'history': history},
                       os.path.join(save_dir, f'student_ep{epoch+1:03d}.pth'))

        gc.collect()
        torch.cuda.empty_cache()

        # Early stopping simples: patience epochs sem melhoria após min_epochs
        if epoch >= 20 and patience_c >= patience:
            log.info(f"Early stop: {patience} epochs sem melhoria.")
            break

    log.info(f"\nTreino concluído. Melhor epoch: {best_epoch+1} (loss={best_loss:.5f})")
    return model, history


# ============================================================================
# INFERÊNCIA — SCORE DE ANOMALIA
# ============================================================================

@torch.no_grad()
def compute_anomaly_scores(model: TeacherStudentModel,
                           data_list: list,
                           device: torch.device,
                           memory_bank: NormalMemoryBank = None,
                           alpha_dist: float = 0.8,
                           alpha_bank: float = 0.2) -> list:
    """
    Score composto por ponto:

      score(p) = α_dist · distill_score(p)  +  α_bank · bank_dist(p)

    distill_score(p) = soma ponderada das distâncias coseno Teacher-Student
                       nas 3 escalas (principal sinal de anomalia)
    bank_dist(p)     = distância coseno média aos k vizinhos normais no banco
                       de memória (sinal complementar [PatchCore / Uni-3DAD])

    Se memory_bank=None: score = distill_score apenas (alpha_dist=1.0).
    """
    model.eval()
    results = []

    for d in data_list:
        x      = torch.tensor(d['features'], dtype=torch.float32).to(device)
        labels = d['labels']

        # Score de distilação por ponto (3 escalas)
        raw_score = model.anomaly_score_per_point(x)   # (N,) numpy

        # Score do banco de memória
        if memory_bank is not None and memory_bank.bank is not None:
            btn       = model.teacher(x)               # (N, 512)
            bank_dist = memory_bank.score(btn.cpu(), k=5)
            lo, hi    = bank_dist.min(), bank_dist.max()
            bank_norm = (bank_dist - lo) / (hi - lo + 1e-8)
            lo2, hi2  = raw_score.min(), raw_score.max()
            dist_norm = (raw_score - lo2) / (hi2 - lo2 + 1e-8)
            score     = alpha_dist * dist_norm + alpha_bank * bank_norm
        else:
            lo, hi = raw_score.min(), raw_score.max()
            score  = (raw_score - lo) / (hi - lo + 1e-8)

        results.append({
            'filename'  : d['filename'],
            'has_crack' : d['has_crack'],
            'score'     : score,
            'gt_labels' : labels,
            'n_points'  : len(labels),
            'xyz'       : d['features'][:, :3],   # coordenadas para filtro espacial
        })

    return results


# ============================================================================
# THRESHOLD GMM — idêntico ao usado pela GAN
# ============================================================================

def build_model(device: torch.device) -> TeacherStudentModel:
    model = TeacherStudentModel(
        input_dim=INPUT_DIM,
        teacher_ckpt=PTRANSF_WEIGHTS,
    ).to(device)
    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M')
    print("\n" + "="*70)
    print("  PROFESSOR-ALUNO — REVERSE DISTILLATION")
    print("  PointTransformer (Teacher) + Student Decoder | GMM Threshold")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    for p in [RESULTS, MODELS, VIS_PATH, LOGS_PATH]:
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
        log.error("Nenhum dado encontrado. Verifique DATA_TRAIN / DATA_TEST.")
        return

    train_list, labeled_list, eval_list = split_dataset(all_data)
    train_dl, labeled_dl, _            = make_loaders(train_list, eval_list,
                                                       labeled_list=labeled_list)

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    model = build_model(device)

    # ── 3. Memory bank ────────────────────────────────────────────────────────
    memory_bank = NormalMemoryBank(max_size=100_000, subsample_k=1_000)

    # ── 4. Treino ─────────────────────────────────────────────────────────────
    log.info("\nIniciando treino do Student decoder...")
    t0 = time.time()

    model, history = train_teacher_student(
        model, train_dl, device,
        num_epochs=NUM_EPOCHS,
        lr=LR_STUDENT,
        use_early_stopping=True,
        memory_bank=memory_bank,
        save_dir=MODELS,
        labeled_loader=labeled_dl,
    )
    start = (time.time()-t0)

    log.info(f"\n Treino: {start/3600:.1f}h")
    plot_training_history(history, VIS_PATH, ts=ts)

    # ── 5. Carregar melhor Student ────────────────────────────────────────────
    best_path = os.path.join(MODELS, 'best_student.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.student.load_state_dict(ckpt['student'])
        log.info(f"Melhor Student carregado (epoch {ckpt['epoch']+1})")

    memory_bank.save(os.path.join(MODELS, 'ts_memory_bank.pt'))

    # ── 6. Scores de anomalia ─────────────────────────────────────────────────
    log.info("\nComputando anomaly scores...")
    eval_data   = eval_list if eval_list else train_list
    results     = compute_anomaly_scores(model, eval_data, device, memory_bank,
                                         alpha_dist=0.65, alpha_bank=0.3)

    log.info("Computando scores de referência (nuvens normais)...")
    normal_ref  = compute_anomaly_scores(
        model, train_list[:min(20, len(train_list))],
        device, memory_bank)

    # ── 7. Threshold GMM ──────────────────────────────────────────────────────
    thr, gmm_info = fit_gmm_threshold(results, normal_results=normal_ref)
    log.info(f"Threshold ({gmm_info['method']}): {thr:.4f}")
    results       = apply_threshold(results, thr)

    # Sanidade: nuvens normais devem ter baixo % de anomalias
    normal_check = apply_threshold(
        compute_anomaly_scores(model, train_list[:5], device), thr)
    for r in normal_check:
        pct = r['pred_labels'].mean() * 100
        log.info(f"  Sanidade {r['filename']}: {pct:.1f}% "
                 f"{'OK !' if pct < 20 else 'WARNING'}")

    # ── 8. Comparação de estratégias de threshold ─────────────────────────────
    log.info("\n" + "="*65)
    log.info("COMPARAÇÃO DE ESTRATÉGIAS DE THRESHOLD")
    log.info("="*65)

    strategies = {
        'F1'         : calibrate_threshold_f1(results),
        'G-mean'     : calibrate_threshold_gmean(results),
        'F-beta(0.5)': calibrate_threshold_fbeta(results, beta=0.5),
    }

    comparison = []   # lista de dicts para exportar como CSV
    best_strategy  = None
    best_f1        = -1.0

    for name, thr_s in strategies.items():
        # Aplicar threshold e filtro espacial numa cópia isolada
        import copy
        res_s = apply_threshold(copy.deepcopy(results), thr_s)
        res_s = apply_spatial_coherence(res_s, min_neighbors=3, k=20)
        m_s   = evaluate(res_s)

        row = {
            'estrategia'  : name,
            'threshold'   : round(thr_s, 6),
            'precision'   : round(m_s.get('precision',  0), 4),
            'recall'      : round(m_s.get('recall',     0), 4),
            'f1'          : round(m_s.get('f1',         0), 4),
            'f1_macro'    : round(m_s.get('f1_macro',   0), 4),
            'iou'         : round(m_s.get('iou',        0), 4),
            'auroc'       : round(m_s.get('auroc',      0), 4),
            'avg_precision': round(m_s.get('average_precision', 0), 4),
        }
        comparison.append(row)

        log.info(f"\n  [{name}]  thr={thr_s:.4f}")
        log.info(f"    Precision={row['precision']:.4f}  Recall={row['recall']:.4f}"
                 f"  F1={row['f1']:.4f}  IoU={row['iou']:.4f}  AUROC={row['auroc']:.4f}")

        if row['f1'] > best_f1:
            best_f1       = row['f1']
            best_strategy = name
            results_best  = res_s
            metrics_best  = m_s
            thr_best      = thr_s

    log.info(f"\n  Melhor estratégia: [{best_strategy}] (F1={best_f1:.4f})")
    log.info("="*65)

    # Salvar tabela de comparação
    comp_path = os.path.join(RESULTS, f'comparacao_thresholds_{ts}.csv')
    pd.DataFrame(comparison).to_csv(comp_path, index=False)
    log.info(f"Comparação salva: {comp_path}")

    # Usar a melhor estratégia para o restante do pipeline
    results = results_best
    metrics = metrics_best
    thr_f1  = thr_best   # mantém variável para compatibilidade abaixo

    # ── Severidade e classificação normativa ABNT ──────────────────────────────
    results    = compute_severity(results)
    relatorio  = severity_report(results)

    
    relatorio_path = os.path.join(RESULTS, f'relatorio_severidade_{ts}.csv')
    with open(relatorio_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['tipo', 'chave', 'valor'])
        for cat, n in relatorio['categorias'].items():
            writer.writerow(['categoria', cat, n])
        for risco, n in relatorio['riscos'].items():
            writer.writerow(['risco', risco, n])
        writer.writerow(['resumo', 'pct_conforme_abnt', f"{relatorio['pct_conforme']:.1f}%"])
        writer.writerow(['resumo', 'limite_abnt_mm',    relatorio['limite_abnt_mm']])
        writer.writerow(['resumo', 'caa',               relatorio['caa']])
        writer.writerow(['resumo', 'n_nuvens_avaliadas',relatorio['n_avaliadas']])

    # Salvar também por nuvem (abertura, profundidade, categoria)
    nuvens_path = os.path.join(RESULTS, f'severidade_por_nuvem_{ts}.csv')
    with open(nuvens_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['arquivo', 'categoria', 'abertura_mm', 'profundidade_mm',
                        'severidade_pct', 'risco', 'conforme_abnt'])
        for r in results:
            if not r['has_crack']:
                continue
            cl = r.get('classificacao', {})
            writer.writerow([
                r['filename'],
                cl.get('categoria', '?'),
                f"{r.get('abertura_mm', float('nan')):.4f}",
                f"{r.get('profundidade_mm', float('nan')):.4f}",
                f"{r.get('severidade_pct', 0):.2f}",
                cl.get('risco', '?'),
                cl.get('conforme_abnt', '?'),
            ])
    log.info(f"Relatório salvo: {relatorio_path}")
    log.info(f"Por nuvem salvo: {nuvens_path}")

    # ── Visualizações com contorno vermelho ───────────────────────────────────
    try:
        visualize_cracks(results, save_dir=VIS_PATH, max_clouds=10, ts=ts)
    except Exception as e:
        log.error(f"visualize_cracks falhou: {e}", exc_info=True)

    # ── 9. Distribuição de scores ─────────────────────────────────────────────
    try:
        plot_score_distribution(results, thr_f1, VIS_PATH, gmm_info, ts=ts)
    except Exception as e:
        log.error(f"plot_score_distribution falhou: {e}", exc_info=True)

    # ── Métricas e predições ──────────────────────────────────────────────────
    try:
        save_results(metrics, results, history, thr_f1, gmm_info, VIS_PATH, ts=ts)
    except Exception as e:
        log.error(f"save_results falhou: {e}", exc_info=True)

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTADOS FINAIS — PROFESSOR-ALUNO")
    print("="*70)
    if metrics:
        print(f"  F1    : {metrics.get('f1',  0):.4f}  (target ≥ 0.75)")
        print(f"  IoU   : {metrics.get('iou', 0):.4f}  (target ≥ 0.60)")
        print(f"  AUROC : {metrics.get('auroc', 0):.4f}  (target ≥ 0.85)")
    print(f"\n  Threshold ({gmm_info['method']}): {thr:.4f}")
    print(f"  Resultados em: {RESULTS}")
    print("="*70)

    return model, history, results, metrics


if __name__ == '__main__':
    main()
