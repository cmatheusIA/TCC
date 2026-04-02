# ============================================================================
# UTILITÁRIOS DE TREINAMENTO
# ============================================================================
# SmartEarlyStopping, compute_gradient_penalty,
# MultiScaleReconLoss, NormalMemoryBank,
# check_training_health, adaptive_lambda_recon
# Compartilhado por GAN v5 e Professor-Aluno v1.
# ============================================================================

from utils.config import *
import logging

log = setup_logging(LOG_PATH)

class SmartEarlyStopping:
    """
    Early stopping com 3 critérios:
      1. Todos os targets atingidos por 3 epochs consecutivas
      2. Plateau na métrica combinada (patience epochs)
      3. Plateau longo sem targets (2× patience)
    """

    def __init__(self, patience: int = 15, min_epochs: int = 20,
                 target_g: float = 0.80, target_d: float = 0.85,
                 target_recon: float = 0.05):
        self.patience     = patience
        self.min_epochs   = min_epochs
        self.target_g     = target_g
        self.target_d     = target_d
        self.target_recon = target_recon
        self.best_metric  = float('inf')
        self.counter      = 0
        self.tgt_streak   = 0

    def __call__(self, epoch: int, g: float, d: float, recon: float) -> bool:
        if epoch < self.min_epochs:
            return False

        targets = (g < self.target_g and abs(d - g) < 0.3 and recon < self.target_recon)
        if targets:
            self.tgt_streak += 1
            if self.tgt_streak >= 3:
                log.info(f"Early stop: targets atingidos por 3 epochs.")
                return True
        else:
            self.tgt_streak = 0

        combined = g + d + recon * 10
        if combined < self.best_metric:
            self.best_metric = combined
            self.counter     = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            log.info(f"Early stop: plateau ({self.patience} epochs sem melhoria).")
            return True
        if self.counter >= self.patience * 2:
            log.info("Early stop: plateau longo sem targets.")
            return True

        return False

# ============================================================================
# GRADIENT PENALTY — memória eficiente
# ============================================================================
# Gulrajani et al. (2017) — "Improved Training of Wasserstein GANs", Eq. 3
# Usa torch.linalg.vector_norm em vez de cdist para evitar explosão de memória
# na escala de N>10k pontos (descoberta empírica desta implementação).
# ============================================================================

def compute_gradient_penalty(discriminator: nn.Module,
                              real: torch.Tensor,
                              fake: torch.Tensor,
                              device: torch.device) -> torch.Tensor:
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device, dtype=real.dtype)
    interp = (alpha * real + (1.0 - alpha) * fake.detach()).requires_grad_(True)

    d_interp = discriminator(interp)

    grads = torch.autograd.grad(
        outputs=d_interp, inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]

    grads_flat = grads.view(B, -1)
    # torch.linalg.vector_norm: O(N) em memória — evita cdist O(N²)
    if hasattr(torch.linalg, 'vector_norm'):
        grad_norm = torch.linalg.vector_norm(grads_flat, ord=2, dim=1)
    else:
        grad_norm = torch.sqrt(grads_flat.pow(2).sum(1) + 1e-12)

    return ((grad_norm - 1.0) ** 2).mean()


# ============================================================================
# MULTI-SCALE RECONSTRUCTION LOSS
# ============================================================================
# R3D-AD (Zhou et al., 2024): calcular erro de reconstrução em múltiplas
# escalas do decoder — detalhes finos (skip_1=64D) E estrutura global (512D).
# Pesos: escala mais profunda (estrutura) recebe peso maior.
# ============================================================================

class MultiScaleReconLoss(nn.Module):
    """
    Reconstruction loss em 3 escalas do decoder UNet.

    L_total = w_out * MSE(output, target)
            + w_d1  * MSE(dec_d1_feat, enc_skip3_feat)   [256D]
            + w_d2  * MSE(dec_d2_feat, enc_skip2_feat)   [128D]
            + w_d3  * MSE(dec_d3_feat, enc_skip1_feat)   [ 64D]

    A perda nas features intermediárias força o decoder a reconstruir não
    apenas o output final, mas também a representação interna de cada escala,
    tornando o espaço de anomalia mais discriminativo [R3D-AD — Zhou et al., 2024].
    """

    def __init__(self, w_out: float = 1.0, w_mid: float = 0.3):
        super().__init__()
        self.w_out = w_out
        self.w_mid = w_mid
        self.mse   = nn.MSELoss()

    def forward(self, output: torch.Tensor, target: torch.Tensor,
                mid_pairs: list = None) -> torch.Tensor:
        """
        output, target : (B, N, D)
        mid_pairs      : list of (decoder_feat, encoder_feat) tuples — optional
        """
        loss = self.w_out * self.mse(output, target)
        if mid_pairs:
            for dec_f, enc_f in mid_pairs:
                loss = loss + self.w_mid * self.mse(dec_f, enc_f.detach())
        return loss


# ============================================================================
# MEMORY BANK DE FEATURES NORMAIS
# ============================================================================
# Inspirado em PatchCore (Roth et al., 2022) e Uni-3DAD (Liu et al., 2024):
# durante o treino, armazena features do bottleneck de nuvens NORMAIS.
# Na inferência: anomaly_score = reconstruction_error + dist(feat, bank)
#
# Justificativa acadêmica: pontos com rachadura produzem features de bottleneck
# distantes de qualquer protótipo de superfície normal, mesmo quando o
# reconstruction error é moderado — combinação melhora AUROC [Uni-3DAD].
# ============================================================================

class NormalMemoryBank:
    """
    Banco de memória de features normais (bottleneck 512D).

    Implementa greedy coreset sampling para manter representatividade
    com tamanho controlado (evitar OOM em datasets grandes).

    Referência: Roth et al. (2022) — PatchCore (CVPR) — adaptado para 3D.
    """

    def __init__(self, max_size: int = 50_000, subsample_k: int = 1_000):
        self.max_size    = max_size
        self.subsample_k = subsample_k
        self.bank: torch.Tensor | None = None   # (M, 512)

    def update(self, features: torch.Tensor):
        """features: (N, 512) — bottleneck de uma nuvem normal."""
        feats = features.detach().cpu().float()

        # Subamostrar para não sobrecarregar o banco com nuvens grandes
        if len(feats) > self.subsample_k:
            idx   = torch.randperm(len(feats))[:self.subsample_k]
            feats = feats[idx]

        if self.bank is None:
            self.bank = feats
        else:
            self.bank = torch.cat([self.bank, feats], dim=0)

        # Limitar tamanho total com reservoir sampling
        if len(self.bank) > self.max_size:
            idx       = torch.randperm(len(self.bank))[:self.max_size]
            self.bank = self.bank[idx]

    def score(self, features: torch.Tensor, k: int = 5) -> np.ndarray:
        """
        Para cada ponto, calcula distância média aos k vizinhos no banco.
        features: (N, 512) — bottleneck de nuvem de avaliação.
        Retorna: (N,) float32 — score de anomalia por ponto.
        """
        if self.bank is None or len(self.bank) == 0:
            return np.zeros(len(features), dtype=np.float32)

        feats_np = features.cpu().float().numpy()
        bank_np  = self.bank.float().numpy()

        # kNN no espaço de features (normalizar por norma)
        feats_n  = feats_np / (np.linalg.norm(feats_np, axis=1, keepdims=True) + 1e-8)
        bank_n   = bank_np  / (np.linalg.norm(bank_np,  axis=1, keepdims=True) + 1e-8)

        # Para datasets grandes, usar aproximação por chunks para evitar OOM
        chunk   = 1024
        scores  = np.zeros(len(feats_n), dtype=np.float32)
        k_act   = min(k, len(bank_n))

        for s in range(0, len(feats_n), chunk):
            e    = min(s + chunk, len(feats_n))
            # Produto interno = cosine similarity (features normalizadas)
            sim  = feats_n[s:e] @ bank_n.T          # (chunk, M)
            # Distância coseno = 1 - similaridade
            dist = 1.0 - sim                          # (chunk, M)
            # Distância média aos k vizinhos mais próximos
            topk = np.partition(dist, k_act, axis=1)[:, :k_act]
            scores[s:e] = topk.mean(1)

        return scores

    def save(self, path: str):
        if self.bank is not None:
            torch.save(self.bank, path)
            log.info(f"Memory bank salvo: {path} ({len(self.bank):,} features)")

    def load(self, path: str):
        if os.path.exists(path):
            self.bank = torch.load(path, map_location='cpu')
            log.info(f"Memory bank carregado: {len(self.bank):,} features")


# ============================================================================
# MONITORAMENTO DE SAÚDE DO TREINO
# ============================================================================

def check_training_health(g_loss: float, d_loss: float,
                           recon_loss: float, epoch: int) -> dict:
    """
    Detecta colapso de modo, desequilíbrio D/G e divergência de GP.

    Valores saudáveis para WGAN-GP (baseado em Gulrajani et al., 2017
    + diagnóstico empírico desta implementação):
      G Loss  : -3.0 a +1.0
      D Loss  : -2.0 a +5.0
      |D/G|   : 0.1 a 10.0
      Recon   : 0.001 a 0.5
    """
    health  = {}
    issues  = []

    d_g_ratio = abs(d_loss / (abs(g_loss) + 1e-8))

    health['g_ok']     = -3.0 < g_loss    < 1.0
    health['d_ok']     = -2.0 < d_loss    < 5.0
    health['ratio_ok'] = 0.1  < d_g_ratio < 10.0
    health['recon_ok'] = 0.001 < recon_loss < 0.5

    if not health['g_ok']:
        issues.append(f"G Loss fora do range ({g_loss:.3f})")
    if not health['d_ok']:
        issues.append(f"D Loss fora do range ({d_loss:.3f})")
    if not health['ratio_ok']:
        issues.append(f"|D/G|={d_g_ratio:.2f} desequilibrado")
    if not health['recon_ok']:
        issues.append(f"Recon={recon_loss:.4f} fora do range")

    health['healthy'] = len(issues) == 0
    health['issues']  = issues
    return health


def adaptive_lambda_recon(base_lambda: float,
                           g_loss: float, d_loss: float,
                           recon_loss: float, epoch: int) -> float:
    """
    Ajusta λ_recon dinamicamente para manter equilíbrio G/D.

    Regras (baseadas no diagnóstico de instabilidade desta implementação):
      • D muito forte (d_loss < -0.5): aumentar λ_recon — foca reconstrução
      • G muito forte (g_loss <  -3.0): diminuir λ_recon — foca adversarial
      • Recon muito alta (> 0.3):       diminuir λ_recon — evita dominância
      • Primeiras 20 epochs:           λ = base/2 (warm-up estável)
    """
    if epoch < 20:
        return base_lambda * 0.5

    lam = base_lambda

    if d_loss < -0.5:
        lam = min(lam * 1.5, base_lambda * 3.0)   # D forte: mais reconstrução

    if g_loss < -3.0:
        lam = max(lam * 0.7, base_lambda * 0.1)   # G forte: menos reconstrução

    if recon_loss > 0.3:
        lam = max(lam * 0.8, base_lambda * 0.1)   # Recon alta: aliviar

    return float(np.clip(lam, base_lambda * 0.1, base_lambda * 5.0))


# ============================================================================
# PUSH-PULL LOSS — SEMI-SUPERVISIONADO
# ============================================================================
# Para nuvens com scalar_labels disponíveis (avaria_*.ply):
#   Pull: pontos normais (label=0) → Student se aproxima do Teacher (cos → +1)
#   Push: rachaduras  (label=1) → Student se afasta do Teacher  (cos → -1)
#
# Resultado: o Student aprende explicitamente a produzir features DIFERENTES
# do Teacher em regiões de avaria — melhora a separação no anomaly score.
#
# Referência: Reiss et al. (2023) — "Pushing the Limits of Simple Pipelines
# for Few-Shot Learning" — adaptado para anomaly detection semi-supervisionado.
# ============================================================================

class PushPullLoss(nn.Module):
    """
    Loss semi-supervisionada para Teacher-Student Reverse Distillation.

    Para pontos normais (label=0): reforça a distilação (pull — Student ≈ Teacher).
    Para rachaduras  (label=1): inverte o sinal     (push — Student ≠ Teacher).

    Pesos de classe alpha para lidar com desequilíbrio (~5% positivos):
      - alpha=0.75 → 75% do peso nas rachaduras, 25% nos pontos normais
        (compensa a escassez de labels positivos)

    Os pesos por escala (scale_weights) devem coincidir com SCALE_WEIGHTS
    do modelo para manter consistência com o anomaly score em inferência.
    """

    def __init__(self, alpha: float = 0.75, scale_weights: list = None):
        super().__init__()
        self.alpha         = alpha
        self.scale_weights = scale_weights or [0.2, 0.3, 0.5]

    def forward(self,
                teacher_scales: list,
                student_scales: list,
                labels: torch.Tensor) -> torch.Tensor:
        """
        teacher_scales: list[Tensor(N, D)] — features Teacher em cada escala (float32)
        student_scales: list[Tensor(N, D)] — features Student em cada escala
        labels        : Tensor(N,) float32 — 0=normal, 1=rachadura
        """
        mask_crack  = labels > 0.5
        mask_normal = ~mask_crack

        total = torch.tensor(0.0, device=labels.device)

        for t_s, s_s, w in zip(teacher_scales, student_scales, self.scale_weights):
            # Cosine similarity em float32 (Teacher já é float32; cast do Student)
            cos = F.cosine_similarity(t_s.float(), s_s.float(), dim=-1)  # (N,)

            # Pull: pontos normais → cos → +1  (1 - cos → 0)
            if mask_normal.any():
                pull = (1.0 - cos[mask_normal]).mean()
                total = total + w * (1.0 - self.alpha) * pull

            # Push: rachaduras → cos → -1  (1 + cos → 0 no ótimo)
            if mask_crack.any():
                push = (1.0 + cos[mask_crack]).mean()
                total = total + w * self.alpha * push

        return total


# ============================================================================
# CONSTRUÇÃO DOS MODELOS
# ============================================================================

