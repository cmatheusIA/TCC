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
from utils.config import *
from utils.data import *
from utils.evaluation import (
    ScalarFieldGMM, save_colored_ply,
    fit_gmm_threshold, apply_threshold, evaluate,
    calibrate_threshold_f1, calibrate_threshold_gmean, calibrate_threshold_fbeta,
)

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
TEACHER_CHUNK_SIZE = 8_000


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
    _, idx = tree.query(feats, k=k_query, workers=-1)   # usa todos os CPUs
    # Remove self (coluna 0)
    idx = idx[:, 1:k+1] if idx.shape[1] > k else idx[:, 1:]
    return torch.from_numpy(idx.astype(np.int64))


# ============================================================================
# KNN GPU — chunked torch.cdist (sem transferência GPU→CPU, sem OOM)
# ============================================================================

def build_knn_gpu(h: torch.Tensor, k: int, chunk: int = 1000) -> torch.Tensor:
    """
    kNN diretamente na GPU usando torch.cdist por chunks.
    Evita a transferência GPU→CPU→GPU do build_knn_idx.
    Controla OOM via chunk: chunk=1000, N=60k → ~240MB por chunk.

    h     : (N, D) tensor na GPU
    k     : número de vizinhos (excluindo self)
    chunk : linhas processadas por vez (reduzir se OOM)
    Returns: (N, k) torch.int64 na GPU
    """
    N = h.size(0)
    k_eff = min(k, N - 1)
    idx = torch.zeros(N, k_eff, dtype=torch.long, device=h.device)

    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        dists = torch.cdist(h[s:e], h)              # (chunk, N)
        # Mascara self-loop com inf
        dists[:, s:e].fill_diagonal_(float('inf'))
        idx[s:e] = dists.topk(k_eff, dim=1, largest=False).indices

    return idx


def _knn1_worker(args):
    """Worker top-level para multiprocessing.Pool (não pode ser função local)."""
    h_np, k = args
    return build_knn_idx(h_np, k)


def _fuse_gmm_worker(args):
    """
    Worker top-level para ProcessPoolExecutor — fora da função para ser picklável.
    Executa ScalarFieldGMM + fusão de scores para uma nuvem.
    """
    d, score_raw = args
    sf_raw   = d['features'][:, 9]
    sf_gmm   = ScalarFieldGMM(sf_raw).fit()
    sf_score = sf_gmm.anomaly_probability()

    lo, hi    = score_raw.min(), score_raw.max()
    dist_norm = (score_raw - lo) / (hi - lo + 1e-8)
    score     = 0.7 * dist_norm + 0.3 * sf_score

    return {
        'filename'    : d['filename'],
        'has_crack'   : d['has_crack'],
        'score'       : score,
        'gt_labels'   : d['labels'],
        'n_points'    : len(d['labels']),
        'xyz'         : d['features'][:, :3],
        'rgb'         : d['features'][:, 3:6],
        'scalar_field': sf_raw,
    }


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
        self.teacher = None   # definido externamente via build_dgcnn()

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

    def forward(
        self,
        x: torch.Tensor,
        h_cached: torch.Tensor = None,
        knn1_cached: torch.Tensor = None,
        knn_chunk: int = 1000,
    ) -> tuple:
        """
        x           : (N, input_dim)
        h_cached    : (N, 512) bottleneck Teacher pré-computado (opcional).
                      Evita re-rodar PTv3 a cada epoch (Teacher é frozen).
        knn1_cached : (N, k) kNN pré-computado sobre features Teacher (opcional).
                      Block 1 opera em h_teacher que nunca muda — pode ser fixo.
        knn_chunk   : chunk para build_knn_gpu (blocks 2+3, ajustar se OOM).
        Returns: (score (N,), recon (N, 512))
        """
        # Teacher features — usa cache se disponível (Teacher é frozen)
        h = h_cached if h_cached is not None else self._teacher_bottleneck(x)

        # Block 1 — kNN sobre features Teacher (fixo entre epochs se Teacher frozen)
        if knn1_cached is not None:
            knn1 = knn1_cached.to(x.device)
        else:
            knn1 = build_knn_idx(h.detach().cpu().numpy(), self.k).to(x.device)
        h = self.conv1(h, knn1)              # (N, 256)

        # Blocks 2 e 3 — grafo dinâmico na GPU (sem transferência CPU↔GPU)
        knn2 = build_knn_gpu(h.detach(), self.k, chunk=knn_chunk)
        h = self.conv2(h, knn2)              # (N, 128)

        knn3 = build_knn_gpu(h.detach(), self.k, chunk=knn_chunk)
        h = self.conv3(h, knn3)              # (N, 64)

        score = self.score_head(h).squeeze(-1)   # (N,)
        recon = self.recon_head(h)               # (N, 512)
        return score, recon


# ============================================================================
# BUILD DO MODELO
# ============================================================================

def build_dgcnn(device: torch.device) -> ScalarFieldDGCNN:
    """Instancia ScalarFieldDGCNN com Teacher via fallback A→B→C."""
    from teacher_student_v1 import build_teacher
    teacher = build_teacher(
        input_dim=INPUT_DIM,
        ptv3_ckpt=PTRANSF_WEIGHTS,
        s3dis_ckpt=PTRANSF_WEIGHTS_S3DIS,
    )
    model = ScalarFieldDGCNN(input_dim=INPUT_DIM, k=DGCNN_K)
    model.teacher = teacher
    model._freeze_teacher()
    return model.to(device)


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

    # ── Pré-computação: Teacher bottleneck + knn1 (ambos fixos durante o treino) ──
    # Teacher é frozen → h_teacher não muda entre epochs.
    # knn1 opera sobre h_teacher (512D) → cKDTree é ineficiente em alta dimensão
    # (curse of dimensionality → O(N²)); usamos build_knn_gpu na mesma passagem.
    # Guardamos h em fp16 na CPU para economizar RAM (~50% vs fp32).
    log.info("Pré-computando Teacher bottlenecks + knn1 na GPU (1× antes do treino)...")
    teacher_cache    = []   # list[Tensor(N,512) fp16 CPU]
    teacher_cache_f32 = []  # fp32 para l_recon (evita cast no loop interno)
    knn1_cache       = []   # list[Tensor(N,k)  int64 CPU]
    for idx_c, d in enumerate(all_data):
        x_c = torch.tensor(d['features'], dtype=torch.float32).to(device)
        with torch.no_grad():
            h_t   = model._teacher_bottleneck(x_c)          # (N, 512) fp32 GPU
            knn1  = build_knn_gpu(h_t, model.k, chunk=1000) # (N, k)   int64 GPU
        teacher_cache.append(h_t.half().cpu())      # fp16 CPU — metade da RAM
        teacher_cache_f32.append(h_t.cpu())         # fp32 CPU — para l_recon
        knn1_cache.append(knn1.cpu())               # int64 CPU
        del x_c, h_t, knn1
        torch.cuda.empty_cache()
        if (idx_c + 1) % 10 == 0:
            log.info(f"  Cache: {idx_c+1}/{len(all_data)}")
    log.info(f"Teacher + knn1 cache pronto: {len(teacher_cache)} nuvens")

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

            # Recupera caches (Teacher frozen → fixo entre epochs)
            h_cached   = teacher_cache[i].float().to(device)    # fp16→fp32 na GPU
            knn1_c     = knn1_cache[i]                          # (N,k) CPU int64
            teacher_btn_cached = teacher_cache_f32[i].to(device)  # para l_recon

            optimizer.zero_grad(set_to_none=True)

            try:
                with autocast():
                    score, recon = model(x, h_cached=h_cached, knn1_cached=knn1_c)

                    # ── L_recon (todas as nuvens) — usa cache, sem re-rodar Teacher
                    l_recon = F.mse_loss(recon, teacher_btn_cached.detach())

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

        # ── Transição warmup→self-sup: reset de métricas de parada ──────────
        # As losses das duas fases são incomparáveis (L_recon vs L_contrast+L_recon).
        # Sem reset, epoch 30 vira "melhor" por acidente e patience conta errado,
        # causando early stop prematuro ~20 epochs depois.
        if epoch == WARMUP_EPOCHS:
            best_loss  = avg_total
            patience_c = 0
            log.info(f"Transição self-sup: best_loss={best_loss:.5f} | patience reset")
            # Reinicia scheduler para que a self-sup comece com LR alto (T_0)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=15, T_mult=2, eta_min=1e-6,
            )
            log.info("Scheduler reiniciado para fase self-sup")

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

    Execução em duas fases para maximizar throughput:
      Fase 1 (GPU, sequencial): model(x) para todas as nuvens
      Fase 2 (CPU, paralela):   ScalarFieldGMM + fusão via ThreadPoolExecutor
                                 (sem overhead de pickle — threads compartilham memória)
    """
    from multiprocessing import cpu_count

    model.eval()

    # ── Pipeline GPU + CPU ───────────────────────────────────────────────────
    # Em vez de Fase1-completa → Fase2-completa, submete cada GMM assim que
    # o score GPU da nuvem fica pronto. GPU processa nuvem N enquanto CPU
    # já executa GMM das nuvens anteriores — overlap real de trabalho.
    #
    # torch.from_numpy + ascontiguousarray: zero-copy na CPU (evita o torch.tensor
    # que sempre copia). non_blocking=True: inicia DMA CPU→GPU sem bloquear Python.
    from concurrent.futures import ThreadPoolExecutor
    n_workers = min(cpu_count(), len(data_list))
    futures   = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for d in data_list:
            feat = np.ascontiguousarray(d['features'], dtype=np.float32)
            x    = torch.from_numpy(feat).to(device, non_blocking=True)

            h_t      = model._teacher_bottleneck(x)
            knn1     = build_knn_gpu(h_t, model.k, chunk=1000)
            score_raw, _ = model(x, h_cached=h_t, knn1_cached=knn1)
            score_np = score_raw.cpu().numpy()   # já float32 — sem astype extra
            del h_t, knn1, x

            # Submete GMM imediatamente — roda em paralelo com próxima nuvem GPU
            futures.append(executor.submit(_fuse_gmm_worker, (d, score_np)))

        results = [f.result() for f in futures]

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
