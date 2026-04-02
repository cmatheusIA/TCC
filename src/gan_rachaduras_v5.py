# ============================================================================
# GAN HÍBRIDA v5 — DETECÇÃO DE AVARIAS NÃO-SUPERVISIONADA
# ============================================================================
# Arquitetura:
#   Generator    : KPFCNNInspiredAdvanced  (KPConv + UNet skip connections)
#   Discriminator: PointTransformerInspiredAdvanced
# Protocolo:
#   Treino APENAS em nuvens normais (sem rachadura).
#   Detecção via erro de reconstrução por ponto + distância ao memory bank.
# Referências:
#   KPConv          : Thomas et al., 2019
#   PointTransformer: Zhao et al., 2021
#   WGAN-GP         : Gulrajani et al., 2017
#   R3D-AD          : Zhou et al., 2024
#   Uni-3DAD        : Liu et al., 2024
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.config import *
from utils import *
from utils.config import *
from utils.architectures import *
from utils.building_blocks import * 
from utils.data import *
from utils.evaluation import *
from utils.training_utils import *
import logging

log = setup_logging(LOG_PATH)

# ── Caminhos e hiperparâmetros específicos da GAN ─────────────────────────────
RESULTS    = f'{BASE_PATH}/results_v5'
MODELS     = f'{BASE_PATH}/models_v5'
VIS_PATH   = f'{BASE_PATH}/visualizations_v5'
LOGS_PATH  = f'{BASE_PATH}/logs_v5'

NUM_EPOCHS   = 100
LR_G         = 1e-4
LR_D         = 1e-4
LAMBDA_GP    = 10.0
LAMBDA_RECON = 1.0

log = setup_logging(LOGS_PATH)

def build_models(device: torch.device) -> tuple:
    log.info("Construindo modelos...")
    gen  = GANGenerator(INPUT_DIM,     checkpoint_path=KPCONV_WEIGHTS).to(device)
    disc = GANDiscriminator(INPUT_DIM,  checkpoint_path=PTRANSF_WEIGHTS).to(device)
    log.info(f"Generator:     {sum(p.numel() for p in gen.parameters()):,}  params")
    log.info(f"Discriminator: {sum(p.numel() for p in disc.parameters()):,} params")
    return gen, disc


# ============================================================================
# TREINAMENTO WGAN-GP  (versão melhorada)
# ============================================================================
# Melhorias aplicadas vs v5:
#   1. N_CRITIC=5  — padrão WGAN-GP [Gulrajani et al., 2017 — Algorithm 1]
#      Discriminador treinado 5× por passo do Generator, garantindo que a
#      função Wasserstein seja bem aproximada antes do passo adversarial.
#   2. G_STEPS=1   — Generator atualizado 1× (com N_CRITIC adequado é suficiente)
#   3. MultiScaleReconLoss — loss em 3 escalas do decoder [R3D-AD]
#   4. Memory bank — coleta features normais durante treino [Uni-3DAD / PatchCore]
#   5. Adaptive lambda_recon — ajuste dinâmico do peso de reconstrução
#   6. Logging correto — np.mean(lista) em vez de soma acumulada
#   7. Health monitoring por epoch com log de alertas
# ============================================================================

N_CRITIC = 5    # Padrão WGAN-GP [Gulrajani et al., 2017 — Algorithm 1, linha 5]


def train_wgan(generator: nn.Module,
               discriminator: nn.Module,
               train_loader: DataLoader,
               device: torch.device,
               num_epochs: int       = NUM_EPOCHS,
               lr_g: float           = LR_G,
               lr_d: float           = LR_D,
               lambda_gp: float      = LAMBDA_GP,
               lambda_recon: float   = LAMBDA_RECON,
               use_early_stopping: bool = True,
               memory_bank: 'NormalMemoryBank' = None,
               save_dir: str         = MODELS) -> tuple:
    """
    Treinamento WGAN-GP com N_CRITIC=5 em duas fases de transfer learning.

    Fase 1 (0 … phase1_end):
      Encoders pré-treinados congelados. Apenas as camadas novas
      (feature_projection, feature_adapter, decoder, heads) aprendem.
      Previne destruição dos pesos S3DIS antes de estabilizar o treino.

    Fase 2 (phase1_end … fim):
      Fine-tuning completo com LR×0.1 para ajuste sutil ao domínio
      da Igreja dos Homens Pretos sem apagar o conhecimento transferido.

    Referência: Kumar et al. (2022) — Transfer Learning para Point Clouds
    de Patrimônio Cultural.
    """
    os.makedirs(save_dir, exist_ok=True)

    # ── Losses ───────────────────────────────────────────────────────────────
    ms_recon = MultiScaleReconLoss(w_out=1.0, w_mid=0.3)

    # ── Fase 1: congelar encoders ─────────────────────────────────────────────
    phase1_end  = max(15, num_epochs // 3)
    phase2_done = False

    log.info(f"\n{'='*70}")
    log.info(f"FASE 1: Transfer Learning (epochs 0–{phase1_end})")
    log.info(f"{'='*70}")
    generator.freeze_encoder()
    discriminator.freeze_features()

    def make_optimizers(gen, disc, lr_g_val, lr_d_val):
        oG = AdamW(filter(lambda p: p.requires_grad, gen.parameters()),
                   lr=lr_g_val, betas=(0.5, 0.999), weight_decay=0.01)
        oD = AdamW(filter(lambda p: p.requires_grad, disc.parameters()),
                   lr=lr_d_val, betas=(0.5, 0.999), weight_decay=0.01)
        sG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(oG, T_0=10, T_mult=2, eta_min=1e-6)
        sD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(oD, T_0=10, T_mult=2, eta_min=1e-6)
        return oG, oD, sG, sD

    opt_G, opt_D, sched_G, sched_D = make_optimizers(generator, discriminator, lr_g, lr_d)
    scaler = GradScaler()

    early_stop = SmartEarlyStopping(patience=15, min_epochs=20) if use_early_stopping else None

    history = {
        'g_loss': [], 'd_loss': [], 'recon_loss': [], 'gp': [],
        'lr_g': [], 'lr_d': [], 'lambda_recon': [], 'health_score': []
    }
    best_g     = float('inf')
    best_epoch = 0
    batch_times = deque(maxlen=20)
    curr_lambda = lambda_recon

    # ─────────────────────────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ── Transição Fase 1 → Fase 2 ────────────────────────────────────────
        if epoch == phase1_end and not phase2_done:
            log.info(f"\n{'='*70}")
            log.info(f"FASE 2: Fine-tuning completo (epoch {epoch+1}+)")
            log.info(f"{'='*70}")
            generator.unfreeze_encoder()
            discriminator.unfreeze_features()
            opt_G, opt_D, sched_G, sched_D = make_optimizers(
                generator, discriminator, lr_g * 0.1, lr_d * 0.1)
            phase2_done = True

        generator.train()
        discriminator.train()

        ep_g, ep_d, ep_r, ep_gp = [], [], [], []
        n_skip = 0

        for batch in train_loader:
            batch_t = time.time()
            try:
                # Desempacotar (batch_size=1 — nuvem completa, sem padding)
                real = batch['features'][0].unsqueeze(0).to(device, non_blocking=True)
                # real: (1, N, 15)

                if real.size(1) < 32:
                    n_skip += 1
                    continue

                B     = real.size(0)
                N_pts = real.size(1)

                # ── N_CRITIC adaptativo ao tamanho da nuvem ───────────────────
                # Nuvens grandes consomem mais VRAM por passo do D.
                # Reduzir N_CRITIC para nuvens grandes mantém a semântica WGAN-GP
                # (D ainda converge antes de cada passo do G) sem exigir subsampling.
                #
                #   < 200k pts   → N_CRITIC=5  (padrão Gulrajani 2017)
                #   200k–500k    → N_CRITIC=3  (equilíbrio memória/convergência)
                #   > 500k pts   → N_CRITIC=1  (nuvem grande — 1 passo suficiente
                #                               pois G atualiza raramente)
                #
                # Fedus et al. (2018): 1–3 critic steps produzem resultados
                # equivalentes a 5 para distribuições unimodais.
                if N_pts < 200_000:
                    n_critic_eff = N_CRITIC
                elif N_pts < 500_000:
                    n_critic_eff = max(1, N_CRITIC - 2)
                else:
                    n_critic_eff = 1
                    log.debug(f"Nuvem grande ({N_pts:,} pts) → N_CRITIC=1")

                def _train_batch(use_ckpt: bool = False):
                    """
                    Passo completo D×n_critic + G×1.
                    use_ckpt=True ativa gradient checkpointing no encoder/decoder:
                    troca ~30% compute por ~40% menos VRAM de ativações.
                    Ativado apenas na segunda tentativa, após OOM na primeira.
                    """
                    if use_ckpt:
                        generator.encoder.use_checkpoint    = True
                        discriminator.encoder.use_checkpoint = True

                    d_loss_last = None
                    gp_last     = None

                    for _ in range(n_critic_eff):
                        opt_D.zero_grad(set_to_none=True)

                        with autocast():
                            real_v = discriminator(real)
                            d_real = -real_v.mean()

                            with torch.no_grad():
                                fake = generator(real)

                            fake_v = discriminator(fake.detach())
                            d_fake = fake_v.mean()

                            gp     = compute_gradient_penalty(
                                discriminator, real, fake, device)
                            d_loss = d_real + d_fake + lambda_gp * gp

                        scaler.scale(d_loss).backward()
                        torch.nn.utils.clip_grad_norm_(
                            discriminator.parameters(), 1.0)
                        scaler.step(opt_D)
                        scaler.update()

                        d_loss_last = d_loss.item()
                        gp_last     = gp.item()

                    opt_G.zero_grad(set_to_none=True)

                    with autocast():
                        fake  = generator(real)
                        val_g = discriminator(fake)
                        g_adv = -val_g.mean()
                        g_recon = ms_recon(fake, real, mid_pairs=None)
                        lam    = adaptive_lambda_recon(
                            lambda_recon, d_loss_last,
                            g_adv.item(), g_recon.item(), epoch)
                        g_loss = g_adv + lam * g_recon

                    scaler.scale(g_loss).backward()
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
                    scaler.step(opt_G)
                    scaler.update()

                    if use_ckpt:
                        generator.encoder.use_checkpoint    = False
                        discriminator.encoder.use_checkpoint = False

                    return g_loss.item(), g_recon.item(), d_loss_last, gp_last, lam

                # ── Tentativa 1: sem gradient checkpointing ────────────────────
                try:
                    g_v, r_v, d_v, gp_v, curr_lambda = _train_batch(use_ckpt=False)

                except RuntimeError as oom1:
                    if 'out of memory' not in str(oom1).lower():
                        raise

                    # ── Tentativa 2: gradient checkpointing ativado ────────────
                    torch.cuda.empty_cache()
                    log.warning(
                        f"OOM em {N_pts:,} pts → ativando gradient checkpointing..."
                    )
                    try:
                        g_v, r_v, d_v, gp_v, curr_lambda = _train_batch(use_ckpt=True)
                        log.info(f"  Retentativa OK ({N_pts:,} pts com checkpointing)")

                    except RuntimeError as oom2:
                        if 'out of memory' not in str(oom2).lower():
                            raise
                        # Nuvem genuinamente não cabe — pular sem interromper o treino
                        torch.cuda.empty_cache()
                        log.error(
                            f"OOM irrecuperável: {N_pts:,} pts mesmo com checkpointing"
                            f" — nuvem ignorada nesta epoch."
                        )
                        n_skip += 1
                        continue

                ep_d.append(d_v)
                ep_gp.append(gp_v)
                ep_g.append(g_v)
                ep_r.append(r_v)

                # ── Memory bank: coletar features normais ─────────────────────
                if memory_bank is not None and epoch >= phase1_end:
                    with torch.no_grad():
                        x_flat    = real.reshape(-1, real.size(-1))
                        bottleneck, _ = generator.encoder(x_flat)
                    memory_bank.update(bottleneck.cpu())

                batch_times.append(time.time() - batch_t)

            except RuntimeError as e:
                log.warning(f"Batch erro inesperado: {e}")
                torch.cuda.empty_cache()
                n_skip += 1
                continue

        if not ep_g:
            log.warning(f"Epoch {epoch+1}: sem batches válidos!")
            continue

        # ── Médias da epoch (logging correto: mean não soma) ──────────────────
        avg_g    = float(np.mean(ep_g))
        avg_d    = float(np.mean(ep_d))
        avg_r    = float(np.mean(ep_r))
        avg_gp   = float(np.mean(ep_gp))
        elapsed  = time.time() - epoch_start

        history['g_loss'].append(avg_g)
        history['d_loss'].append(avg_d)
        history['recon_loss'].append(avg_r)
        history['gp'].append(avg_gp)
        history['lr_g'].append(opt_G.param_groups[0]['lr'])
        history['lr_d'].append(opt_D.param_groups[0]['lr'])
        history['lambda_recon'].append(curr_lambda)

        # ── Health monitoring ─────────────────────────────────────────────────
        health = check_training_health(avg_g, avg_d, avg_r, epoch)
        h_score = sum(health[k] for k in ('g_ok', 'd_ok', 'ratio_ok', 'recon_ok'))
        history['health_score'].append(h_score)

        phase = "P1" if epoch < phase1_end else "P2"
        vram  = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        log.info(
            f"Epoch {epoch+1:03d}/{num_epochs} [{phase}] | "
            f"G={avg_g:.4f} D={avg_d:.4f} R={avg_r:.4f} GP={avg_gp:.3f} | "
            f"λ={curr_lambda:.3f} | Health={h_score}/4 | "
            f"VRAM={vram:.1f}GB | {elapsed/60:.1f}min"
        )
        if not health['healthy']:
            log.warning(f"  ⚠️  {' | '.join(health['issues'])}")

        sched_G.step()
        sched_D.step()

        # ── Salvar melhor modelo ──────────────────────────────────────────────
        if avg_g < best_g:
            best_g, best_epoch = avg_g, epoch
            ckpt = {
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'opt_G': opt_G.state_dict(),
                'opt_D': opt_D.state_dict(),
                'history': history,
                'best_g': best_g,
                'timestamp': datetime.now().isoformat(),
            }
            torch.save(ckpt, os.path.join(save_dir, 'best_model.pth'))
            log.info(f"   💾 Melhor salvo (epoch {epoch+1}, G={best_g:.4f})")

        if (epoch + 1) % 10 == 0:
            torch.save({'epoch': epoch, 'generator': generator.state_dict(),
                        'discriminator': discriminator.state_dict(), 'history': history},
                       os.path.join(save_dir, f'checkpoint_ep{epoch+1:03d}.pth'))

        gc.collect()
        torch.cuda.empty_cache()

        if early_stop and early_stop(epoch, avg_g, avg_d, avg_r):
            log.info(f"Early stop no epoch {epoch+1}")
            break

    log.info(f"\nTreino concluído! Melhor epoch: {best_epoch+1} (G={best_g:.4f})")
    return generator, discriminator, history


# ============================================================================
# INFERÊNCIA / DETECÇÃO DE ANOMALIAS  (versão melhorada)
# ============================================================================
# Score composto = reconstrução + distância ao banco de memória
# [R3D-AD — Zhou et al., 2024 | Uni-3DAD — Liu et al., 2024 | PatchCore — Roth et al., 2022]
# ============================================================================

@torch.no_grad()
def compute_anomaly_scores(generator: nn.Module,
                            data_list: list,
                            device: torch.device,
                            memory_bank: NormalMemoryBank = None,
                            alpha_recon: float = 0.7,
                            alpha_bank:  float = 0.3) -> list:
    """
    Score de anomalia composto por dois termos:

    score(p) = α_recon · recon_error(p)  +  α_bank · bank_dist(p)

    Onde:
      recon_error(p) = ||x_p - G(x_p)||²     (por ponto, normalizado)
      bank_dist(p)   = distância coseno média aos k vizinhos no banco de features

    Justificativa:
      - recon_error captura deformações geométricas grandes (buracos, rupturas)
      - bank_dist captura feições fora da distribuição normal aprendida,
        mesmo quando a reconstrução é parcialmente bem-sucedida
        [Uni-3DAD — Liu et al., 2024]

    Se memory_bank=None, usa apenas recon_error (alpha_recon=1.0).
    """
    generator.eval()
    results = []

    for d in data_list:
        features = torch.tensor(
            d['features'], dtype=torch.float32).unsqueeze(0).to(device)
        labels   = d['labels']

        # ── Reconstrução ──────────────────────────────────────────────────────
        with autocast():
            recon = generator(features)          # (1, N, 15)

        recon_err = (recon - features).pow(2).mean(dim=-1).squeeze(0).cpu().numpy()
        # Normalizar intra-nuvem
        lo, hi = recon_err.min(), recon_err.max()
        recon_norm = (recon_err - lo) / (hi - lo + 1e-8)

        # ── Distância ao banco de memória ─────────────────────────────────────
        if memory_bank is not None and memory_bank.bank is not None:
            x_flat    = features.reshape(-1, features.size(-1))
            bottleneck, _ = generator.encoder(x_flat)   # (N, 512)
            bank_dist  = memory_bank.score(bottleneck.cpu(), k=5)
            # Normalizar
            lo2, hi2   = bank_dist.min(), bank_dist.max()
            bank_norm  = (bank_dist - lo2) / (hi2 - lo2 + 1e-8)
            score      = alpha_recon * recon_norm + alpha_bank * bank_norm
        else:
            score = recon_norm

        results.append({
            'filename'   : d['filename'],
            'has_crack'  : d['has_crack'],
            'recon_norm' : recon_norm,
            'score'      : score,    # score composto — usado para threshold
            'gt_labels'  : labels,
            'n_points'   : len(labels),
        })

    return results


# ============================================================================
# THRESHOLD: GMM — mais defensável que percentil fixo
# ============================================================================
# Fraqueza do percentil fixo (e.g. P85): assume que exatamente 15% dos
# pontos são anômalos em QUALQUER nuvem, independente do conteúdo.
# GMM modela a distribuição bimodal (normal + anômalo) e encontra a
# fronteira de decisão de forma orientada por dados.
# Referência: McLachlan & Peel (2000) — Finite Mixture Models.
# ============================================================================

def plot_training_history(history: dict, save_dir: str = VIS_PATH, ts: str = None):
    os.makedirs(save_dir, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')
    epochs = range(1, len(history['g_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('GAN v5 — Histórico de Treinamento', fontsize=14)

    axes[0,0].plot(epochs, history['g_loss'],    label='G', color='blue')
    axes[0,0].plot(epochs, history['d_loss'],    label='D', color='red')
    axes[0,0].set_title('Wasserstein Loss'); axes[0,0].legend(); axes[0,0].grid(True)
    axes[0,0].axhline(0, color='k', linestyle='--', linewidth=0.5)

    axes[0,1].plot(epochs, history['recon_loss'], color='green')
    axes[0,1].set_title('Reconstruction Loss (MSE)'); axes[0,1].grid(True)

    axes[0,2].plot(epochs, history['gp'], color='orange')
    axes[0,2].set_title('Gradient Penalty'); axes[0,2].grid(True)

    axes[1,0].plot(epochs, history['lr_g'], label='G'); axes[1,0].plot(epochs, history['lr_d'], label='D')
    axes[1,0].set_title('Learning Rate'); axes[1,0].legend(); axes[1,0].grid(True)

    if 'lambda_recon' in history:
        axes[1,1].plot(epochs, history['lambda_recon'], color='purple')
        axes[1,1].set_title('λ_recon Adaptativo'); axes[1,1].grid(True)

    if 'health_score' in history:
        axes[1,2].plot(epochs, history['health_score'], color='teal', marker='.')
        axes[1,2].set_ylim(-0.1, 4.5)
        axes[1,2].axhline(4, color='green', linestyle='--', label='Saudável')
        axes[1,2].set_title('Health Score (0-4)'); axes[1,2].legend(); axes[1,2].grid(True)

    plt.tight_layout()
    path = os.path.join(save_dir, f'training_history_{ts}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Histórico salvo: {path}")


def plot_error_distribution(results: list, thr: float,
                             gmm_info: dict = None, save_dir: str = VIS_PATH,
                             ts: str = None):
    os.makedirs(save_dir, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')
    crack_res  = [r for r in results if r['has_crack']]
    normal_res = [r for r in results if not r['has_crack']]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribuição do Score de Anomalia', fontsize=13)

    ax = axes[0]
    if normal_res:
        ns = np.concatenate([r['score'] for r in normal_res])
        ax.hist(ns, bins=100, alpha=0.6, label='Nuvens Normais', color='blue', density=True)
    if crack_res:
        gt_pos = np.concatenate([r['score'][r['gt_labels'] == 1] for r in crack_res if r['gt_labels'].sum() > 0])
        gt_neg = np.concatenate([r['score'][r['gt_labels'] == 0] for r in crack_res])
        ax.hist(gt_neg, bins=100, alpha=0.5, label='Pts Normais (GT)', color='orange', density=True)
        if len(gt_pos): ax.hist(gt_pos, bins=100, alpha=0.5, label='Pts Rachadura (GT)', color='red', density=True)
    ax.axvline(thr, color='black', linestyle='--', linewidth=2,
               label=f"Thr={thr:.3f} ({gmm_info.get('method','?') if gmm_info else '?'})")
    ax.set_xlabel('Score de Anomalia Composto'); ax.set_ylabel('Densidade')
    ax.legend(fontsize=8); ax.grid(True)

    ax2 = axes[1]
    if crack_res:
        gt    = np.concatenate([r['gt_labels'] for r in crack_res])
        score = np.concatenate([r['score']     for r in crack_res])
        try:
            from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
            fpr, tpr, _ = roc_curve(gt, score)
            auroc = roc_auc_score(gt, score)
            ax2.plot(fpr, tpr, lw=2, label=f'ROC (AUROC={auroc:.3f})', color='blue')
            ax2.plot([0,1],[0,1],'k--'); ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR')
            ax2.set_title('Curva ROC'); ax2.legend(); ax2.grid(True)
        except Exception:
            pass

    plt.tight_layout()
    path = os.path.join(save_dir, f'anomaly_distribution_{ts}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f"Distribuição salva: {path}")


def save_results(metrics: dict, results: list, history: dict,
                 thr: float, gmm_info: dict, save_dir: str = RESULTS,
                 ts: str = None):
    os.makedirs(save_dir, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')

    def _to_py(v):
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    metrics_out = {k: _to_py(v) for k, v in metrics.items() if k != 'per_cloud'}
    metrics_out.update({'threshold': thr, 'gmm': gmm_info,
                        'timestamp': datetime.now().isoformat()})

    with open(os.path.join(save_dir, f'metrics_{ts}.json'), 'w') as f:
        json.dump(metrics_out, f, indent=2, default=str)

    # CSV por ponto
    rows = []
    for r in results:
        for i, (gt, pred, sc) in enumerate(
                zip(r['gt_labels'], r['pred_labels'], r['score'])):
            rows.append({'filename': r['filename'], 'point_idx': i,
                         'gt': int(gt), 'pred': int(pred), 'score': float(sc)})
    pd.DataFrame(rows).to_csv(
        os.path.join(save_dir, f'predictions_{ts}.csv'), index=False)

    pd.DataFrame(history).to_csv(
        os.path.join(save_dir, f'training_history_{ts}.csv'), index=False)
    log.info(f"Resultados salvos em {save_dir} (ts={ts})")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M')
    print("\n" + "="*70)
    print("  GAN HÍBRIDA v5 — DETECÇÃO DE AVARIAS NÃO-SUPERVISIONADA")
    print("  KPConv + PointTransformer | WGAN-GP | Memory Bank | GMM Threshold")
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
        log.error("Nenhum dado carregado. Verifique DATA_TRAIN e DATA_TEST.")
        return

    train_list, _, eval_list = split_dataset(all_data)
    train_dl, _, eval_dl    = make_loaders(train_list, eval_list)

    # ── 2. Modelos ────────────────────────────────────────────────────────────
    generator, discriminator = build_models(device)

    # ── 3. Memory bank ────────────────────────────────────────────────────────
    memory_bank = NormalMemoryBank(max_size=100_000, subsample_k=500)

    # ── 4. Treino ─────────────────────────────────────────────────────────────
    log.info("\nIniciando treinamento WGAN-GP...")
    t0 = time.time()

    generator, discriminator, history = train_wgan(
        generator, discriminator, train_dl, device,
        num_epochs=NUM_EPOCHS,
        lr_g=LR_G, lr_d=LR_D,
        lambda_gp=LAMBDA_GP,
        lambda_recon=LAMBDA_RECON,
        use_early_stopping=True,
        memory_bank=memory_bank,
        save_dir=MODELS,
    )

    log.info(f"\nTreino: {(time.time()-t0)/3600:.1f}h")
    plot_training_history(history, ts=ts)

    # ── 5. Carregar melhor modelo ─────────────────────────────────────────────
    best_path = os.path.join(MODELS, 'best_model.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        generator.load_state_dict(ckpt['generator'])
        log.info(f"Melhor modelo carregado (epoch {ckpt['epoch']+1})")

    # Salvar memory bank
    memory_bank.save(os.path.join(MODELS, 'memory_bank.pt'))

    # ── 6. Avaliação com score composto ───────────────────────────────────────
    log.info("\nComputando anomaly scores...")
    eval_data = eval_list if eval_list else train_list

    results = compute_anomaly_scores(
        generator, eval_data, device,
        memory_bank=memory_bank,
        alpha_recon=0.7, alpha_bank=0.3,
    )

    # Scores das nuvens normais de treino — usadas para calibrar GMM
    log.info("Computando scores de referência (nuvens normais)...")
    normal_scores = compute_anomaly_scores(
        generator, train_list[:min(20, len(train_list))],
        device, memory_bank=memory_bank)

    # ── 7. Threshold GMM ──────────────────────────────────────────────────────
    thr, gmm_info = fit_gmm_threshold(results, normal_results=normal_scores)
    log.info(f"Threshold ({gmm_info['method']}): {thr:.4f}")
    results = apply_threshold(results, thr)

    # Verificação sanidade: nuvens normais devem ter baixo % de "anomalias"
    normal_check = apply_threshold(normal_scores, thr)
    for r in normal_check[:5]:
        pct = r['pred_labels'].mean() * 100
        status = "✅" if pct < 20 else "⚠️"
        log.info(f"  Sanidade {r['filename']}: {pct:.1f}% anomalias {status}")

    # ── 8. Métricas ───────────────────────────────────────────────────────────
    metrics = evaluate(results)

    # ── 9. Visualizações + Salvamento ─────────────────────────────────────────
    plot_error_distribution(results, thr, gmm_info, ts=ts)
    save_results(metrics, results, history, thr, gmm_info, ts=ts)

    # ── Resumo final ──────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTADOS FINAIS")
    print("="*70)
    if metrics:
        print(f"  F1-Score : {metrics.get('f1',  0):.4f}  (target ≥ 0.75)")
        print(f"  IoU      : {metrics.get('iou', 0):.4f}  (target ≥ 0.60)")
        print(f"  AUROC    : {metrics.get('auroc', 0):.4f}  (target ≥ 0.85)")
        print(f"  AP       : {metrics.get('average_precision', 0):.4f}")
    print(f"\n  Threshold ({gmm_info['method']}): {thr:.4f}")
    print(f"  Resultados em : {RESULTS}")
    print("="*70)

    return generator, discriminator, history, results, metrics


if __name__ == '__main__':
    main()
