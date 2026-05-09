# ============================================================================
# SCALAR FIELD INTERVAL CLASSIFIER — CAMINHO B v3
# ============================================================================
#
# Hipótese:
#   A geometria local (XYZ + normais) carrega informação suficiente para
#   prever em qual intervalo do SF o ponto deveria estar (normal vs. crack).
#   Score de anomalia = discordância entre predição geométrica e SF real.
#
# Diferença vs. scalar_field_mae.py (v2):
#   MAE (v2): regressão → score = |SF_pred - SF_real|
#   Aqui   : classificação de intervalo → score = p_geom_normal * p_sf_crack
#
# Por que classificação de intervalo é mais adequada:
#   1. O SF é bimodal — o sinal é "em qual componente?", não "qual valor exato?"
#   2. A amplitude exata do SF varia entre nuvens (calibração do scanner).
#      O intervalo (componente GMM) é mais estável entre sessões.
#   3. BCE é melhor calibrado para o objetivo binário do que Huber/MSE.
#
# Arquitetura — LocalSFIntervalClassifier:
#   Input  : XYZ + normais (6D) — RGB e scalar_field EXCLUÍDOS
#   MLP    : 6D → 64D  (features por ponto)
#   EdgeConv: 64D → 128D  (contexto local, kNN em XYZ)
#   Global : max-pool + MLP → 64D
#   Head   : Concat(128+64) → 64D → 1D → Sigmoid
#   Output : P(SF ∈ componente_normal | geometria) ∈ [0, 1]
#
# Targets de treino (gerados por ScalarFieldGMM):
#   p_sf_normal(i) = 1 − GMM.anomaly_probability(i)
#   Soft labels: pontos claramente normais → ~0.95; claramente crack → ~0.05
#   Treino APENAS em nuvens sem rachadura (n_avaria_*).
#
# Score de anomalia por ponto:
#   score(i) = p_geom_normal(i) × p_sf_crack(i)
#            = p_geom_normal × GMM.anomaly_probability
#   Alto quando: geometria prevê "normal" MAS SF real está no intervalo de crack.
#   Baixo quando: ambos concordam (ambos normais OU geometria prevê crack).
#
# Referências:
#   He et al. (2022) — MAE: Masked Autoencoders Are Scalable Vision Learners.
#   Wang et al. (2019) — DGCNN: Dynamic Graph CNN.
#   Reynolds (2009) — Gaussian Mixture Models.
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import copy, gc, time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast

import pandas as pd

from utils import *
from utils.config import *
from utils.data import *
from utils.evaluation import (
    ScalarFieldGMM,
    fit_gmm_threshold, apply_threshold, evaluate,
    calibrate_threshold_f1, calibrate_threshold_gmean, calibrate_threshold_fbeta,
    save_colored_ply,
)
from utils.training_utils import *
from teacher_student_v3 import (
    _xyz_knn, spatial_smooth_scores, _boundary_mask,
    SMOOTH_K, BOUNDARY_RADIUS, BOUNDARY_MIN_NB,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
RESULTS_DIR = f'{BASE_PATH}/results_sf_interval'
MODELS_DIR  = f'{BASE_PATH}/models_sf_interval'
VIS_DIR     = f'{BASE_PATH}/visualizations_sf_interval'
PLY_DIR     = f'{BASE_PATH}/results_sf_interval/ply'
MODEL_CKP   = f'{MODELS_DIR}/sf_interval_classifier.pth'

log = setup_logging(f'{BASE_PATH}/logs_sf_interval')

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
K_EDGE    = 16
LR        = 3e-4
EPOCHS    = 100
PATIENCE  = 15
CHUNK     = 8_000   # pontos por chunk no EdgeConv

# Input: XYZ (0-2) + normais (6-8) — 6D
# RGB excluído: proxy direto de SF → leakage
# Scalar_field excluído: é o sinal que queremos prever
INPUT_COLS   = [0, 1, 2, 6, 7, 8]
INPUT_DIM    = len(INPUT_COLS)   # 6


# ============================================================================
# UTILITÁRIOS
# ============================================================================

def compute_gmm_targets(sf_raw: np.ndarray) -> np.ndarray:
    """
    Gera soft labels P(SF ∈ componente_normal) via ScalarFieldGMM.

    Retorna vetor em [0, 1] por ponto:
      ~1.0 → ponto quase certamente no intervalo normal (SF alto)
      ~0.0 → ponto quase certamente no intervalo de crack (SF baixo)
    """
    gmm = ScalarFieldGMM(sf_raw).fit()
    p_crack = gmm.anomaly_probability()   # (N,) ∈ [0, 1]
    return (1.0 - p_crack).astype(np.float32)


# ============================================================================
# EDGE CONV CHUNKED — memória eficiente
# ============================================================================

def _edge_conv_chunked(
    h: torch.Tensor,
    knn_idx: np.ndarray,
    mlp: nn.Module,
    chunk: int = CHUNK,
) -> torch.Tensor:
    """
    EdgeConv em chunks para evitar OOM em nuvens grandes.

    Para cada ponto i e k vizinhos j:
      edge_feat(i,j) = [h_i || h_j − h_i]
      h_i_new = max_{j} MLP(edge_feat)
    """
    N, C_in = h.shape
    k       = knn_idx.shape[1]
    C_out   = next(m.out_features for m in reversed(list(mlp.modules()))
                   if isinstance(m, nn.Linear))
    out = torch.zeros(N, C_out, device=h.device, dtype=h.dtype)

    for s in range(0, N, chunk):
        e   = min(s + chunk, N)
        cs  = e - s
        idx = torch.from_numpy(knn_idx[s:e].astype(np.int64)).to(h.device)

        neighbors = h[idx.view(-1)].view(cs, k, C_in)
        h_i       = h[s:e].unsqueeze(1).expand(-1, k, -1)
        edge_feat = torch.cat([h_i, neighbors - h_i], dim=-1)   # (cs*k, 2*C_in)

        out_chunk = mlp(edge_feat.reshape(cs * k, -1))
        out[s:e]  = out_chunk.view(cs, k, C_out).max(dim=1).values

    return out


# ============================================================================
# LOCAL SF INTERVAL CLASSIFIER
# ============================================================================

class LocalSFIntervalClassifier(nn.Module):
    """
    Classifica se a geometria local indica SF no intervalo normal ou de crack.

    Mesma arquitetura do LocalSFPredictor (EdgeConv + global context),
    mas com target binário soft (GMM) e BCELoss em vez de regressão + HuberLoss.

    Output: P(SF ∈ intervalo_normal | geometria) ∈ [0, 1]
    """

    def __init__(self, k: int = K_EDGE):
        super().__init__()
        self.k = k

        self.point_mlp = nn.Sequential(
            nn.LayerNorm(INPUT_DIM),
            nn.Linear(INPUT_DIM, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.edge_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        self.global_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Saída: logit → Sigmoid aplicado externamente (BCEWithLogits no treino)
        self.pred_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, knn_idx: np.ndarray) -> torch.Tensor:
        """
        x      : (N, 6)  — XYZ + normais
        knn_idx: (N, k)  — índices k-NN em XYZ (numpy, pré-computados)
        Returns: (N,) — logits (usar sigmoid para probabilidade)
        """
        h        = self.point_mlp(x)
        h_local  = _edge_conv_chunked(h, knn_idx, self.edge_mlp)
        h_global = self.global_mlp(h_local.max(dim=0).values)
        h_global = h_global.unsqueeze(0).expand(x.size(0), -1)
        h_cat    = torch.cat([h_local, h_global], dim=1)
        return self.pred_head(h_cat).squeeze(-1)


# ============================================================================
# TREINAMENTO
# ============================================================================

def train_interval_classifier(
    model: LocalSFIntervalClassifier,
    all_data: list,
    device: torch.device,
    num_epochs: int = EPOCHS,
    lr: float = LR,
    save_path: str = MODEL_CKP,
) -> LocalSFIntervalClassifier:
    """
    Treina em nuvens normais apenas.

    Para cada nuvem de treino:
      1. Calcula ScalarFieldGMM(sf_raw) → target = P(SF ∈ normal)
      2. Loss = BCEWithLogits(logits, target)

    O modelo aprende quais geometrias correspondem ao intervalo normal de SF.
    Em inferência, discordância com o GMM do SF real indica rachadura.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    normal_data = [d for d in all_data if not d.get('has_crack', True)]
    if len(normal_data) < 2:
        log.warning(f"Apenas {len(normal_data)} nuvens normais — usando todas")
        normal_data = all_data

    n_val   = max(1, len(normal_data) // 5)
    train_d = normal_data[n_val:]
    val_d   = normal_data[:n_val]

    log.info(f"\n{'='*65}")
    log.info("SF INTERVAL CLASSIFIER — Treino (Caminho B v3)")
    log.info(f"Treino: {len(train_d)} nuvens normais | Val: {len(val_d)} nuvens")
    log.info(f"Epochs: {num_epochs} | LR: {lr} | k={K_EDGE}")
    log.info(f"{'='*65}")

    model     = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr,
                      betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = GradScaler()

    log.info("Cacheando kNN XYZ e targets GMM...")
    train_cache = []
    for d in train_d:
        knn = _xyz_knn(d['features'][:, :3], K_EDGE)
        tgt = compute_gmm_targets(d['features'][:, 9])
        train_cache.append((knn, tgt))

    val_cache = []
    for d in val_d:
        knn = _xyz_knn(d['features'][:, :3], K_EDGE)
        tgt = compute_gmm_targets(d['features'][:, 9])
        val_cache.append((knn, tgt))

    log.info("Cache pronto.")

    best_val_loss = float('inf')
    patience_c    = 0

    for epoch in range(num_epochs):
        model.train()
        ep_losses = []
        perm = np.random.permutation(len(train_d))

        for i in perm:
            d          = train_d[i]
            knn, tgt   = train_cache[i]
            x          = torch.tensor(d['features'][:, INPUT_COLS],
                                      dtype=torch.float32).to(device)
            target     = torch.tensor(tgt, dtype=torch.float32).to(device)

            if x.size(0) < K_EDGE + 1:
                continue

            optimizer.zero_grad(set_to_none=True)
            try:
                with autocast():
                    logits = model(x, knn)
                    loss   = F.binary_cross_entropy_with_logits(logits, target)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                ep_losses.append(loss.item())

            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    torch.cuda.empty_cache()
                    continue
                raise

        scheduler.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, d in enumerate(val_d):
                knn, tgt = val_cache[i]
                x        = torch.tensor(d['features'][:, INPUT_COLS],
                                        dtype=torch.float32).to(device)
                target   = torch.tensor(tgt, dtype=torch.float32).to(device)
                if x.size(0) < K_EDGE + 1:
                    continue
                with autocast():
                    logits = model(x, knn)
                    val_losses.append(
                        F.binary_cross_entropy_with_logits(logits, target).item())

        if not ep_losses:
            continue

        avg_train = float(np.mean(ep_losses))
        avg_val   = float(np.mean(val_losses)) if val_losses else float('nan')

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch+1:03d}/{num_epochs} | "
                     f"Train={avg_train:.5f} | Val={avg_val:.5f} | "
                     f"LR={optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_c    = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': best_val_loss}, save_path)
        else:
            patience_c += 1

        if patience_c >= PATIENCE:
            log.info(f"Early stop: {PATIENCE} epochs sem melhoria (val={avg_val:.5f})")
            break

        gc.collect()
        torch.cuda.empty_cache()

    log.info(f"\nTreino concluído. Melhor val loss: {best_val_loss:.5f}")

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        log.info(f"Melhor modelo carregado (epoch {ckpt['epoch']+1})")

    return model


# ============================================================================
# INFERÊNCIA — ANOMALY SCORES
# ============================================================================

@torch.no_grad()
def compute_interval_anomaly_scores(
    model: LocalSFIntervalClassifier,
    data_list: list,
    device: torch.device,
) -> list:
    """
    Score de anomalia por ponto = p_geom_normal × p_sf_crack.

    p_geom_normal : o modelo prevê que a geometria corresponde a SF normal
    p_sf_crack    : o GMM diz que o SF real está no intervalo de crack

    Discordância alta → ponto com geometria de superfície normal mas SF de crack.
    """
    model.eval()
    results = []

    for d in data_list:
        sf_raw = d['features'][:, 9]
        xyz_np = d['features'][:, :3]

        x   = torch.tensor(d['features'][:, INPUT_COLS],
                           dtype=torch.float32).to(device)
        knn = _xyz_knn(xyz_np, K_EDGE)

        with autocast():
            logits = model(x, knn).cpu().numpy().astype(np.float32)

        p_geom_normal = 1.0 / (1.0 + np.exp(-logits))   # sigmoid

        # GMM no SF real desta nuvem
        gmm        = ScalarFieldGMM(sf_raw).fit()
        p_sf_crack = gmm.anomaly_probability().astype(np.float32)

        # Score de discordância: geometria diz normal MAS SF diz crack
        score_raw = p_geom_normal * p_sf_crack

        # Pós-processamento idêntico ao scalar_field_mae
        score = spatial_smooth_scores(xyz_np, score_raw, k=SMOOTH_K)
        bnd   = _boundary_mask(xyz_np, radius=BOUNDARY_RADIUS,
                               min_neighbors=BOUNDARY_MIN_NB)
        score[bnd] = 0.0

        lo, hi = np.percentile(score, 2), np.percentile(score, 98)
        score  = np.clip((score - lo) / (hi - lo + 1e-8), 0, 1)

        results.append({
            'filename'      : d['filename'],
            'has_crack'     : d['has_crack'],
            'score'         : score,
            'gt_labels'     : d['labels'],
            'n_points'      : len(d['labels']),
            'xyz'           : xyz_np,
            'rgb'           : d['features'][:, 3:6],
            'scalar_field'  : sf_raw,
            'p_geom_normal' : p_geom_normal,
            'p_sf_crack'    : p_sf_crack,
        })

        del x
        torch.cuda.empty_cache()

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print("  SF INTERVAL CLASSIFIER — CAMINHO B v3")
    print("  Score = P(geom→normal) × P(SF real é crack)  [discordância]")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    run_results = os.path.join(RESULTS_DIR, f'run_{ts}')
    run_ply     = os.path.join(PLY_DIR,     f'run_{ts}')

    for p in [run_results, run_ply, MODELS_DIR]:
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

    _, _, eval_list = split_dataset(all_data)
    eval_data = eval_list if eval_list else train_all

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    model    = LocalSFIntervalClassifier(k=K_EDGE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"LocalSFIntervalClassifier: {n_params:,} parâmetros")

    # ── 3. Treino ─────────────────────────────────────────────────────────────
    if os.path.exists(MODEL_CKP):
        log.info(f"Checkpoint encontrado: {MODEL_CKP} — carregando...")
        ckpt = torch.load(MODEL_CKP, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
    else:
        log.info("\nIniciando treino...")
        t0    = time.time()
        model = train_interval_classifier(
            model, all_data, device,
            num_epochs=EPOCHS, lr=LR, save_path=MODEL_CKP,
        )
        log.info(f"Treino: {(time.time()-t0)/3600:.1f}h")

    # ── 4. Scores de anomalia ─────────────────────────────────────────────────
    log.info("\nComputando anomaly scores (discordância geom × SF-GMM)...")
    results = compute_interval_anomaly_scores(model, eval_data, device)

    # ── 5. Threshold + métricas ───────────────────────────────────────────────
    normal_ref = compute_interval_anomaly_scores(
        model,
        [d for d in train_all if not d.get('has_crack')][:10],
        device,
    )
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
    best_f1       = -1.0
    best_name     = None

    log.info("\n" + "="*65)
    log.info("COMPARAÇÃO DE ESTRATÉGIAS DE THRESHOLD")
    log.info("="*65)

    for name, thr_s in strategies.items():
        res_s = apply_threshold(copy.deepcopy(results), thr_s)
        m_s   = evaluate(res_s)
        row   = {
            'estrategia'   : name,
            'threshold'    : round(thr_s, 6),
            'precision'    : round(m_s.get('precision',         0), 4),
            'recall'       : round(m_s.get('recall',            0), 4),
            'f1'           : round(m_s.get('f1',                0), 4),
            'iou'          : round(m_s.get('iou',               0), 4),
            'auroc'        : round(m_s.get('auroc',             0), 4),
            'avg_precision': round(m_s.get('average_precision', 0), 4),
        }
        comparison.append(row)
        log.info(f"\n  [{name}]  thr={thr_s:.4f}")
        log.info(f"    P={row['precision']:.4f}  R={row['recall']:.4f}  "
                 f"F1={row['f1']:.4f}  IoU={row['iou']:.4f}  "
                 f"AUROC={row['auroc']:.4f}")

        if name == 'G-mean':
            results_gmean = res_s
            metrics_gmean = m_s
        if row['f1'] > best_f1:
            best_f1   = row['f1']
            best_name = name

    log.info(f"\n  Melhor F1: [{best_name}] {best_f1:.4f}")
    log.info("="*65)

    results = results_gmean
    metrics = metrics_gmean

    pd.DataFrame(comparison).to_csv(
        os.path.join(run_results, f'comparacao_{ts}.csv'), index=False)

    # ── 6. PLY coloridos ──────────────────────────────────────────────────────
    log.info("\nSalvando PLY coloridos...")
    for r in results:
        if not r['has_crack']:
            continue
        save_colored_ply(
            xyz=r['xyz'], rgb_orig=r['rgb'],
            pred_labels=r['pred_labels'],
            path=os.path.join(run_ply,
                              r['filename'].replace('.ply', '_sfinterval.ply')))

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTADOS — SF INTERVAL CLASSIFIER (Caminho B v3)")
    print("="*70)
    if metrics:
        print(f"  F1    : {metrics.get('f1',    0):.4f}")
        print(f"  IoU   : {metrics.get('iou',   0):.4f}")
        print(f"  AUROC : {metrics.get('auroc', 0):.4f}")
    print(f"  Threshold: {thr:.4f} | Resultados em: {run_results}")
    print("="*70)

    return model, results, metrics


if __name__ == '__main__':
    main()
