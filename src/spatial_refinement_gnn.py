# ============================================================================
# SPATIAL REFINEMENT GNN — Refinamento espacial de scores de anomalia
# ============================================================================
#
# Motivação (síntese das falhas anteriores):
#
#   O SF-GMM (AUROC=0.887) já identifica quais pontos provavelmente são
#   rachadura. O problema remanescente no v1 é ALOCAÇÃO ESPACIAL:
#   pontos isolados com SF baixo (sombra, sujeira) recebem score alto
#   junto com rachaduras reais. O modelo não distingue clusters coerentes
#   de rachaduras de ruído isolado.
#
#   Este GNN aprende "como rachaduras se organizam no espaço": estruturas
#   conectadas, estreitas e contínuas — não pontos isolados.
#
# O que cada falha contribuiu:
#   scalar_field_unsup  → kNN DEVE ser em XYZ (não espaço de features)
#                          EdgeConv é o bloco arquitetural correto
#   v3 geom pretraining → curvatura/densidade/variância como INPUT direto
#   v3 adapter          → Teacher raw consistency (AUROC=0.686) como feature
#   Caminho B (SF MAE)  → SF-GMM pseudo-labels (AUROC=0.887) como supervisão
#
# Arquitetura:
#   Entrada: 12D por ponto (SF-GMM, SF_raw, RGB, normais, geom features)
#   Grafo  : kNN em XYZ (k=16) — nunca em espaço de features
#   Rede   : 2× EdgeConv → global max-pool → head → score refinado ∈ [0,1]
#
# Treino não-supervisionado:
#   Pseudo-labels do SF-GMM com alta confiança:
#     prob > PSEUDO_THR_POS → pseudo-crack     (1)
#     prob < PSEUDO_THR_NEG → pseudo-normal    (0)
#     intermediário         → ignorado
#   Loss: BCE sobre pseudo-labels + suavização TV opcional
#
# Referências:
#   Wang et al. (2019) — DGCNN, ACM TOG.
#   Veličković et al. (2018) — GAT, ICLR.
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import csv, copy, gc, time
from datetime import datetime
from scipy.spatial import cKDTree

from utils import *
from utils.config import *
from utils.data import *
from utils.evaluation import *
from utils.training_utils import *
from teacher_student_v3 import (
    _xyz_knn, spatial_smooth_scores, _boundary_mask,
    SMOOTH_K, BOUNDARY_RADIUS, BOUNDARY_MIN_NB,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
RESULTS_GNN = f'{BASE_PATH}/results_gnn'
MODELS_GNN  = f'{BASE_PATH}/models_gnn'
VIS_GNN     = f'{BASE_PATH}/visualizations_gnn'
PLY_GNN     = f'{BASE_PATH}/results_gnn/ply'
MODEL_CKP   = f'{MODELS_GNN}/spatial_gnn.pth'

log = setup_logging(f'{BASE_PATH}/logs_gnn')

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
K_GNN         = 16      # vizinhos XYZ para o grafo
LR_GNN        = 3e-4
EPOCHS_GNN    = 80
PATIENCE_GNN  = 15
CHUNK_GNN     = 8_000   # pontos por chunk no EdgeConv

PSEUDO_THR_POS = 0.85   # SF-GMM prob > 0.85 → pseudo-crack
PSEUDO_THR_NEG = 0.10   # SF-GMM prob < 0.10 → pseudo-normal

# Índices das features no vetor 16D para o GNN (excluindo XYZ)
# [SF_raw/255, R, G, B, nx, ny, nz, curv, dens, var, sv] = 11D
# + SF_GMM_score computado separadamente → total 12D
GNN_FEAT_COLS = [9, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]  # 11 colunas
GNN_INPUT_DIM = 12  # 11 + SF_GMM_score


# ============================================================================
# EDGE CONV CHUNKED (cópia local — evita imports circulares)
# ============================================================================

def _edge_conv_chunked(
    h: torch.Tensor,
    knn_idx: np.ndarray,
    mlp: nn.Module,
    chunk: int = CHUNK_GNN,
) -> torch.Tensor:
    """
    EdgeConv em chunks: [h_i || h_j−h_i] → MLP → max-pool sobre vizinhos.
    Processa `chunk` pontos por vez para evitar OOM em nuvens grandes.

    h      : (N, C_in) tensor GPU
    knn_idx: (N, k)    numpy int64
    Returns: (N, C_out) tensor GPU
    """
    N, C_in = h.shape
    k       = knn_idx.shape[1]
    C_out   = next(m.out_features for m in reversed(list(mlp.modules()))
                   if isinstance(m, nn.Linear))
    out     = torch.zeros(N, C_out, device=h.device, dtype=h.dtype)

    for s in range(0, N, chunk):
        e   = min(s + chunk, N)
        cs  = e - s
        idx = torch.from_numpy(knn_idx[s:e].astype(np.int64)).to(h.device)

        neighbors = h[idx.view(-1)].view(cs, k, C_in)
        h_i       = h[s:e].unsqueeze(1).expand(-1, k, -1)
        edge_feat = torch.cat([h_i, neighbors - h_i], dim=-1)  # (cs, k, 2*C_in)

        out_chunk = mlp(edge_feat.reshape(cs * k, -1))
        out[s:e]  = out_chunk.view(cs, k, -1).max(dim=1).values

    return out


# ============================================================================
# SPATIAL REFINEMENT GNN — modelo principal
# ============================================================================

class SpatialRefinementGNN(nn.Module):
    """
    GNN para refinamento espacial de scores de anomalia.

    Recebe features brutas (incluindo SF-GMM score) e aprende a distinguir
    clusters espacialmente coerentes de rachadura de pontos isolados de ruído.

    Fluxo:
      (N, 12) → EdgeConv₁ (12→32) → EdgeConv₂ (32→64)
              → global max-pool (64→32)
              → Concat(64+32=96) → Head(96→1) → score refinado ∈ [0,1]

    Grafo: kNN em XYZ (k=16). Ambos EdgeConv usam o mesmo grafo espacial.
    O grafo fixo em XYZ evita o problema do kNN em alta dimensão identificado
    em scalar_field_unsup.
    """

    def __init__(self, input_dim: int = GNN_INPUT_DIM, k: int = K_GNN):
        super().__init__()
        self.k = k

        # EdgeConv 1: [h_i || h_j−h_i] = 24D → 32D
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(2 * input_dim, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 32),            nn.BatchNorm1d(32), nn.GELU(),
        )

        # EdgeConv 2: [h_i || h_j−h_i] = 64D → 64D
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.GELU(),
            nn.Linear(64, 64), nn.BatchNorm1d(64), nn.GELU(),
        )

        # Contexto global: 64D → 32D
        self.global_mlp = nn.Sequential(
            nn.Linear(64, 32), nn.GELU(),
        )

        # Cabeça de predição: (64+32)=96D → 1D
        self.head = nn.Sequential(
            nn.Linear(96, 32), nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid(),
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
        x      : (N, 12) — features de nó (inclui SF-GMM score)
        knn_idx: (N, k)   — índices kNN em XYZ (numpy, CPU)
        Returns: (N,) — score refinado ∈ [0, 1]
        """
        h1 = _edge_conv_chunked(x, knn_idx, self.edge_mlp1)    # (N, 32)
        h2 = _edge_conv_chunked(h1, knn_idx, self.edge_mlp2)   # (N, 64)

        h_global = self.global_mlp(h2.max(dim=0).values)        # (32,)
        h_global = h_global.unsqueeze(0).expand(x.size(0), -1) # (N, 32)

        h = torch.cat([h2, h_global], dim=1)                    # (N, 96)
        return self.head(h).squeeze(-1)                          # (N,)


# ============================================================================
# EXTRAÇÃO DE FEATURES DO NÓ
# ============================================================================

def build_node_features(d: dict, sf_gmm_score: np.ndarray) -> np.ndarray:
    """
    Constrói o vetor de features 12D por ponto para o GNN.

    Dimensões:
      [0]    SF_GMM_score          — sinal primário (AUROC=0.887)
      [1]    SF_raw / 255          — valor bruto normalizado
      [2-4]  RGB                   — aparência (R, G, B)
      [5-7]  normais               — orientação de superfície (nx, ny, nz)
      [8]    curvatura             — descontinuidade de superfície
      [9]    densidade             — void interno de rachadura
      [10]   variância             — rugosidade local
      [11]   surface_variation     — variação de superfície local

    Todas as features já estão no vetor 16D do dataset (exceto SF_GMM_score).
    """
    feats = d['features']
    raw   = feats[:, GNN_FEAT_COLS].copy().astype(np.float32)
    # Col 0 (índice 9 do original) é SF_raw — normalizar para [0,1]
    raw[:, 0] = raw[:, 0] / 255.0

    # SF_GMM_score como primeira feature
    node_feats = np.concatenate([
        sf_gmm_score[:, None],   # (N, 1)
        raw,                      # (N, 11)
    ], axis=1).astype(np.float32)

    return node_feats   # (N, 12)


def compute_sf_gmm_score(d: dict) -> np.ndarray:
    """Computa SF-GMM anomaly probability para uma nuvem."""
    sf_raw = d['features'][:, 9]
    gmm    = ScalarFieldGMM(sf_raw).fit()
    return gmm.anomaly_probability()   # (N,) ∈ [0, 1]


def compute_pseudo_labels(
    sf_gmm_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gera pseudo-labels de alta confiança a partir do SF-GMM score.

    Returns:
      mask  : (N,) bool — True nos pontos pseudo-rotulados
      labels: (N,) float32 — 1.0 para crack, 0.0 para normal (ignorado fora de mask)
    """
    mask_pos = sf_gmm_score > PSEUDO_THR_POS
    mask_neg = sf_gmm_score < PSEUDO_THR_NEG
    mask     = mask_pos | mask_neg
    labels   = mask_pos.astype(np.float32)
    return mask, labels


# ============================================================================
# TREINAMENTO
# ============================================================================

def train_spatial_gnn(
    model: SpatialRefinementGNN,
    all_data: list,
    device: torch.device,
    num_epochs: int = EPOCHS_GNN,
    lr: float = LR_GNN,
    save_path: str = MODEL_CKP,
) -> SpatialRefinementGNN:
    """
    Treina SpatialRefinementGNN com pseudo-labels do SF-GMM.

    Protocolo:
      1. Pré-computa SF-GMM scores, pseudo-labels e kNN XYZ (fixos entre epochs).
      2. Para cada epoch, para cada nuvem:
           - Forward pass: GNN(features, knn) → score refinado
           - Loss: BCE sobre pontos pseudo-rotulados (alta confiança)
           - Backprop apenas pelos parâmetros do GNN
      3. Early stopping em val_loss (20% das nuvens como validação).

    Treina em TODAS as nuvens (supervisionado via pseudo-labels do SF-GMM).
    Nuvens normais (n_avaria) só têm pseudo-negativos → GNN aprende que
    superfícies normais devem ter score baixo.
    Nuvens de avaria têm pseudo-positivos e negativos → GNN aprende a separar.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n_val   = max(1, len(all_data) // 5)
    val_d   = all_data[:n_val]
    train_d = all_data[n_val:]

    log.info(f"\n{'='*65}")
    log.info("SPATIAL REFINEMENT GNN — Treino")
    log.info(f"Treino: {len(train_d)} nuvens | Val: {len(val_d)} nuvens")
    log.info(f"Epochs: {num_epochs} | LR: {lr} | k={K_GNN}")
    log.info(f"Pseudo-labels: pos>{PSEUDO_THR_POS} | neg<{PSEUDO_THR_NEG}")
    log.info(f"{'='*65}")

    model     = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr,
                      betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = GradScaler()

    # ── Pré-computação: SF-GMM, pseudo-labels, kNN (fixos entre epochs) ───────
    log.info("Pré-computando SF-GMM + pseudo-labels + kNN XYZ...")

    def _precompute(data):
        cache = []
        for d in data:
            sf_score = compute_sf_gmm_score(d)
            mask, pl = compute_pseudo_labels(sf_score)
            node_feat = build_node_features(d, sf_score)
            knn       = _xyz_knn(d['features'][:, :3], K_GNN)
            n_pos     = mask.sum() and (pl[mask] == 1).sum()
            n_neg     = mask.sum() and (pl[mask] == 0).sum()
            cache.append({
                'node_feat': node_feat, 'knn': knn,
                'mask': mask, 'labels': pl,
                'n_pos': int(n_pos), 'n_neg': int(n_neg),
                'filename': d['filename'],
            })
        return cache

    train_cache = _precompute(train_d)
    val_cache   = _precompute(val_d)

    total_pos = sum(c['n_pos'] for c in train_cache)
    total_neg = sum(c['n_neg'] for c in train_cache)
    log.info(f"Pseudo-labels treino: {total_pos:,} crack | {total_neg:,} normal")

    best_val  = float('inf')
    patience_c = 0

    for epoch in range(num_epochs):
        model.train()
        ep_losses = []
        perm = np.random.permutation(len(train_cache))

        for i in perm:
            c = train_cache[i]
            if c['mask'].sum() < 10:
                continue

            x   = torch.tensor(c['node_feat'], dtype=torch.float32).to(device)
            msk = torch.tensor(c['mask'],   dtype=torch.bool).to(device)
            lbl = torch.tensor(c['labels'], dtype=torch.float32).to(device)

            optimizer.zero_grad(set_to_none=True)
            try:
                with autocast():
                    score = model(x, c['knn'])         # (N,)
                    loss  = F.binary_cross_entropy(
                        score[msk], lbl[msk])

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
            finally:
                del x, msk, lbl

        scheduler.step()

        # Validação
        model.eval()
        val_losses = []
        with torch.no_grad():
            for c in val_cache:
                if c['mask'].sum() < 5:
                    continue
                x   = torch.tensor(c['node_feat'], dtype=torch.float32).to(device)
                msk = torch.tensor(c['mask'],   dtype=torch.bool).to(device)
                lbl = torch.tensor(c['labels'], dtype=torch.float32).to(device)
                with autocast():
                    score = model(x, c['knn'])
                    val_losses.append(
                        F.binary_cross_entropy(score[msk], lbl[msk]).item())
                del x, msk, lbl

        if not ep_losses:
            continue

        avg_t = float(np.mean(ep_losses))
        avg_v = float(np.mean(val_losses)) if val_losses else float('nan')
        lr_   = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch+1:03d}/{num_epochs} | "
                     f"Train={avg_t:.5f} | Val={avg_v:.5f} | LR={lr_:.2e}")

        if avg_v < best_val:
            best_val   = avg_v
            patience_c = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': best_val}, save_path)
        else:
            patience_c += 1

        if patience_c >= PATIENCE_GNN:
            log.info(f"Early stop (val={avg_v:.5f})")
            break

        gc.collect()
        torch.cuda.empty_cache()

    log.info(f"\nTreino concluído. Melhor val loss: {best_val:.5f}")
    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        log.info(f"Melhor modelo carregado (epoch {ckpt['epoch']+1})")

    return model


# ============================================================================
# INFERÊNCIA
# ============================================================================

@torch.no_grad()
def compute_gnn_scores(
    model: SpatialRefinementGNN,
    data_list: list,
    device: torch.device,
) -> list:
    """
    Computa score refinado por ponto para cada nuvem.

    Score final = GNN(features_12D, kNN_XYZ)
    O GNN integra SF-GMM score + contexto espacial de vizinhança.

    Pós-processamento:
      1. Suavização k-NN em XYZ: dilui FPs isolados residuais
      2. Supressão de borda do recorte
      3. Normalização percentil 2–98
    """
    model.eval()
    results = []

    for d in data_list:
        xyz_np    = d['features'][:, :3]
        sf_score  = compute_sf_gmm_score(d)
        node_feat = build_node_features(d, sf_score)
        knn       = _xyz_knn(xyz_np, K_GNN)

        x = torch.tensor(node_feat, dtype=torch.float32).to(device)
        with autocast():
            score = model(x, knn).cpu().numpy().astype(np.float32)
        del x
        torch.cuda.empty_cache()

        # Pós-processamento
        score = spatial_smooth_scores(xyz_np, score, k=SMOOTH_K)
        bnd   = _boundary_mask(xyz_np, radius=BOUNDARY_RADIUS,
                                min_neighbors=BOUNDARY_MIN_NB)
        score[bnd] = 0.0
        lo, hi = np.percentile(score, 2), np.percentile(score, 98)
        score  = np.clip((score - lo) / (hi - lo + 1e-8), 0, 1)

        results.append({
            'filename'    : d['filename'],
            'has_crack'   : d['has_crack'],
            'score'       : score,
            'gt_labels'   : d['labels'],
            'n_points'    : len(d['labels']),
            'xyz'         : xyz_np,
            'rgb'         : d['features'][:, 3:6],
            'scalar_field': d['features'][:, 9],
        })

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    ts = datetime.now().strftime('%d%m%Y_%H%M%S')
    print("\n" + "="*70)
    print("  SPATIAL REFINEMENT GNN")
    print("  EdgeConv × 2 | pseudo-labels SF-GMM | kNN em XYZ")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    run_results = os.path.join(RESULTS_GNN, f'run_{ts}')
    run_vis     = os.path.join(VIS_GNN,     f'run_{ts}')
    run_ply     = os.path.join(PLY_GNN,     f'run_{ts}')
    for p in [run_results, run_vis, run_ply, MODELS_GNN]:
        os.makedirs(p, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Dispositivo: {device}")

    # ── Dados ─────────────────────────────────────────────────────────────────
    log.info("\nCarregando dados...")
    all_data  = load_folder(DATA_TRAIN) + load_folder(DATA_TEST)
    _, _, eval_list = split_dataset(all_data)

    if not all_data:
        log.error("Nenhum dado encontrado.")
        return

    # ── Modelo ────────────────────────────────────────────────────────────────
    model    = SpatialRefinementGNN(input_dim=GNN_INPUT_DIM, k=K_GNN)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"SpatialRefinementGNN: {n_params:,} parâmetros")

    # ── Treino ────────────────────────────────────────────────────────────────
    if os.path.exists(MODEL_CKP):
        log.info(f"Checkpoint encontrado: {MODEL_CKP}")
        ckpt = torch.load(MODEL_CKP, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
    else:
        t0    = time.time()
        model = train_spatial_gnn(model, all_data, device,
                                   num_epochs=EPOCHS_GNN, lr=LR_GNN,
                                   save_path=MODEL_CKP)
        log.info(f"Treino: {(time.time()-t0)/3600:.1f}h")

    # ── Scores ────────────────────────────────────────────────────────────────
    log.info("\nComputando GNN scores...")
    eval_data = eval_list if eval_list else all_data
    results   = compute_gnn_scores(model, eval_data, device)

    normal_ref = compute_gnn_scores(
        model,
        [d for d in all_data if not d.get('has_crack')][:10],
        device,
    )

    # ── Threshold + métricas ──────────────────────────────────────────────────
    thr, gmm_info = fit_gmm_threshold(results, normal_results=normal_ref)
    results       = apply_threshold(results, thr)

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
            best_f1  = row['f1']
            best_name = name

    log.info(f"\n  Melhor F1: [{best_name}] {best_f1:.4f}")

    results = results_gmean
    metrics = metrics_gmean

    pd.DataFrame(comparison).to_csv(
        os.path.join(run_results, f'comparacao_{ts}.csv'), index=False)

    # ── PLY coloridos ─────────────────────────────────────────────────────────
    for r in results:
        if not r['has_crack']:
            continue
        save_colored_ply(
            xyz=r['xyz'], rgb_orig=r['rgb'],
            pred_labels=r['pred_labels'],
            path=os.path.join(run_ply,
                              r['filename'].replace('.ply', '_gnn.ply')))

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTADOS — SPATIAL REFINEMENT GNN")
    print("="*70)
    if metrics:
        print(f"  F1    : {metrics.get('f1',    0):.4f}")
        print(f"  IoU   : {metrics.get('iou',   0):.4f}")
        print(f"  AUROC : {metrics.get('auroc', 0):.4f}")
    print(f"  Threshold: {thr:.4f} | Resultados: {run_results}")
    print("="*70)

    return model, results, metrics


if __name__ == '__main__':
    main()
