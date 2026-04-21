# ============================================================================
# SCALAR FIELD MAE — CAMINHO B
# ============================================================================
#
# Hipótese central:
#   Um modelo treinado em superfícies normais aprende a relação
#   (XYZ, RGB, normais) → scalar_field esperado para aquela geometria.
#   Em uma rachadura, a geometria e cor da superfície vizinha indicam
#   SF alto (~200), mas o SF real é baixo (~40).
#   Erro de reconstrução alto → anomalia.
#
# Arquitetura — LocalSFPredictor:
#   Input  : XYZ + RGB + normais (9D) — scalar_field EXCLUÍDO do input
#   Camada 1: MLP por ponto: 9 → 64
#   Camada 2: EdgeConv (kNN em XYZ): 64 → 128  [contexto de vizinhança]
#   Camada 3: Global max-pool → MLP: 128 → 64  [contexto global da nuvem]
#   Head   : Concat(128+64) → 64 → 1 → Sigmoid  [SF normalizado ∈ [0,1]]
#
# Por que excluir cols 10–15 (curvatura, densidade, variância, etc.)?
#   Essas features geométricas já são anômalas em pontos de rachadura.
#   Usá-las no input seria treinar o modelo para prever "SF baixo quando
#   curvatura alta", aprendendo o sinal de rachadura diretamente em vez
#   de a partir de evidências de superfície normal.
#
# Treino: apenas superfícies normais (n_avaria_*).
# Score : |SF_pred - SF_real / 255| por ponto, suavizado por k-NN em XYZ.
#
# Referências:
#   He et al. (2022) — MAE: Masked Autoencoders Are Scalable Vision Learners.
#   Wang et al. (2019) — DGCNN: Dynamic Graph CNN for Learning on Point Clouds.
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
RESULTS_B  = f'{BASE_PATH}/results_sf_mae'
MODELS_B   = f'{BASE_PATH}/models_sf_mae'
VIS_B      = f'{BASE_PATH}/visualizations_sf_mae'
PLY_B      = f'{BASE_PATH}/results_sf_mae/ply'
MODEL_CKP  = f'{MODELS_B}/sf_predictor.pth'

log = setup_logging(f'{BASE_PATH}/logs_sf_mae')

# ── Hiperparâmetros ────────────────────────────────────────────────────────────
K_EDGE       = 16     # vizinhos XYZ para o EdgeConv
LR_SF        = 3e-4
EPOCHS_SF    = 100
PATIENCE_SF  = 15
CHUNK_EDGE   = 8_000  # pontos por chunk no EdgeConv (controla VRAM)

# Colunas do input: XYZ (0-2) + RGB (3-5) + normais (6-8) = 9D
# Scalar_field (9) excluído — é o target
INPUT_COLS = list(range(9))
INPUT_DIM_SF = len(INPUT_COLS)   # 9


# ============================================================================
# EDGE CONV CHUNKED — memória eficiente para nuvens grandes
# ============================================================================

def _edge_conv_chunked(
    h: torch.Tensor,
    knn_idx: np.ndarray,
    mlp: nn.Module,
    chunk: int = CHUNK_EDGE,
) -> torch.Tensor:
    """
    Aplica um bloco EdgeConv em chunks para evitar OOM com nuvens grandes.

    Para cada ponto i e seus k vizinhos j:
      edge_feat(i,j) = [h_i || h_j − h_i]   (2·C_in dimensional)
      h_i_new = max_{j∈KNN(i)} MLP(edge_feat(i,j))

    O kNN é pré-computado em XYZ (cpu, scipy) e passado como argumento.
    Processar em chunks: apenas `chunk` pontos × k vizinhos na GPU por vez.

    h      : (N, C_in) tensor na GPU
    knn_idx: (N, k)    numpy int64 CPU
    mlp    : MLP que recebe (N*k, 2*C_in) → (N*k, C_out)
    Returns: (N, C_out) tensor na GPU
    """
    N, C_in = h.shape
    k       = knn_idx.shape[1]

    # Inferir C_out da última camada Linear do MLP (evita dummy forward com BN)
    C_out = next(m.out_features for m in reversed(list(mlp.modules()))
                 if isinstance(m, nn.Linear))

    out = torch.zeros(N, C_out, device=h.device, dtype=h.dtype)

    for s in range(0, N, chunk):
        e    = min(s + chunk, N)
        cs   = e - s
        idx  = torch.from_numpy(knn_idx[s:e].astype(np.int64)).to(h.device)

        neighbors  = h[idx.view(-1)].view(cs, k, C_in)    # (cs, k, C_in)
        h_i        = h[s:e].unsqueeze(1).expand(-1, k, -1) # (cs, k, C_in)
        edge_feat  = torch.cat([h_i, neighbors - h_i], dim=-1)  # (cs, k, 2*C_in)

        out_chunk  = mlp(edge_feat.reshape(cs * k, -1))    # (cs*k, C_out)
        out[s:e]   = out_chunk.view(cs, k, C_out).max(dim=1).values

    return out


# ============================================================================
# LOCAL SF PREDICTOR — modelo principal
# ============================================================================

class LocalSFPredictor(nn.Module):
    """
    Preditor de scalar_field a partir de XYZ + RGB + normais (9D).

    Arquitetura:
      1. Point MLP     : 9D → 64D  (features por ponto)
      2. EdgeConv      : 64D → 128D (contexto local via kNN em XYZ)
      3. Global max-pool + MLP: 128D → 64D (contexto global da nuvem)
      4. Concat local + global = 192D
      5. Prediction head: 192D → 64D → 1D → Sigmoid

    O kNN é construído em XYZ (espaço 3D real), não no espaço de features.
    Isso garante que o contexto local é geograficamente significativo.

    Output: SF normalizado ∈ [0, 1] por ponto.
    Anomaly score em inferência: |SF_pred - SF_real/255|.
    """

    def __init__(self, k: int = K_EDGE):
        super().__init__()
        self.k = k

        # ── 1. MLP por ponto: 9 → 32 → 64 ────────────────────────────────────
        self.point_mlp = nn.Sequential(
            nn.LayerNorm(INPUT_DIM_SF),
            nn.Linear(INPUT_DIM_SF, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        # ── 2. EdgeConv MLP: [h_i || h_j−h_i] = 128D → 128D ─────────────────
        self.edge_mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )

        # ── 3. Global context: 128D → 64D ─────────────────────────────────────
        self.global_mlp = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # ── 4. Prediction head: 192D → 64D → 1D ──────────────────────────────
        self.pred_head = nn.Sequential(
            nn.Linear(192, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        knn_idx: np.ndarray,
    ) -> torch.Tensor:
        """
        x      : (N, 9)  — XYZ + RGB + normais
        knn_idx: (N, k)  — índices k-NN em XYZ (pré-computados, numpy)
        Returns: (N,) — SF predito normalizado ∈ [0, 1]
        """
        # 1. MLP por ponto
        h = self.point_mlp(x)                                  # (N, 64)

        # 2. EdgeConv em chunks
        h_local = _edge_conv_chunked(h, knn_idx, self.edge_mlp)  # (N, 128)

        # 3. Global max-pool → broadcast
        h_global = self.global_mlp(h_local.max(dim=0).values)    # (64,)
        h_global = h_global.unsqueeze(0).expand(x.size(0), -1)   # (N, 64)

        # 4. Concat + head
        h_cat = torch.cat([h_local, h_global], dim=1)            # (N, 192)
        return self.pred_head(h_cat).squeeze(-1)                  # (N,)


# ============================================================================
# TREINAMENTO
# ============================================================================

def train_sf_predictor(
    model: LocalSFPredictor,
    all_data: list,
    device: torch.device,
    num_epochs: int = EPOCHS_SF,
    lr: float = LR_SF,
    save_path: str = MODEL_CKP,
    normal_only: bool = True,
) -> LocalSFPredictor:
    """
    Treina LocalSFPredictor em superfícies normais.

    Protocolo:
      - Usa apenas nuvens sem rachadura (n_avaria_*) por padrão.
        Rachaduras contaminariam o target: queremos aprender o SF esperado
        de superfície normal, não de rachadura.
      - Loss: Smooth L1 (Huber, beta=0.05) sobre SF normalizado [0,1].
        Mais robusta a outliers de SF do que MSE.
      - kNN XYZ é cacheado por nuvem (XYZ não muda entre epochs).
      - Early stopping em loss de validação (últimas 20% das nuvens).
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    data = ([d for d in all_data if not d.get('has_crack', True)]
            if normal_only else all_data)
    if len(data) < 2:
        log.warning(f"SF MAE: apenas {len(data)} nuvens normais — usando todas")
        data = all_data

    # Split treino / validação (80/20)
    n_val   = max(1, len(data) // 5)
    train_d = data[n_val:]
    val_d   = data[:n_val]

    log.info(f"\n{'='*65}")
    log.info("SCALAR FIELD MAE — Treino (Caminho B)")
    log.info(f"Treino: {len(train_d)} nuvens | Val: {len(val_d)} nuvens")
    log.info(f"Epochs: {num_epochs} | LR: {lr} | k={K_EDGE}")
    log.info(f"{'='*65}")

    model     = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr,
                      betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler    = GradScaler()

    # Cacheia kNN XYZ por nuvem (fixo entre epochs)
    log.info("Cacheando kNN XYZ...")
    knn_cache_train = [_xyz_knn(d['features'][:, :3], K_EDGE) for d in train_d]
    knn_cache_val   = [_xyz_knn(d['features'][:, :3], K_EDGE) for d in val_d]
    log.info(f"Cache pronto: {len(knn_cache_train)+len(knn_cache_val)} nuvens")

    best_val_loss = float('inf')
    patience_c    = 0

    for epoch in range(num_epochs):
        # ── Treino ────────────────────────────────────────────────────────────
        model.train()
        ep_losses = []
        perm = np.random.permutation(len(train_d))

        for i in perm:
            d      = train_d[i]
            x      = torch.tensor(d['features'][:, INPUT_COLS],
                                   dtype=torch.float32).to(device)
            sf_raw = d['features'][:, 9]
            sf_tgt = torch.tensor(sf_raw / 255.0,
                                   dtype=torch.float32).to(device)
            knn    = knn_cache_train[i]

            if x.size(0) < K_EDGE + 1:
                continue

            optimizer.zero_grad(set_to_none=True)
            try:
                with autocast():
                    sf_pred = model(x, knn)
                    loss    = F.smooth_l1_loss(sf_pred, sf_tgt, beta=0.05)

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

        # ── Validação ─────────────────────────────────────────────────────────
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i, d in enumerate(val_d):
                x      = torch.tensor(d['features'][:, INPUT_COLS],
                                       dtype=torch.float32).to(device)
                sf_tgt = torch.tensor(d['features'][:, 9] / 255.0,
                                       dtype=torch.float32).to(device)
                knn    = knn_cache_val[i]
                if x.size(0) < K_EDGE + 1:
                    continue
                with autocast():
                    sf_pred = model(x, knn)
                    val_losses.append(
                        F.smooth_l1_loss(sf_pred, sf_tgt, beta=0.05).item())

        if not ep_losses:
            continue

        avg_train = float(np.mean(ep_losses))
        avg_val   = float(np.mean(val_losses)) if val_losses else float('nan')
        cur_lr    = optimizer.param_groups[0]['lr']

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch+1:03d}/{num_epochs} | "
                     f"Train={avg_train:.5f} | Val={avg_val:.5f} | "
                     f"LR={cur_lr:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_c    = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'val_loss': best_val_loss}, save_path)
        else:
            patience_c += 1

        if patience_c >= PATIENCE_SF:
            log.info(f"Early stop: {PATIENCE_SF} epochs sem melhoria "
                     f"(val={avg_val:.5f})")
            break

        gc.collect()
        torch.cuda.empty_cache()

    log.info(f"\nTreino concluído. Melhor val loss: {best_val_loss:.5f}")
    log.info(f"Checkpoint salvo em: {save_path}")

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        log.info(f"Melhor modelo carregado (epoch {ckpt['epoch']+1})")

    return model


# ============================================================================
# INFERÊNCIA — ANOMALY SCORES
# ============================================================================

@torch.no_grad()
def compute_sf_anomaly_scores(
    model: LocalSFPredictor,
    data_list: list,
    device: torch.device,
) -> list:
    """
    Computa anomaly score por ponto para cada nuvem.

    score(i) = |SF_pred(i) - SF_real(i) / 255|

    Intuição: o modelo aprendeu SF esperado para superfícies normais.
    Em regiões de rachadura, o SF real é baixo (~40) mas o modelo prediz
    alto (~200) porque a geometria e cor vizinha indicam superfície normal.
    Erro alto → anomalia.

    Pós-processamento:
      1. Suavização k-NN em XYZ: dilui FPs isolados
      2. Supressão de borda: remove pontos de fronteira do recorte
      3. Normalização percentil 2–98: [0, 1]
    """
    model.eval()
    results = []

    for d in data_list:
        x      = torch.tensor(d['features'][:, INPUT_COLS],
                               dtype=torch.float32).to(device)
        sf_raw = d['features'][:, 9]
        xyz_np = d['features'][:, :3]
        knn    = _xyz_knn(xyz_np, K_EDGE)

        with autocast():
            sf_pred = model(x, knn).cpu().numpy().astype(np.float32)

        sf_real_norm = sf_raw / 255.0
        score_raw    = np.abs(sf_pred - sf_real_norm)

        # Pós-processamento
        score = spatial_smooth_scores(xyz_np, score_raw, k=SMOOTH_K)
        bnd   = _boundary_mask(xyz_np, radius=BOUNDARY_RADIUS,
                                min_neighbors=BOUNDARY_MIN_NB)
        score[bnd] = 0.0

        lo, hi = np.percentile(score, 2), np.percentile(score, 98)
        score  = np.clip((score - lo) / (hi - lo + 1e-8), 0, 1)

        results.append({
            'filename'  : d['filename'],
            'has_crack' : d['has_crack'],
            'score'     : score,
            'gt_labels' : d['labels'],
            'n_points'  : len(d['labels']),
            'xyz'       : xyz_np,
            'rgb'       : d['features'][:, 3:6],
            'scalar_field': sf_raw,
            'sf_pred'   : sf_pred * 255.0,   # em escala original (0-255)
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
    print("  SCALAR FIELD MAE — CAMINHO B")
    print("  LocalSFPredictor: XYZ+RGB+normais → SF prediction")
    print(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("="*70)

    run_results = os.path.join(RESULTS_B, f'run_{ts}')
    run_vis     = os.path.join(VIS_B, f'run_{ts}')
    run_ply     = os.path.join(PLY_B, f'run_{ts}')

    for p in [run_results, run_vis, run_ply, MODELS_B]:
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

    # ── 2. Modelo ─────────────────────────────────────────────────────────────
    model = LocalSFPredictor(k=K_EDGE)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"LocalSFPredictor: {n_params:,} parâmetros")

    # ── 3. Treino ─────────────────────────────────────────────────────────────
    if os.path.exists(MODEL_CKP):
        log.info(f"Checkpoint encontrado: {MODEL_CKP} — carregando...")
        ckpt = torch.load(MODEL_CKP, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        model = model.to(device)
    else:
        log.info("\nIniciando treino...")
        t0    = time.time()
        model = train_sf_predictor(
            model, all_data, device,
            num_epochs=EPOCHS_SF, lr=LR_SF,
            save_path=MODEL_CKP, normal_only=True,
        )
        log.info(f"Treino: {(time.time()-t0)/3600:.1f}h")

    # ── 4. Scores de anomalia ─────────────────────────────────────────────────
    log.info("\nComputando anomaly scores...")
    eval_data = eval_list if eval_list else train_all
    results   = compute_sf_anomaly_scores(model, eval_data, device)

    # ── 5. Threshold + métricas ───────────────────────────────────────────────
    normal_ref = compute_sf_anomaly_scores(
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
    thr_gmean     = None
    best_f1_any   = -1.0
    best_name_any = None

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
            thr_gmean     = thr_s
        if row['f1'] > best_f1_any:
            best_f1_any   = row['f1']
            best_name_any = name

    log.info(f"\n  Melhor F1: [{best_name_any}] {best_f1_any:.4f}")
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
                              r['filename'].replace('.ply', '_sfmae.ply')))

    # ── Resumo ────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  RESULTADOS — SCALAR FIELD MAE (Caminho B)")
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
