# ============================================================================
# AVALIAÇÃO — threshold GMM, métricas por ponto
# ============================================================================
# fit_gmm_threshold  — threshold dinâmico via cruzamento exato de gaussianas
# apply_threshold    — binariza score em pred_labels
# evaluate           — accuracy, F1, IoU, AUROC, AP por ponto
# Compartilhado por GAN v5 e Professor-Aluno v1.
# Referência: McLachlan & Peel (2000) — Finite Mixture Models.
# ============================================================================

from utils.config import *

import logging
log = setup_logging(LOG_PATH)

def fit_gmm_threshold(results: list,
                      normal_results: list = None,
                      fallback_pctl: float = ANOMALY_PCTL) -> tuple:
    """
    Ajusta GMM de 2 componentes ao score de anomalia.

    Estratégia:
      1. Usa scores das nuvens NORMAIS (de treino) para estimar componente normal.
      2. Encontra ponto de cruzamento entre as duas gaussianas como threshold.
      3. Se GMM falhar (e.g. componentes colapsadas), usa percentil como fallback.

    Retorna (threshold, gmm_info_dict).
    """
    from sklearn.mixture import GaussianMixture

    # Coletar scores para fitting
    if normal_results is not None:
        fit_scores = np.concatenate([r['score'] for r in normal_results])
        eval_scores = np.concatenate([r['score'] for r in results])
    else:
        fit_scores  = np.concatenate([r['score'] for r in results])
        eval_scores = fit_scores

    # Remover NaN/Inf
    fit_scores  = fit_scores[np.isfinite(fit_scores)].reshape(-1, 1)
    eval_scores = eval_scores[np.isfinite(eval_scores)]

    gmm_info = {'method': 'percentile_fallback', 'threshold': None}

    try:
        gmm = GaussianMixture(n_components=2, covariance_type='full',
                               max_iter=200, random_state=42, n_init=5)
        gmm.fit(fit_scores)

        # Identificar qual componente é "normal" (média menor)
        means = gmm.means_.flatten()
        idx_normal = int(np.argmin(means))
        idx_anomal = 1 - idx_normal

        mu_n, sig_n = means[idx_normal], np.sqrt(gmm.covariances_[idx_normal].flatten()[0])
        mu_a, sig_a = means[idx_anomal], np.sqrt(gmm.covariances_[idx_anomal].flatten()[0])
        w_n, w_a    = gmm.weights_[idx_normal], gmm.weights_[idx_anomal]

        # ── Cruzamento numérico exato ─────────────────────────────────────
        # Resolve w_n·N(x;µn,σn) = w_a·N(x;µa,σa) numericamente.
        # A aproximação ponderada  thr≈(µn·wa + µa·wn)/(wn+wa)  pode ter
        # erro de até 0.15 quando σn ≠ σa, levando a threshold incorreto.
        # Usamos mudança de sinal de (p_n − p_a) no intervalo (µn, µa).
        from scipy.stats import norm as _norm

        lo_x, hi_x = min(mu_n, mu_a), max(mu_n, mu_a)
        x_scan = np.linspace(lo_x, hi_x, 10_000)
        p_n_scan = w_n * _norm.pdf(x_scan, mu_n, sig_n)
        p_a_scan = w_a * _norm.pdf(x_scan, mu_a, sig_a)
        diff_scan = p_n_scan - p_a_scan

        sign_changes = np.where(np.diff(np.sign(diff_scan)))[0]
        if len(sign_changes) > 0:
            # Interpolar linearmente para maior precisão
            i = sign_changes[0]
            d1, d2 = diff_scan[i], diff_scan[i + 1]
            thr_gmm = float(x_scan[i] - d1 * (x_scan[i+1] - x_scan[i]) / (d2 - d1))
        else:
            # Fallback: ponto de mínima diferença entre as gaussianas
            thr_gmm = float(x_scan[np.argmin(np.abs(diff_scan))])

        # Validação: threshold deve estar entre as duas médias
        if min(mu_n, mu_a) < thr_gmm < max(mu_n, mu_a):
            gmm_info = {
                'method'    : 'GMM',
                'threshold' : float(thr_gmm),
                'mu_normal' : float(mu_n),
                'mu_anomal' : float(mu_a),
                'w_normal'  : float(w_n),
                'w_anomal'  : float(w_a),
            }
            thr = thr_gmm
            log.info(f"Threshold GMM: {thr:.4f}  "
                     f"[µ_normal={mu_n:.3f}, µ_anomal={mu_a:.3f}]")
        else:
            raise ValueError(f"Threshold fora das médias: {thr_gmm:.3f}")

    except Exception as e:
        log.warning(f"GMM falhou ({e}), usando P{fallback_pctl:.0f} como fallback")
        thr = float(np.percentile(eval_scores, fallback_pctl))
        gmm_info['threshold'] = thr

    return thr, gmm_info


def apply_threshold(results: list, thr: float) -> list:
    """Aplica threshold ao score composto e gera pred_labels."""
    for r in results:
        r['pred_labels'] = (r['score'] > thr).astype(np.int64)
        r['threshold']   = thr
    return results


def compute_crack_sf_interval(data_list: list) -> tuple:
    """
    Aprende o intervalo [sf_min, sf_max] do scalar_Scalar_field para pontos
    de rachadura (label == 1) a partir dos dados de TREINO.

    Por que usar o intervalo de treino?
      As nuvens de treino têm labels por ponto indicando quais são rachadura.
      Em inferência (test), essas labels não existem. Portanto, usamos o
      intervalo aprendido como restrição hard: nenhum ponto fora de
      [sf_min, sf_max] pode ser predito como rachadura.

    Motivação física:
      Rachaduras absorvem mais luz e retornam menor intensidade no scanner
      (scalar_Scalar_field menor). O intervalo aprendido captura exatamente
      a faixa de intensidade observada em pontos confirmados como rachadura.

    Retorna:
      (sf_min, sf_max) — intervalo global de scalar_field para label=1.
      Se não houver pontos label=1 nos dados, retorna (0.0, inf).
    """
    sf_crack = []

    for d in data_list:
        labels = d.get('labels')
        if labels is None:
            continue
        crack_mask = (labels == 1)
        if not crack_mask.any():
            continue
        sf = d['features'][:, 9]   # scalar_Scalar_field — col 9 do vetor de 15 features
        sf_crack.append(sf[crack_mask])

    if not sf_crack:
        log.warning("compute_crack_sf_interval: nenhum ponto label=1 encontrado. "
                    "Intervalo não aplicado.")
        return 0.0, float('inf')

    all_sf = np.concatenate(sf_crack)
    sf_min, sf_max = float(all_sf.min()), float(all_sf.max())
    log.info(f"Intervalo scalar_field para crack (label=1): [{sf_min:.2f}, {sf_max:.2f}]  "
             f"({len(all_sf):,} pontos de rachadura)")
    return sf_min, sf_max


def apply_scalar_field_gate(results: list,
                             sf_min: float,
                             sf_max: float) -> list:
    """
    Pós-filtro supervisionado: reverte pred_labels=1 para 0 em pontos cujo
    scalar_field está FORA do intervalo aprendido no treino [sf_min, sf_max].

    Lógica:
      - O intervalo [sf_min, sf_max] foi derivado de pontos label=1 no treino.
      - Qualquer ponto fora desse intervalo nunca foi observado como rachadura
        no conjunto de treino → não pode ser uma predição válida.
      - Elimina falsos positivos em regiões de alta intensidade de retorno
        (superfícies lisas que o modelo confunde com anomalias).

    Não altera `score` — apenas `pred_labels`. AUROC permanece intacto.
    """
    for r in results:
        sf_raw = r.get('scalar_field')
        if sf_raw is None or 'pred_labels' not in r:
            continue

        sf   = np.asarray(sf_raw, dtype=np.float32)
        pred = r['pred_labels'].copy()

        outside = (sf < sf_min) | (sf > sf_max)
        pred[outside] = 0
        r['pred_labels'] = pred

    return results

# ============================================================================
# SCALAR FIELD GMM — prior estatístico por nuvem
# ============================================================================

class ScalarFieldGMM:
    """
    Ajusta GMM 1D ao scalar_field de uma nuvem para distinguir rachaduras
    de superfície normal sem usar labels humanos.

    Para nuvens com scalar_Scalar_field (bimodal, gap≥0): valley detection
    identifica o threshold com alta confiança.
    Para nuvens com scalar_R (unimodal): retorna comportamento neutro
    (weights=1.0, confidence=0.0) — transparente para o pipeline downstream.

    Uso:
        gmm = ScalarFieldGMM(scalar_field_array).fit()
        probs   = gmm.anomaly_probability()       # (N,) ∈ [0,1]
        weights = gmm.soft_weights()              # (N,) — gate soft
        conf    = gmm.pseudo_label_confidence()   # (N,) — peso para Push-Pull
    """

    def __init__(self, scalar: np.ndarray, n_components: int = 2):
        self.scalar = np.asarray(scalar, dtype=np.float32).ravel()
        self.n_components = n_components
        self._gmm        = None
        self._modality   = None
        self._threshold  = None
        self._crack_idx  = None
        self._fitted     = False

    def fit(self) -> 'ScalarFieldGMM':
        from sklearn.mixture import GaussianMixture

        scalar = self.scalar
        X      = scalar.reshape(-1, 1)

        # Edge case: valores insuficientes ou constantes
        if len(scalar) < 10 or scalar.std() < 1e-6:
            self._modality  = 'unimodal'
            self._threshold = float(scalar.mean())
            self._fitted    = True
            return self

        # ── Fit GMM com 1, 2 e 3 componentes; escolhe por BIC ────────────────
        # 3 componentes cobre nuvens onde o concreto normal tem sub-estrutura
        # (ex: região abaixo E acima do intervalo de rachadura).
        best_gmm  = None
        best_bic  = np.inf
        gmm1_bic  = np.inf
        try:
            for k in (1, 2, 3):
                g = GaussianMixture(
                    n_components=k, covariance_type='full',
                    max_iter=300, random_state=42, n_init=5,
                )
                g.fit(X)
                b = g.bic(X)
                if k == 1:
                    gmm1_bic = b
                if b < best_bic:
                    best_bic = b
                    best_gmm = g
        except Exception as e:
            log.warning(f"ScalarFieldGMM: GMM falhou ({e})")
            self._modality  = 'unimodal'
            self._threshold = float(np.percentile(scalar, 50))
            self._fitted    = True
            return self

        n_best = best_gmm.n_components

        # ── Seleção do componente de rachadura ────────────────────────────────
        # Princípio: rachaduras são a classe MINORITÁRIA — o intervalo de crack
        # representa uma fração pequena dos pontos (~5-15%), independente de
        # qual faixa de scalar_field corresponde (pode ser média, baixa ou alta).
        # argmin(weights) é mais robusto que argmin(means) para intervalos não-extremos.
        weights = best_gmm.weights_.ravel()
        means   = best_gmm.means_.ravel()
        crack_candidate = int(np.argmin(weights))

        # Sanidade: se o componente minoritário ainda tem peso > 0.4, a divisão
        # está muito equilibrada — o GMM não encontrou um cluster isolado de crack.
        # Nesse caso, tratar como unimodal (score neutro 0.5).
        if weights[crack_candidate] > 0.40:
            self._gmm       = None
            self._modality  = 'unimodal'
            self._threshold = float(np.percentile(scalar, 50))
            self._fitted    = True
            return self

        self._gmm       = best_gmm
        self._crack_idx = crack_candidate

        # ── Detecção de multimodalidade significativa ─────────────────────────
        # Exige que o modelo multi-componente seja melhor que o unimodal por BIC
        # e que o componente de crack tenha separação mínima dos demais.
        bic_improvement = gmm1_bic - best_bic   # > 0 se multi é melhor

        normal_means = np.delete(means, crack_candidate)
        crack_mean   = means[crack_candidate]
        crack_std    = float(np.sqrt(best_gmm.covariances_.ravel()[crack_candidate]))
        min_sep      = float(np.min(np.abs(normal_means - crack_mean))) / (crack_std + 1e-8)

        if bic_improvement > 10 and min_sep > 0.5:
            self._modality = 'bimodal'
            # Threshold: mínimo da pdf entre o crack e o componente normal mais próximo
            closest_normal_mean = float(normal_means[np.argmin(np.abs(normal_means - crack_mean))])
            lo = min(crack_mean, closest_normal_mean)
            hi = max(crack_mean, closest_normal_mean)
            x_search  = np.linspace(lo, hi, 500).reshape(-1, 1)
            log_probs  = best_gmm.score_samples(x_search)
            self._threshold = float(x_search[int(np.argmin(log_probs)), 0])
        else:
            self._modality  = 'unimodal'
            self._threshold = float(np.percentile(scalar, 50))

        self._fitted = True
        return self

    def _ensure_fitted(self):
        if not self._fitted:
            self.fit()

    def anomaly_probability(self) -> np.ndarray:
        """P(rachadura | scalar_field[i]) por ponto. Unimodal → ~0.5 uniforme."""
        self._ensure_fitted()
        if self._modality == 'unimodal' or self._gmm is None:
            return np.full(len(self.scalar), 0.5, dtype=np.float32)
        probs = self._gmm.predict_proba(self.scalar.reshape(-1, 1))
        return probs[:, self._crack_idx].astype(np.float32)

    def soft_weights(self) -> np.ndarray:
        """
        Gate soft: peso por ponto para modular o anomaly score.
        Unimodal → 1.0 (transparente). Bimodal → P(crack|sf).
        """
        self._ensure_fitted()
        if self._modality == 'unimodal':
            return np.ones(len(self.scalar), dtype=np.float32)
        return self.anomaly_probability()

    def pseudo_label_confidence(self) -> np.ndarray:
        """
        Confiança do pseudo-label por ponto.
        Unimodal → 0.0. Bimodal → |P(crack) - 0.5| * 2 ∈ [0,1].
        """
        self._ensure_fitted()
        if self._modality == 'unimodal':
            return np.zeros(len(self.scalar), dtype=np.float32)
        probs = self.anomaly_probability()
        return (np.abs(probs - 0.5) * 2.0).astype(np.float32)

    def crack_interval(self) -> tuple:
        """(x_min, x_max) do cluster de rachadura no scalar_field."""
        self._ensure_fitted()
        crack_mask = self.scalar <= self._threshold
        if crack_mask.sum() == 0:
            return (float(self.scalar.min()), float(self._threshold))
        crack_vals = self.scalar[crack_mask]
        return (float(crack_vals.min()), float(crack_vals.max()))

    @property
    def threshold(self) -> float:
        self._ensure_fitted()
        return self._threshold

    @property
    def modality(self) -> str:
        self._ensure_fitted()
        return self._modality


# ============================================================================
# SAÍDA PLY COLORIDA
# ============================================================================

def save_colored_ply(
    xyz: np.ndarray,
    rgb_orig: np.ndarray,
    pred_labels: np.ndarray,
    path: str,
    crack_color: tuple = (255, 0, 0),
) -> None:
    """
    Salva nuvem de pontos PLY com rachaduras coloridas.
    Pontos normais (pred_labels=0): cor original preservada.
    Pontos de rachadura (pred_labels=1): crack_color (vermelho padrão).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    rgb_out = rgb_orig.copy()
    crack_rgb_norm = np.array(crack_color, dtype=np.float32) / 255.0
    rgb_out[pred_labels == 1] = crack_rgb_norm

    rgb_uint8 = (rgb_out * 255.0).clip(0, 255).astype(np.uint8)

    n = len(xyz)
    vertex = np.zeros(n, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ])
    vertex['x']     = xyz[:, 0]
    vertex['y']     = xyz[:, 1]
    vertex['z']     = xyz[:, 2]
    vertex['red']   = rgb_uint8[:, 0]
    vertex['green'] = rgb_uint8[:, 1]
    vertex['blue']  = rgb_uint8[:, 2]

    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(path)
    log.info(f"PLY salvo: {path}  ({n:,} pts, "
             f"{int(pred_labels.sum()):,} rachaduras em vermelho)")

# ============================================================================
# CHAMFER DISTANCE
# ============================================================================

def chamfer_distance(points_pred: np.ndarray,
                     points_gt:   np.ndarray) -> float:
    """
    Distância de Chamfer simétrica entre dois conjuntos de pontos 3D.

    CD(P,Q) = (1/|P|) Σ_{p∈P} min_{q∈Q} ||p-q||²
            + (1/|Q|) Σ_{q∈Q} min_{p∈P} ||q-p||²  dividido por 2

    Mede o alinhamento espacial entre avarias preditas e reais.
    Referência: Wu et al. (2021) — Density-aware Chamfer Distance.
    """
    if len(points_pred) == 0 or len(points_gt) == 0:
        return float('nan')
    tree_pred = cKDTree(points_pred)
    tree_gt   = cKDTree(points_gt)
    d_p2q, _  = tree_gt.query(points_pred, k=1)
    d_q2p, _  = tree_pred.query(points_gt,  k=1)
    return float((np.mean(d_p2q**2) + np.mean(d_q2p**2)) / 2.0)


def _chamfer_for_result(r: dict) -> float:
    try:
        xyz = r.get('xyz')
        if xyz is None:
            return float('nan')
        pts_pred = xyz[r['pred_labels'].astype(bool)]
        pts_gt   = xyz[r['gt_labels'].astype(bool)]
        return chamfer_distance(pts_pred, pts_gt)
    except Exception:
        return float('nan')


# ============================================================================
# AVALIAÇÃO
# ============================================================================

def evaluate(results: list) -> dict:
    """
    Métricas por ponto:
      Accuracy, Precision, Recall, F1, IoU (Threshold-dependent)
      AUROC, Average Precision              (Threshold-free)

    Apenas nuvens com ground-truth (has_crack=True) são avaliadas.
    """
    crack_results = [r for r in results if r['has_crack']]
    if not crack_results:
        log.warning("Nenhuma nuvem com rachadura para avaliar.")
        return {}

    all_gt    = np.concatenate([r['gt_labels']  for r in crack_results])
    all_pred  = np.concatenate([r['pred_labels'] for r in crack_results])
    all_score = np.concatenate([r['score']       for r in crack_results])

    # ── Threshold-dependent ───────────────────────────────────────────────────
    acc   = accuracy_score(all_gt, all_pred)
    p, rec, f1, _ = precision_recall_fscore_support(
        all_gt, all_pred, average='binary', zero_division=0)
    

        # F1 Macro — equilibrado entre classes, robusto ao desbalanceamento
    # [Chicco & Jurman, 2020; Faceli et al., 2016]
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        all_gt, all_pred, average='macro', zero_division=0)
    
    iou   = jaccard_score(all_gt, all_pred, zero_division=0)

    try:
        auroc = roc_auc_score(all_gt, all_score)
        ap    = average_precision_score(all_gt, all_score)
    except Exception:
        auroc = ap = float('nan')

    # ── Chamfer Distance ──────────────────────────────────────────────────────
    cd_vals = [_chamfer_for_result(r) for r in crack_results]
    cd_vals = [v for v in cd_vals if np.isfinite(v)]
    cd_mean = float(np.mean(cd_vals)) if cd_vals else float('nan')

    # Métricas por nuvem (para análise de consistência)
    per_cloud = []
    for r in crack_results:
        if r['gt_labels'].sum() == 0:
            continue
        try:
            pc_f1  = precision_recall_fscore_support(
                r['gt_labels'], r['pred_labels'], average='binary', zero_division=0)[2]
            pc_iou = jaccard_score(r['gt_labels'], r['pred_labels'], zero_division=0)
            pc_cd  = _chamfer_for_result(r)
            per_cloud.append({'file': r['filename'], 
                            'f1'          : float(pc_f1),
                            'iou'         : float(pc_iou),
                            'chamfer'     : float(pc_cd),
                            'n_crack_pred': r['pred_labels'].sum(),
                            'n_crack_gt':   r['gt_labels'].sum()})
        except Exception:
            pass

    metrics = {
        # Threshold-dependent
        'accuracy'         : acc,
        'precision'        : p,
        'recall'           : rec,
        'f1'               : f1,
        'f1_macro'          : float(f1_mac),
        'precision_macro'   : float(p_mac),
        'recall_macro'      : float(r_mac),
        'iou'              : iou,
        # Threshold-free
        'auroc'            : auroc,
        'average_precision': ap,
        # Geométrica
        'chamfer_distance'  : float(cd_mean),
        # Metadados
        'n_crack_clouds'   : len(crack_results),
        'n_points_total'   : int(len(all_gt)),
        'n_positive_gt'    : int(all_gt.sum()),
        'per_cloud'        : per_cloud,
    }

    log.info(f"\n{'='*65}")
    log.info(f"RESULTADOS — {len(crack_results)} nuvens com rachadura")
    log.info(f"{'='*65}")
    log.info(f"  [Threshold-dependent]")
    log.info(f"  Accuracy   : {acc:.4f}")
    log.info(f"  Precision  : {p:.4f}")
    log.info(f"  Recall     : {rec:.4f}")
    log.info(f"  F1         : {f1:.4f}   ← target ≥ 0.75")
    log.info(f"  IoU        : {iou:.4f}   ← target ≥ 0.60")
    log.info(f"  [Threshold-free]")
    log.info(f"  AUROC      : {auroc:.4f}   ← target ≥ 0.85")
    log.info(f"  AP         : {ap:.4f}")
    log.info(f"  [Geométrica]")
    log.info(f"  Chamfer Distance : {cd_mean:.6f}  ← menor = melhor")
    log.info(f"{'='*65}")

    if per_cloud:
        log.info("  Por nuvem:")
        for pc in per_cloud:
            log.info(f"    {pc['file']:40s}"  
                     f"F1={pc['f1']:.3f}  IoU={pc['iou']:.3f}"
                     f"CD={pc['chamfer']:.5f}  "
                     f"pred={pc['n_crack_pred']}/{pc['n_crack_gt']} pts")

    return metrics

# ============================================================================
# TESTES ESTATÍSTICOS — Nemenyi / Bonferroni-Dunn
# ============================================================================
def statistical_comparison(model_scores: dict,
                            metric: str = 'f1') -> dict:
    """
    Compara múltiplos modelos com teste não-paramétrico de Wilcoxon
    e correção de Bonferroni-Dunn (equivalente ao teste de Nemenyi
    para comparações pareadas).

    Referência: Faceli et al. (2016) — Inteligência Artificial:
    Uma Abordagem de Aprendizado de Máquina.

    Args:
        model_scores : {'nome_modelo': [score_por_nuvem]}
                       ex: {'prof_aluno': [0.81, 0.74],
                            'gan_v5':     [0.78, 0.69]}
        metric       : nome da métrica (usado apenas no log)

    Returns:
        dict com p-values, significância e vencedor por par.
    """
    from itertools import combinations
    from scipy.stats import wilcoxon, friedmanchisquare

    models = list(model_scores.keys())
    out = {'metric': metric, 'n_models': len(models),
           'pairwise': {}, 'friedman_p': None}

    if len(models) < 2:
        log.warning("São necessários ≥ 2 modelos para comparação estatística.")
        return out

    # Teste de Friedman (omnibus): existe diferença entre algum par?
    try:
        min_len = min(len(model_scores[m]) for m in models)
        arrays  = [model_scores[m][:min_len] for m in models]
        _, p_fr = friedmanchisquare(*arrays)
        out['friedman_p'] = float(p_fr)
        log.info(f"Friedman ({metric}): p = {p_fr:.4f} "
                 f"{'← sig.' if p_fr < 0.05 else ''}")
    except Exception as e:
        log.warning(f"Friedman: {e}")

    # Wilcoxon pareado com correção de Bonferroni
    pairs = list(combinations(models, 2))
    alpha_bonf = 0.05 / len(pairs)

    for m1, m2 in pairs:
        s1 = np.array(model_scores[m1])
        s2 = np.array(model_scores[m2])
        n  = min(len(s1), len(s2))
        s1, s2 = s1[:n], s2[:n]
        try:
            p = 1.0 if np.allclose(s1, s2) else wilcoxon(s1, s2)[1]
            sig = bool(p < alpha_bonf)
            winner = (m1 if np.mean(s1) > np.mean(s2) else m2) if sig else 'empate'
            out['pairwise'][f'{m1} vs {m2}'] = {
                'p_value'   : float(p),
                'significant': sig,
                'alpha_bonf': float(alpha_bonf),
                'winner'    : winner,
                'mean_m1'   : float(np.mean(s1)),
                'mean_m2'   : float(np.mean(s2)),
            }
            log.info(f"  {m1} vs {m2}: p={p:.4f} "
                     f"({'sig.' if sig else 'n.s.'}) → {winner}")
        except Exception as e:
            log.warning(f"Wilcoxon {m1} vs {m2}: {e}")

    return out


# ============================================================================
# CLASSIFICAÇÃO NORMATIVA DE AVARIAS
# ============================================================================
# Categorias baseadas em:
#   ABNT NBR 6118:2014 — Projeto de estruturas de concreto
#   DNIT 061/2004-TER  — Inspeção de obras de arte especiais
#
# A abertura estimada é calculada como o diâmetro máximo da região
# de rachadura predita, aproximado pela distância máxima entre pontos
# de avaria adjacentes (vizinhança k-NN). Em nuvens TLS, 1 unidade ≈ 1 m,
# portanto multiplicamos por 1000 para converter em mm.
# ============================================================================

# Tabela normativa de categorias (abertura em mm)
CRACK_CATEGORIES = [
    ('Microfissura', 0.0,   0.05,  'Quase invisível; geralmente estética'),
    ('Fissura',      0.05,  0.5,   'Pode atingir armadura; requer monitoramento'),
    ('Trinca',       0.5,   1.5,   'Ruptura do elemento; pode comprometer segurança'),
    ('Rachadura',    1.5,   float('inf'), 'Risco estrutural imediato e visível'),
]

# Limites ABNT NBR 6118:2014 por classe de agressividade ambiental
ABNT_LIMITS_MM = {
    'CAA_I_II':   0.3,   # Ambientes rurais/urbanos
    'CAA_III_IV': 0.2,   # Ambientes marinhos/industriais
}

# A Igreja dos Homens Pretos está em Aracati-CE, zona costeira → CAA III
CAA_PROJETO = 'CAA_III_IV'


def classify_crack(abertura_mm: float) -> dict:
    """
    Classifica a abertura estimada conforme ABNT NBR 6118:2014.

    Args:
        abertura_mm: largura estimada da fissura em milímetros

    Returns:
        dict com categoria, descrição, conformidade ABNT e nível de risco
    """
    if not np.isfinite(abertura_mm) or abertura_mm <= 0:
        return {'categoria': 'Indeterminado', 'descricao': '',
                'conforme_abnt': None, 'risco': 'Indeterminado',
                'abertura_mm': abertura_mm}

    categoria = 'Rachadura'
    descricao = CRACK_CATEGORIES[-1][3]

    for nome, lo, hi, desc in CRACK_CATEGORIES:
        if lo <= abertura_mm < hi:
            categoria = nome
            descricao = desc
            break

    limite = ABNT_LIMITS_MM[CAA_PROJETO]
    conforme = abertura_mm <= limite

    # Nível de risco estrutural
    if abertura_mm < 0.05:
        risco = 'Mínimo'
    elif abertura_mm < 0.2:
        risco = 'Baixo'
    elif abertura_mm < 0.5:
        risco = 'Médio — requer monitoramento'
    elif abertura_mm < 1.5:
        risco = 'Alto — intervenção recomendada'
    else:
        risco = 'Crítico — intervenção imediata'

    return {
        'categoria'    : categoria,
        'descricao'    : descricao,
        'abertura_mm'  : float(abertura_mm),
        'limite_abnt'  : limite,
        'caa'          : CAA_PROJETO,
        'conforme_abnt': conforme,
        'risco'        : risco,
    }

def compute_severity(results: list) -> list:
    """
    Para cada nuvem com rachadura, calcula:
      - severidade_pct : proporção de pontos preditos como avaria (%)
      - abertura_mm    : estimativa da abertura máxima via k-NN entre
                         pontos de avaria adjacentes (em mm)
      - profundidade_mm: distância máxima do ponto de avaria ao plano
                         local estimado por PCA nos pontos normais
      - classificacao  : categoria normativa (ABNT NBR 6118 / DNIT)

    A abertura é estimada como a mediana das distâncias ao 2º vizinho
    mais próximo dentro da região de avaria, multiplicada por 1000
    (conversão metros → mm, válida para nuvens TLS em metros).

    Referências:
      ABNT NBR 6118:2014 — Projeto de estruturas de concreto
      DNIT 061/2004-TER  — Inspeção de obras de arte especiais
    """
    for r in results:
        if not r['has_crack'] or r['pred_labels'].sum() == 0:
            r.update({'severidade_pct': 0.0, 'abertura_mm': 0.0,
                      'profundidade_mm': 0.0,
                      'classificacao': classify_crack(0.0)})
            continue

        xyz      = r.get('xyz')
        pred     = r['pred_labels'].astype(bool)
        n_pred   = int(pred.sum())

        # ── Severidade ────────────────────────────────────────────────────
        r['severidade_pct'] = float(pred.mean() * 100)

        if xyz is None or n_pred < 5:
            r.update({'abertura_mm': float('nan'),
                      'profundidade_mm': float('nan'),
                      'classificacao': classify_crack(float('nan'))})
            continue

        crack_pts  = xyz[pred]
        normal_pts = xyz[~pred]

        # ── Abertura: distância mediana ao 2º vizinho dentro da avaria ───
        # O 2º vizinho (k=2, índice 1) exclui o próprio ponto.
        # Para evitar viés de densidade, usamos a mediana em vez da média.
        try:
            k_ab   = min(3, len(crack_pts))
            tree_c = cKDTree(crack_pts)
            dists, _ = tree_c.query(crack_pts, k=k_ab)
            # dists[:, 0] = distância ao próprio ponto (0.0), pular
            nearest = dists[:, 1] if k_ab >= 2 else dists[:, 0]
            abertura_m  = float(np.median(nearest))
            abertura_mm = abertura_m * 1000.0
        except Exception:
            abertura_mm = float('nan')

        r['abertura_mm'] = abertura_mm

        # ── Profundidade: distância ao plano local (PCA nos normais) ─────
        try:
            if len(normal_pts) >= 10:
                centroid   = normal_pts.mean(axis=0)
                centered   = normal_pts - centroid
                _, _, Vt   = np.linalg.svd(centered, full_matrices=False)
                normal_vec = Vt[-1]             # menor componente = normal ao plano
                dists_prof = np.abs((crack_pts - centroid) @ normal_vec)
                r['profundidade_mm'] = float(dists_prof.max() * 1000.0)
            else:
                r['profundidade_mm'] = float('nan')
        except Exception:
            r['profundidade_mm'] = float('nan')

        # ── Classificação normativa ───────────────────────────────────────
        r['classificacao'] = classify_crack(abertura_mm)

    return results


def severity_report(results: list) -> dict:
    """
    Consolida as avaliações de severidade em um relatório por categoria.

    Retorna dict com:
      - contagem por categoria (Microfissura, Fissura, Trinca, Rachadura)
      - conformidade ABNT geral (% de nuvens conformes)
      - distribuição de risco
    """
    crack_res = [r for r in results
                 if r['has_crack'] and r.get('classificacao')]

    categorias  = {}
    riscos      = {}
    n_conforme  = 0
    n_total     = 0

    for r in crack_res:
        cl  = r['classificacao']
        cat = cl.get('categoria', 'Indeterminado')
        risco = cl.get('risco', 'Indeterminado')
        conforme = cl.get('conforme_abnt')

        categorias[cat]   = categorias.get(cat, 0) + 1
        riscos[risco]     = riscos.get(risco, 0) + 1

        if conforme is not None:
            n_total += 1
            if conforme:
                n_conforme += 1

    pct_conforme = (n_conforme / n_total * 100) if n_total > 0 else float('nan')

    log.info(f"\n{'='*65}")
    log.info(f"RELATÓRIO DE SEVERIDADE — ABNT NBR 6118:2014 (CAA: {CAA_PROJETO})")
    log.info(f"{'='*65}")
    log.info(f"  Limite de abertura ({CAA_PROJETO}): "
             f"{ABNT_LIMITS_MM[CAA_PROJETO]} mm")
    log.info(f"  Nuvens avaliadas: {len(crack_res)}")
    log.info(f"  Conformidade ABNT: {pct_conforme:.1f}% das nuvens")
    log.info(f"\n  Distribuição por categoria:")
    for cat, n in sorted(categorias.items(),
                         key=lambda x: ['Microfissura','Fissura',
                                        'Trinca','Rachadura'].index(x[0])
                         if x[0] in ['Microfissura','Fissura',
                                     'Trinca','Rachadura'] else 99):
        log.info(f"    {cat:15s}: {n:3d} nuvens")
    log.info(f"\n  Distribuição por risco:")
    for risco, n in riscos.items():
        log.info(f"    {risco:40s}: {n} nuvens")
    log.info(f"{'='*65}")

    return {
        'categorias'     : categorias,
        'riscos'         : riscos,
        'pct_conforme'   : float(pct_conforme),
        'limite_abnt_mm' : ABNT_LIMITS_MM[CAA_PROJETO],
        'caa'            : CAA_PROJETO,
        'n_avaliadas'    : len(crack_res),
    }


# ============================================================================
# CALIBRAÇÃO DO THRESHOLD
# ============================================================================

def calibrate_threshold_f1(results: list) -> float:
    """
    Estratégia 1 — Threshold que maximiza o F1-Score (β=1).

    Encontra o ponto ótimo na curva Precision-Recall onde
    F1 = 2·P·R / (P+R) é máximo.

    Limitação: sensível ao desequilíbrio de classes — quando a fração
    de positivos muda (entre datasets), o threshold ótimo se desloca.
    """
    crack_res = [r for r in results if r['has_crack']]
    if not crack_res:
        return ANOMALY_PCTL / 100.0

    all_score = np.concatenate([r['score']     for r in crack_res])
    all_gt    = np.concatenate([r['gt_labels'] for r in crack_res])

    prec, rec, thrs = precision_recall_curve(all_gt, all_score)
    f1s    = 2 * prec * rec / (prec + rec + 1e-8)
    best_i = np.argmax(f1s[:-1])
    best_t = float(thrs[best_i])

    log.info(f"[Estratégia 1 — F1] threshold={best_t:.4f} "
             f"→ P={prec[best_i]:.3f}  R={rec[best_i]:.3f}  F1={f1s[best_i]:.3f}")
    return best_t


def calibrate_threshold_gmean(results: list) -> float:
    """
    Estratégia 2 — Threshold que maximiza o G-mean.

    G-mean = √(TPR × TNR)  =  √(Recall × Especificidade)

    Vantagem sobre F1 em datasets muito desequilibrados: o G-mean penaliza
    igualmente erros nas duas classes — não é afetado pela proporção de
    positivos (~6.7% para rachaduras). Permanece estável quando a taxa de
    positivos varia entre conjuntos de dados distintos.

    Referência: Kubat & Matwin (1997) — Addressing the Curse of
    Imbalancedness: One-sided Selection.
    """
    crack_res = [r for r in results if r['has_crack']]
    if not crack_res:
        return ANOMALY_PCTL / 100.0

    all_score = np.concatenate([r['score']     for r in crack_res])
    all_gt    = np.concatenate([r['gt_labels'] for r in crack_res])

    # Curva ROC fornece TPR e FPR em cada threshold
    fpr, tpr, thrs = roc_curve(all_gt, all_score)
    tnr    = 1.0 - fpr                           # especificidade = TNR
    gmean  = np.sqrt(tpr * tnr)                  # √(TPR × TNR)

    # Excluir o ponto extra (threshold=∞) que roc_curve adiciona
    best_i = int(np.argmax(gmean[:-1]))
    best_t = float(thrs[best_i])

    log.info(f"[Estratégia 2 — G-mean] threshold={best_t:.4f} "
             f"→ TPR={tpr[best_i]:.3f}  TNR={tnr[best_i]:.3f}  "
             f"G-mean={gmean[best_i]:.3f}")
    return best_t


def calibrate_threshold_fbeta(results: list, beta: float = 0.5) -> float:
    """
    Estratégia 3 — Threshold que maximiza o F-beta (β < 1 → prioriza precisão).

    F_β = (1 + β²) · P · R / (β² · P + R)

    β < 1: precisão tem mais peso que recall
      β=0.5 → precisão vale 4× mais que recall
      β=1.0 → equivale ao F1 padrão

    Justificativa para inspeção estrutural:
      - Falso positivo: parede sã marcada como rachada → inspeção desnecessária
      - Falso negativo: rachadura não detectada → risco estrutural real
    β=0.5 assume que o custo de uma inspeção falsa é ≈ 1/4 do risco de
    não detectar uma rachadura real — ajuste este valor conforme o contexto.

    Referência: Van Rijsbergen (1979) — Information Retrieval, 2ª ed.
    """
    crack_res = [r for r in results if r['has_crack']]
    if not crack_res:
        return ANOMALY_PCTL / 100.0

    all_score = np.concatenate([r['score']     for r in crack_res])
    all_gt    = np.concatenate([r['gt_labels'] for r in crack_res])

    prec, rec, thrs = precision_recall_curve(all_gt, all_score)
    b2      = beta ** 2
    fbeta   = (1 + b2) * prec * rec / (b2 * prec + rec + 1e-8)
    best_i  = np.argmax(fbeta[:-1])
    best_t  = float(thrs[best_i])

    log.info(f"[Estratégia 3 — F-beta β={beta}] threshold={best_t:.4f} "
             f"→ P={prec[best_i]:.3f}  R={rec[best_i]:.3f}  "
             f"F{beta}={fbeta[best_i]:.3f}")
    return best_t


# ============================================================================
# FILTRO DE COERÊNCIA ESPACIAL
# ============================================================================

def apply_spatial_coherence(results: list,
                             min_neighbors: int = 3,
                             k: int = 20) -> list:
    """
    Pós-processamento: reverte pred_labels=1 de pontos isolados para 0.

    Um ponto é mantido como rachadura apenas se ≥ `min_neighbors` de seus
    `k` vizinhos mais próximos também estão marcados como rachadura.
    Pontos isolados (falsos positivos típicos de superfície ruidosa) são
    revertidos para normal.

    Parâmetros:
      min_neighbors : mínimo de vizinhos crack para manter a predição (padrão 3)
      k             : vizinhos K-NN a considerar (padrão 20)

    Não altera `score` — apenas `pred_labels`. AUROC permanece intacto.
    """
    for r in results:
        if 'pred_labels' not in r:
            continue

        pred = r['pred_labels'].copy()
        n    = len(pred)
        if pred.sum() == 0 or n < k + 1:
            r['pred_labels'] = pred
            continue

        # Coordenadas XYZ (primeiros 3 campos das features brutas, ou de 'features')
        # Suporte a resultados com ou sem campo 'xyz'
        if 'xyz' in r:
            xyz = np.asarray(r['xyz'], dtype=np.float32)
        elif 'features' in r:
            xyz = np.asarray(r['features'])[:, :3].astype(np.float32)
        else:
            # Sem coordenadas: não aplica filtro nesta nuvem
            continue

        k_act = min(k, n - 1)
        tree  = cKDTree(xyz)
        # Para cada ponto marcado como crack, conta vizinhos também marcados
        crack_idx = np.where(pred == 1)[0]
        if len(crack_idx) == 0:
            continue

        _, neigh_idx = tree.query(xyz[crack_idx], k=k_act + 1)
        neigh_idx    = neigh_idx[:, 1:]   # excluir o próprio ponto

        crack_set    = set(crack_idx.tolist())
        for i, ci in enumerate(crack_idx):
            n_crack_neigh = sum(1 for ni in neigh_idx[i] if ni in crack_set)
            if n_crack_neigh < min_neighbors:
                pred[ci] = 0   # reverter ponto isolado

        r['pred_labels'] = pred

    return results


# ============================================================================
# VISUALIZAÇÕES
# ============================================================================
def visualize_cracks(results: list,
                     save_dir: str,
                     max_clouds: int = 10,
                     ts: str = None) -> None:
    """
    Para cada nuvem com avaria, salva:
      1. .ply com pontos coloridos por classificação:
           vermelho  = avaria predita corretamente (TP)
           laranja   = falso positivo (FP)
           azul      = falso negativo (FN — avaria não detectada)
           cinza     = normal
      2. .png com 3 subplots:
           score heatmap | predição | TP/FP/FN
      3. linha no log com categoria ABNT e abertura estimada
    """
    os.makedirs(save_dir, exist_ok=True)

    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')
    crack_res = [r for r in results
                 if r['has_crack'] and r.get('xyz') is not None]
    crack_res.sort(key=lambda r: r['pred_labels'].sum(), reverse=True)

    for r in crack_res[:max_clouds]:
        fname = r['filename'].replace('.ply', '')
        xyz   = r['xyz']
        score = r['score']
        pred  = r['pred_labels'].astype(bool)
        gt    = r['gt_labels'].astype(bool)
        cl    = r.get('classificacao', {})

        tp = pred & gt
        fp = pred & ~gt
        fn = ~pred & gt

        # ── PLY colorido + scalar_field ───────────────────────────────────
        # Usa cores reais da nuvem; destaca apenas a rachadura predita em vermelho.
        rgb_orig = r.get('rgb')   # (N, 3) em [0, 1], pode ser None
        if rgb_orig is not None:
            colors = (np.clip(rgb_orig, 0.0, 1.0) * 255).astype(np.uint8)
        else:
            colors = np.full((len(xyz), 3), 200, dtype=np.uint8)

        colors[pred] = [255, 0, 0]   # vermelho — rachadura detectada (TP + FP)

        sf_vals = r.get('scalar_field')   # (N,) ou None

        ply_path = os.path.join(save_dir, f'{fname}_avarias_{ts}.ply')

        if sf_vals is not None:
            # Salva PLY binário com scalar_field e anomaly_score como propriedades extras.
            # Usa PlyData/PlyElement para escrita eficiente (sem loop Python por ponto).
            from plyfile import PlyElement
            n = len(xyz)
            xy    = xyz.astype(np.float32)
            sf_arr = np.asarray(sf_vals, dtype=np.float32)
            sc_arr = score.astype(np.float32)
            vertex_data = np.empty(n, dtype=[
                ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
                ('scalar_field', 'f4'),
                ('anomaly_score', 'f4'),
            ])
            vertex_data['x'] = xy[:, 0]
            vertex_data['y'] = xy[:, 1]
            vertex_data['z'] = xy[:, 2]
            vertex_data['red']   = colors[:, 0]
            vertex_data['green'] = colors[:, 1]
            vertex_data['blue']  = colors[:, 2]
            vertex_data['scalar_field']  = sf_arr
            vertex_data['anomaly_score'] = sc_arr
            PlyData([PlyElement.describe(vertex_data, 'vertex')]).write(ply_path)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(
                colors.astype(np.float64) / 255.0)
            o3d.io.write_point_cloud(ply_path, pcd)

        # ── PNG com 3 subplots ────────────────────────────────────────────
        categoria  = cl.get('categoria', '?')
        abertura   = cl.get('abertura_mm', float('nan'))
        risco      = cl.get('risco', '?')
        conforme   = cl.get('conforme_abnt')
        limite_val = cl.get('limite_abnt')
        if conforme is None:
            abnt_str = 'Abertura indeterminada'
        elif conforme:
            abnt_str = 'Conforme ABNT'
        elif isinstance(limite_val, (int, float)):
            abnt_str = f'Fora do limite ABNT ({limite_val:.1f} mm)'
        else:
            abnt_str = 'Fora do limite ABNT'

        titulo = (f"{fname}  |  {categoria}  |  "
                  f"Abertura: {abertura:.3f} mm  |  {risco}  |  {abnt_str}")

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(titulo, fontsize=9)

        # Projeção XZ: X horizontal (largura da parede), Z vertical (altura)
        # Vista frontal — revela rachaduras verticais/horizontais na face da parede
        sc0 = axes[0].scatter(xyz[:,0], xyz[:,2], c=score,
                              cmap='RdYlGn_r', s=0.5, alpha=0.8,
                              vmin=0, vmax=1)
        axes[0].set_title('Score de anomalia (vermelho = alto)')
        plt.colorbar(sc0, ax=axes[0])

        c_pred = np.where(pred, 'red', 'lightgray')
        axes[1].scatter(xyz[:,0], xyz[:,2], c=c_pred, s=0.5, alpha=0.7)
        axes[1].set_title('Avarias preditas (vermelho)')

        for mask, col, lbl in [
            (~pred & ~gt, 'lightgray', 'Normal'),
            (tp,          'red',       f'Detectado (TP) — {tp.sum():,}'),
            (fp,          'orange',    f'Falso positivo (FP) — {fp.sum():,}'),
            (fn,          'blue',      f'Não detectado (FN) — {fn.sum():,}'),
        ]:
            if mask.sum() > 0:
                axes[2].scatter(xyz[mask,0], xyz[mask,2],
                                c=col, s=0.5, alpha=0.7, label=lbl)
        axes[2].legend(markerscale=6, fontsize=7)
        axes[2].set_title('TP / FP / FN')

        for ax in axes:
            ax.set_aspect('equal')
            ax.set_xlabel('X (m)', fontsize=7)
            ax.set_ylabel('Z (m)', fontsize=7)
            ax.tick_params(labelsize=6)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{fname}_avarias_{ts}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        log.info(
            f"  {fname:35s}  "
            f"{categoria:12s}  "
            f"abertura={abertura:.3f}mm  "
            f"prof={r.get('profundidade_mm', float('nan')):.2f}mm  "
            f"sev={r.get('severidade_pct',0):.1f}%  "
            f"{'✅' if conforme else '❌'} ABNT"
        )



def save_results(metrics: dict, results: list, history: dict,
                 thr: float, gmm_info: dict, save_dir: str,
                 ts: str = None) -> None:
    os.makedirs(save_dir, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')

    def _to_py(v):
        if isinstance(v, (np.floating, np.integer)): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        return v

    out = {k: _to_py(v) for k, v in metrics.items() if k != 'per_cloud'}
    out.update({'threshold': thr, 'gmm': gmm_info,
                'timestamp': datetime.now().isoformat()})

    with open(os.path.join(save_dir, f'metrics_{ts}.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

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


def plot_training_history(history: dict, save_dir: str, ts: str = None):
    os.makedirs(save_dir, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')
    epochs = range(1, len(history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Professor-Aluno — Reverse Distillation', fontsize=13)
    axes[0].plot(epochs, history['loss'], color='#388bfd', lw=2)
    axes[0].set_title('Distillation Loss'); axes[0].set_xlabel('Epoch'); axes[0].grid(True, alpha=0.3)
    axes[1].plot(epochs, history['lr'], color='#3fb950', lw=2)
    axes[1].set_title('Learning Rate (Student)'); axes[1].set_xlabel('Epoch'); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ts_training_history_{ts}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_score_distribution(results: list, thr: float,
                            save_dir: str, gmm_info: dict = None,
                            ts: str = None):
    os.makedirs(save_dir, exist_ok=True)
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')
    crack_res = [r for r in results if r['has_crack']]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Distribuição do Score de Anomalia — Professor-Aluno', fontsize=13)
    ax = axes[0]
    if crack_res:
        gt_pos = np.concatenate([r['score'][r['gt_labels']==1] for r in crack_res if r['gt_labels'].sum()>0])
        gt_neg = np.concatenate([r['score'][r['gt_labels']==0] for r in crack_res])
        ax.hist(gt_neg, bins=80, alpha=0.6, label='Normal (GT)',   color='#388bfd', density=True)
        if len(gt_pos): ax.hist(gt_pos, bins=80, alpha=0.6, label='Rachadura (GT)', color='#f85149', density=True)
    ax.axvline(thr, color='#3fb950', lw=2.5, linestyle='--',
               label=f"Thr={thr:.3f} ({gmm_info.get('method','?') if gmm_info else '?'})")
    ax.set_xlabel('Score de Anomalia'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax2 = axes[1]
    if crack_res:
        gt    = np.concatenate([r['gt_labels'] for r in crack_res])
        score = np.concatenate([r['score']     for r in crack_res])
        try:
            fpr, tpr, _ = roc_curve(gt, score)
            auroc = roc_auc_score(gt, score)
            ax2.plot(fpr, tpr, lw=2, color='#388bfd', label=f'ROC (AUROC={auroc:.3f})')
            ax2.plot([0,1],[0,1],'k--', lw=0.8); ax2.set_title('Curva ROC')
            ax2.set_xlabel('FPR'); ax2.set_ylabel('TPR'); ax2.legend(); ax2.grid(True, alpha=0.3)
        except Exception: pass
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'ts_score_distribution_{ts}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================================
# ABLATION E COMPARAÇÃO DE MODELOS
# ============================================================================

def evaluate_ablation(
    results_raw: list,
    sf_applied: bool = False,
) -> pd.DataFrame:
    """
    Avalia contribuição marginal de cada componente do v2.
    Recebe results já com 'score', 'scalar_field', 'gt_labels'.
    Retorna DataFrame para tabela de ablation do TCC.

    Configs testadas:
      distill_only   : score bruto sem gate nem fusion
      distill+gate   : score * soft_weight (gate soft GMM)
      distill+fusion : 0.7*score + 0.3*sf_gmm
      v2_completo    : fusion + gate soft
    """
    import copy

    configs = [
        {'name': 'distill_only',    'fusion': False, 'gate': False},
        {'name': 'distill+gate',    'fusion': False, 'gate': True },
        {'name': 'distill+fusion',  'fusion': True,  'gate': False},
        {'name': 'v2_completo',     'fusion': True,  'gate': True },
    ]

    rows = []
    for cfg in configs:
        res = copy.deepcopy(results_raw)

        for r in res:
            sf_raw = r.get('scalar_field')
            score  = r['score'].copy()

            if sf_raw is not None:
                gmm = ScalarFieldGMM(sf_raw).fit()
                sf_prob = gmm.anomaly_probability()
                soft_w  = gmm.soft_weights()

                if cfg['fusion']:
                    lo, hi = score.min(), score.max()
                    dist_n = (score - lo) / (hi - lo + 1e-8)
                    score  = 0.7 * dist_n + 0.3 * sf_prob

                if cfg['gate']:
                    score = score * soft_w

            r['score'] = score

        thr, _ = fit_gmm_threshold(res)
        res     = apply_threshold(res, thr)
        m       = evaluate(res)

        rows.append({
            'config'           : cfg['name'],
            'threshold'        : round(thr, 6),
            'precision'        : round(m.get('precision', 0), 4),
            'recall'           : round(m.get('recall', 0), 4),
            'f1'               : round(m.get('f1', 0), 4),
            'f1_macro'         : round(m.get('f1_macro', 0), 4),
            'iou'              : round(m.get('iou', 0), 4),
            'auroc'            : round(m.get('auroc', 0), 4),
            'average_precision': round(m.get('average_precision', 0), 4),
        })

    df = pd.DataFrame(rows)
    log.info("\nAblation study:\n" + df.to_string(index=False))
    return df


def compare_models(
    results_v1: list,
    results_v2: list,
    results_unsup: list,
    output_dir: str,
    ts: str = None,
) -> pd.DataFrame:
    """
    Compara métricas entre v1, v2 e unsup.
    Exporta CSV e executa Wilcoxon pareado entre os três.

    Args:
        results_v1/v2/unsup : listas com 'pred_labels', 'gt_labels', 'score', 'has_crack'
        output_dir          : diretório para salvar o CSV
        ts                  : timestamp para nome do arquivo

    Returns:
        DataFrame com métricas agregadas por modelo.
    """
    if ts is None:
        ts = datetime.now().strftime('%d%m%Y_%H%M')

    model_results = {
        'v1'   : results_v1,
        'v2'   : results_v2,
        'unsup': results_unsup,
    }

    rows = []
    per_cloud_scores = {name: [] for name in model_results}

    for name, res in model_results.items():
        m = evaluate(res)
        rows.append({
            'model'            : name,
            'precision'        : round(m.get('precision', 0), 4),
            'recall'           : round(m.get('recall', 0), 4),
            'f1'               : round(m.get('f1', 0), 4),
            'f1_macro'         : round(m.get('f1_macro', 0), 4),
            'iou'              : round(m.get('iou', 0), 4),
            'auroc'            : round(m.get('auroc', 0), 4),
            'average_precision': round(m.get('average_precision', 0), 4),
            'chamfer_distance' : round(m.get('chamfer_distance', float('nan')), 8),
        })
        per_cloud_scores[name] = [
            pc['f1'] for pc in m.get('per_cloud', [])
        ]

    df = pd.DataFrame(rows)

    # Wilcoxon pareado
    stat_results = statistical_comparison(per_cloud_scores, metric='f1')

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f'model_comparison_{ts}.csv')
    df.to_csv(csv_path, index=False)
    log.info(f"Comparação de modelos salva: {csv_path}")
    log.info("\nComparação de modelos:\n" + df.to_string(index=False))

    return df

