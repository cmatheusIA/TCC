# tests/test_teacher_student_v2.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import torch
import torch.nn as nn


# ── Dados sintéticos ──────────────────────────────────────────────────────────

def make_fake_result(n=100, has_crack=True, seed=0):
    """Cria um result dict sintético com scalar_field bimodal."""
    rng = np.random.default_rng(seed)
    gt = np.zeros(n, dtype=np.int64)
    gt[:20] = 1

    # Scalar_field: crack=[0,29], normal=[30,100]
    sf = np.concatenate([
        rng.uniform(0, 29, 20),
        rng.uniform(30, 100, 80),
    ]).astype(np.float32)

    score = rng.uniform(0, 1, n).astype(np.float32)
    # Crack points têm score maior
    score[:20] += 0.4
    score = score.clip(0, 1)

    return {
        'filename'    : 'avaria_test.ply',
        'has_crack'   : has_crack,
        'score'       : score,
        'gt_labels'   : gt,
        'pred_labels' : (score > 0.5).astype(np.int64),
        'n_points'    : n,
        'xyz'         : rng.standard_normal((n, 3)).astype(np.float32),
        'rgb'         : rng.uniform(0, 1, (n, 3)).astype(np.float32),
        'scalar_field': sf,
    }


# ── Teste integração 1: score fusion ─────────────────────────────────────────

def test_score_fusion_changes_score():
    """v2 score fusion deve alterar o score original."""
    from utils.evaluation import ScalarFieldGMM
    r = make_fake_result()
    score_orig = r['score'].copy()
    sf_raw     = r['scalar_field']

    gmm      = ScalarFieldGMM(sf_raw).fit()
    sf_score = gmm.anomaly_probability()

    lo, hi    = score_orig.min(), score_orig.max()
    dist_norm = (score_orig - lo) / (hi - lo + 1e-8)
    score_v2  = 0.7 * dist_norm + 0.3 * sf_score

    assert not np.allclose(score_orig, score_v2), "Score v2 deve diferir do original"
    assert score_v2.min() >= 0.0 and score_v2.max() <= 1.0

def test_score_fusion_unimodal_transparent():
    """Para nuvem unimodal, score fusion deve ser próximo ao distill_norm."""
    from utils.evaluation import ScalarFieldGMM
    rng = np.random.default_rng(1)
    # Scalar_field unimodal
    sf_raw     = rng.normal(80, 10, 200).astype(np.float32)
    score_orig = rng.uniform(0, 1, 200).astype(np.float32)

    gmm      = ScalarFieldGMM(sf_raw).fit()
    sf_score = gmm.anomaly_probability()

    lo, hi    = score_orig.min(), score_orig.max()
    dist_norm = (score_orig - lo) / (hi - lo + 1e-8)
    score_v2  = 0.7 * dist_norm + 0.3 * sf_score

    # sf_score ≈ 0.5 para unimodal → score_v2 ≈ 0.7*dist_norm + 0.15
    expected = 0.7 * dist_norm + 0.3 * 0.5
    assert np.allclose(score_v2, expected, atol=0.05)


# ── Teste integração 2: gate soft ─────────────────────────────────────────────

def test_soft_gate_suppresses_normal_points():
    """Pontos claramente normais (alto scalar_field) devem ter score modulado para baixo."""
    from utils.evaluation import ScalarFieldGMM
    r = make_fake_result()
    sf_raw = r['scalar_field']

    gmm     = ScalarFieldGMM(sf_raw).fit()
    weights = gmm.soft_weights()

    score_orig     = r['score'].copy()
    score_modulated = score_orig * weights

    # Pontos normais (sf > 30) devem ter score reduzido
    normal_mask = sf_raw > 30
    assert (score_modulated[normal_mask] <= score_orig[normal_mask]).all()

def test_soft_gate_unimodal_all_ones():
    """Para unimodal, soft gate deve ser transparente (weights=1.0)."""
    from utils.evaluation import ScalarFieldGMM
    rng = np.random.default_rng(2)
    sf_raw = rng.normal(80, 10, 200).astype(np.float32)
    gmm    = ScalarFieldGMM(sf_raw).fit()
    weights = gmm.soft_weights()
    assert np.allclose(weights, 1.0)


# ── Teste integração 3: Push-Pull ponderado ───────────────────────────────────

def test_pseudo_label_confidence_reduces_boundary_gradient():
    """
    Pontos no vale (ambíguos) devem ter confidence baixa → gradiente reduzido.
    """
    from utils.evaluation import ScalarFieldGMM
    # Scalar_field com vale claro em ~30
    sf_bimodal = np.concatenate([
        np.random.uniform(0, 25, 100).astype(np.float32),
        np.random.uniform(35, 100, 100).astype(np.float32),
    ])
    # Pontos no vale
    sf_valley = np.array([27.0, 28.5, 30.0, 31.0, 32.0], dtype=np.float32)

    gmm      = ScalarFieldGMM(sf_bimodal).fit()
    conf_all = gmm.pseudo_label_confidence()

    # Refazer com valley points incluídos
    sf_with_valley = np.concatenate([sf_bimodal, sf_valley])
    gmm2 = ScalarFieldGMM(sf_with_valley).fit()
    conf2 = gmm2.pseudo_label_confidence()

    # Pontos no vale (últimos 5) devem ter confidence < pontos nos núcleos
    valley_conf = conf2[-5:]
    core_conf   = conf2[:10]   # pontos deep crack (0-25)
    assert valley_conf.mean() < core_conf.mean(), \
        f"Vale conf={valley_conf.mean():.3f} deve ser < núcleo conf={core_conf.mean():.3f}"
