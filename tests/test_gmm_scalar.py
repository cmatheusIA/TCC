import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from gmm_scalar import fit_gmm, crack_score_from_gmm, otsu_score


def _bimodal_sf(n_normal=900, n_crack=100, seed=0):
    rng = np.random.default_rng(seed)
    normal = rng.normal(200, 15, n_normal).astype(np.float32)
    crack  = rng.normal(40,  10, n_crack).astype(np.float32)
    sf = np.concatenate([normal, crack])
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_crack)]).astype(np.int64)
    return sf, labels


def test_fit_gmm_returns_two_components():
    sf, _ = _bimodal_sf()
    gmm, lower_idx = fit_gmm(sf)
    assert gmm.n_components == 2
    assert lower_idx in (0, 1)


def test_fit_gmm_lower_mode_is_crack():
    sf, labels = _bimodal_sf()
    gmm, lower_idx = fit_gmm(sf)
    scores = crack_score_from_gmm(gmm, sf, lower_idx)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(labels, scores)
    assert auroc > 0.85, f"AUROC esperado >0.85, obtido {auroc:.3f}"


def test_crack_score_range():
    sf, _ = _bimodal_sf()
    gmm, lower_idx = fit_gmm(sf)
    scores = crack_score_from_gmm(gmm, sf, lower_idx)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_otsu_score_fallback():
    sf = np.concatenate([
        np.random.default_rng(0).normal(200, 5, 900),
        np.random.default_rng(1).normal(40, 5, 100)
    ]).astype(np.float32)
    scores = otsu_score(sf)
    assert len(scores) == len(sf)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0
