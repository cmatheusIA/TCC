import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pytest
from utils.scalar_features import iqr_normalize_sf, extract_local_sf_features


def _make_cloud(n=200, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.random((n, 3)).astype(np.float32)
    features = rng.random((n, 16)).astype(np.float32)
    # col 9 = scalar_field, col 6-8 = normals
    features[:, 9] = rng.uniform(0, 100, n).astype(np.float32)
    return xyz, features


def test_iqr_normalize_shape():
    _, features = _make_cloud(200)
    sf = features[:, 9]
    normed = iqr_normalize_sf(sf)
    assert normed.shape == sf.shape
    assert normed.dtype == np.float32


def test_iqr_normalize_median_zero():
    sf = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    normed = iqr_normalize_sf(sf)
    assert abs(float(normed[2])) < 1e-5  # median maps to 0


def test_iqr_normalize_constant():
    sf = np.ones(50, dtype=np.float32) * 42.0
    normed = iqr_normalize_sf(sf)
    assert np.allclose(normed, 0.0, atol=1e-5)


def test_extract_local_sf_features_shape():
    _, features = _make_cloud(200)
    result = extract_local_sf_features(features, k=16)
    assert result.shape == (200, 8)
    assert result.dtype == np.float32


def test_extract_local_sf_features_finite():
    _, features = _make_cloud(200)
    result = extract_local_sf_features(features, k=16)
    assert np.all(np.isfinite(result))


def test_extract_local_sf_features_small_cloud():
    # k > N should not crash — k is clamped internally
    _, features = _make_cloud(10)
    result = extract_local_sf_features(features, k=32)
    assert result.shape == (10, 8)
