import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np
import pytest
from scalar_memory import coreset_sampling, build_memory_bank, compute_anomaly_scores, calibrate_threshold


def _fake_cloud(n=300, crack_ratio=0.0, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.random((n, 16)).astype(np.float32)
    features[:, 9] = rng.uniform(50, 100, n).astype(np.float32)   # normal sf range
    n_crack = int(n * crack_ratio)
    labels = np.zeros(n, dtype=np.int64)
    if n_crack > 0:
        features[:n_crack, 9] = rng.uniform(0, 20, n_crack).astype(np.float32)  # crack sf
        # Cluster crack points spatially so kNN neighbors are mostly other crack points
        features[:n_crack, :3] = rng.uniform(0, 0.05, (n_crack, 3)).astype(np.float32)
        labels[:n_crack] = 1
    return {'features': features, 'labels': labels, 'filename': f'test_{seed}.ply',
            'has_crack': crack_ratio > 0, 'n_points': n}


def test_coreset_sampling_size():
    X = np.random.rand(500, 8).astype(np.float32)
    idx = coreset_sampling(X, K=100)
    assert len(idx) == 100
    assert len(set(idx.tolist())) == 100  # unique


def test_coreset_sampling_less_than_K():
    X = np.random.rand(50, 8).astype(np.float32)
    idx = coreset_sampling(X, K=200)
    assert len(idx) == 50  # capped at N


def test_build_memory_bank_shapes():
    normals = [_fake_cloud(300, seed=i) for i in range(5)]
    bank, cov_inv = build_memory_bank(normals, K=64, k=16)
    assert bank.shape[1] == 8
    assert bank.shape[0] <= 64
    assert cov_inv.shape == (8, 8)


def test_compute_anomaly_scores_shape():
    normals = [_fake_cloud(300, seed=i) for i in range(5)]
    bank, cov_inv = build_memory_bank(normals, K=64, k=16)
    cloud = _fake_cloud(200, crack_ratio=0.1, seed=99)
    scores = compute_anomaly_scores(cloud, bank, cov_inv, m=3, k=16)
    assert scores.shape == (200,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0)


def test_calibrate_threshold_scalar():
    normals = [_fake_cloud(300, seed=i) for i in range(10)]
    bank, cov_inv = build_memory_bank(normals, K=128, k=16)
    thr = calibrate_threshold(normals, bank, cov_inv, percentile=99, k=16)
    assert isinstance(float(thr), float)
    assert thr > 0


def test_crack_scores_higher_than_normal():
    """Crack points should have higher anomaly scores than normal points."""
    normals = [_fake_cloud(400, seed=i) for i in range(15)]
    bank, cov_inv = build_memory_bank(normals, K=256, k=16)
    crack_cloud = _fake_cloud(300, crack_ratio=0.3, seed=99)
    scores = compute_anomaly_scores(crack_cloud, bank, cov_inv, m=3, k=16)
    crack_mask = crack_cloud['labels'] == 1
    mean_crack  = scores[crack_mask].mean()
    mean_normal = scores[~crack_mask].mean()
    assert mean_crack > mean_normal, (
        f"Crack scores ({mean_crack:.3f}) should be > normal scores ({mean_normal:.3f})"
    )
