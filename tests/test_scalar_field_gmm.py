# tests/test_scalar_field_gmm.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import tempfile


def make_bimodal(n_crack=200, n_normal=800, seed=42):
    rng = np.random.default_rng(seed)
    crack  = rng.uniform(0, 30, n_crack).astype(np.float32)
    normal = rng.uniform(50, 120, n_normal).astype(np.float32)
    return np.concatenate([crack, normal])


def make_unimodal(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    return rng.normal(80, 15, n).astype(np.float32)


def test_bimodal_detection():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_bimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    assert gmm.modality == 'bimodal', f"Esperado bimodal, got {gmm.modality}"
    assert 30 <= gmm.threshold <= 50, f"Threshold {gmm.threshold} fora do esperado [30,50]"


def test_unimodal_detection():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    assert gmm.modality == 'unimodal', f"Esperado unimodal, got {gmm.modality}"


def test_anomaly_probability_bimodal():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_bimodal(n_crack=200, n_normal=800)
    gmm = ScalarFieldGMM(scalar).fit()
    probs = gmm.anomaly_probability()
    assert probs.shape == (1000,)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    assert probs[:200].mean() > 0.5, "Crack points devem ter alta prob de anomalia"


def test_anomaly_probability_unimodal():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    probs = gmm.anomaly_probability()
    assert probs.shape == (1000,)
    assert abs(probs.mean() - 0.5) < 0.1


def test_soft_weights_unimodal_all_ones():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    weights = gmm.soft_weights()
    assert np.allclose(weights, 1.0), "Unimodal deve retornar weights=1.0"


def test_pseudo_label_confidence_unimodal_zeros():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_unimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    conf = gmm.pseudo_label_confidence()
    assert np.allclose(conf, 0.0), "Unimodal deve retornar confidence=0.0"


def test_crack_interval_bimodal():
    from utils.evaluation import ScalarFieldGMM
    scalar = make_bimodal()
    gmm = ScalarFieldGMM(scalar).fit()
    x_min, x_max = gmm.crack_interval()
    assert x_min >= 0.0
    assert x_max <= 50.0, f"x_max {x_max} deve ser <= 50"


def test_edge_cases():
    from utils.evaluation import ScalarFieldGMM
    small = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    gmm = ScalarFieldGMM(small).fit()
    assert gmm.modality in ('bimodal', 'unimodal')
    flat = np.full(100, 42.0, dtype=np.float32)
    gmm2 = ScalarFieldGMM(flat).fit()
    assert gmm2.modality == 'unimodal'


def test_save_colored_ply_shapes():
    from utils.evaluation import save_colored_ply
    from plyfile import PlyData
    rng = np.random.default_rng(0)
    N = 50
    xyz    = rng.standard_normal((N, 3)).astype(np.float32)
    rgb    = rng.uniform(0, 1, (N, 3)).astype(np.float32)
    labels = np.zeros(N, dtype=np.int64)
    labels[10:20] = 1
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        path = f.name
    try:
        save_colored_ply(xyz, rgb, labels, path)
        ply = PlyData.read(path)
        v = ply['vertex']
        assert len(v) == N
        for i in range(10, 20):
            assert v['red'][i] == 255
            assert v['green'][i] == 0
            assert v['blue'][i] == 0
        for i in range(10):
            expected_r = int(rgb[i, 0] * 255)
            assert abs(int(v['red'][i]) - expected_r) <= 1
    finally:
        os.unlink(path)


def test_save_colored_ply_no_cracks():
    from utils.evaluation import save_colored_ply
    from plyfile import PlyData
    rng = np.random.default_rng(1)
    N   = 20
    xyz    = rng.standard_normal((N, 3)).astype(np.float32)
    rgb    = np.full((N, 3), 0.5, dtype=np.float32)
    labels = np.zeros(N, dtype=np.int64)
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        path = f.name
    try:
        save_colored_ply(xyz, rgb, labels, path)
        ply = PlyData.read(path)
        assert len(ply['vertex']) == N
        assert (np.array(ply['vertex']['red']) == 255).sum() == 0
    finally:
        os.unlink(path)
