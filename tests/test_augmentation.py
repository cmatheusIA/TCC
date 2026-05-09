import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.augmentation import CrackDatabase, crack_paste, augment_cloud


def _make_cloud(n=500, has_crack=False):
    rng = np.random.default_rng(0)
    feat = rng.standard_normal((n, 16)).astype(np.float32)
    feat[:, :3] *= 0.5   # xyz in [-1, 1]
    feat[:, 3:6] = np.clip(feat[:, 3:6] * 0.1 + 0.5, 0, 1)   # rgb
    labels = np.zeros(n, dtype=np.int64)
    if has_crack:
        # cluster crack points tightly so DBSCAN finds them
        center = np.array([0.2, 0.2, 0.2], dtype=np.float32)
        feat[:50, :3] = rng.normal(0, 0.01, (50, 3)).astype(np.float32) + center
        labels[:50] = 1
    return {'features': feat, 'labels': labels, 'has_crack': has_crack,
            'filename': 'test.ply'}


def test_crack_database_builds_from_avaria():
    clouds = [_make_cloud(500, has_crack=True) for _ in range(3)]
    db = CrackDatabase(clouds)
    assert len(db) > 0, "Banco deve ter ao menos 1 patch"


def test_crack_database_ignores_normal_clouds():
    clouds = [_make_cloud(500, has_crack=False) for _ in range(5)]
    db = CrackDatabase(clouds)
    assert len(db) == 0, "Nuvens normais não devem contribuir patches"


def test_crack_paste_inserts_crack_labels():
    avaria = [_make_cloud(500, has_crack=True)]
    normal = _make_cloud(500, has_crack=False)
    db = CrackDatabase(avaria)
    if len(db) == 0:
        pytest.skip("Banco vazio — DBSCAN não encontrou cluster no mock")
    result = crack_paste(normal, db.sample(), rng=np.random.default_rng(1))
    assert result['has_crack'] is True
    assert result['labels'].sum() > 0, "Deve haver pontos crack após CrackPaste"
    assert len(result['features']) > 500, "Nuvem cresceu com patch inserido"


def test_augment_cloud_shape_preserved():
    feat = np.random.randn(300, 16).astype(np.float32)
    labels = np.zeros(300, dtype=np.int64)
    feat_aug, lbl_aug = augment_cloud(feat, labels, rng=np.random.default_rng(42))
    assert feat_aug.shape[1] == 16, "Número de features não muda"
    assert feat_aug.shape[0] <= 300, "Dropout só remove pontos"
    assert feat_aug.shape[0] == len(lbl_aug), "Features e labels alinhados"


def test_augment_cloud_scalar_field_unchanged():
    feat = np.random.randn(300, 16).astype(np.float32)
    feat[:, 9] = 42.0  # scalar_field fixo
    feat_aug, _ = augment_cloud(feat, rng=np.random.default_rng(0),
                                feat_jitter_sf=0.0)
    # scalar_field (col 9) não deve mudar por jitter de features
    assert np.allclose(feat_aug[:, 9], 42.0), "scalar_field não deve ser jitterado"
