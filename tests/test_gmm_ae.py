import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch


def _make_cloud(n_normal=900, n_crack=100, n_features=16, seed=0):
    """Nuvem sintética bimodal: col 9 = scalar_field discriminativo."""
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1, (n_normal + n_crack, n_features)).astype(np.float32)
    features[:n_normal, 9] = rng.normal(200, 15, n_normal)
    features[n_normal:, 9] = rng.normal(40,  10, n_crack)
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_crack)]).astype(np.int64)
    return features, labels


# ── CrackAutoencoder ────────────────────────────────────────────────────────

def test_autoencoder_forward_shape():
    from gmm_ae import CrackAutoencoder
    model = CrackAutoencoder(in_dim=16, latent_dim=4)
    x = torch.randn(50, 16)
    x_hat, z = model(x)
    assert x_hat.shape == (50, 16), f"esperado (50,16), obtido {x_hat.shape}"
    assert z.shape     == (50,  4), f"esperado (50, 4), obtido {z.shape}"


def test_autoencoder_trains_mse_decreases():
    from gmm_ae import CrackAutoencoder
    import torch.nn.functional as F
    features, _ = _make_cloud()
    x = torch.tensor(features, dtype=torch.float32)
    model = CrackAutoencoder(in_dim=16, latent_dim=4)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(30):
        x_hat, _ = model(x)
        loss = F.mse_loss(x_hat, x)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "MSE não reduziu após 30 epochs"


# ── fit_ae_gmm ───────────────────────────────────────────────────────────────

def test_fit_ae_gmm_score_range():
    from gmm_ae import fit_ae_gmm
    features, _ = _make_cloud()
    scores = fit_ae_gmm(features, latent_dim=4, epochs=30)
    assert scores.shape == (len(features),), "shape incorreto"
    assert scores.min() >= 0.0,  "score abaixo de 0"
    assert scores.max() <= 1.0,  "score acima de 1"
    assert scores.dtype == np.float32


def test_fit_ae_gmm_auroc_bimodal():
    from gmm_ae import fit_ae_gmm
    from sklearn.metrics import roc_auc_score
    features, labels = _make_cloud()
    torch.manual_seed(42)
    scores = fit_ae_gmm(features, latent_dim=4, epochs=80)
    auroc = roc_auc_score(labels, scores)
    assert auroc > 0.70, f"AUROC esperado >0.70 em dados bimodais sintéticos, obtido {auroc:.3f}"


# ── evaluate_cloud ────────────────────────────────────────────────────────────

def test_evaluate_cloud_returns_expected_keys():
    from gmm_ae import evaluate_cloud
    features, labels = _make_cloud()
    cloud = {'features': features, 'labels': labels, 'filename': 'mock.ply'}
    result = evaluate_cloud(cloud, ae_epochs=20)
    assert result is not None
    for key in ('filename', 'auroc', 'f1', 'ap', 'n_crack', 'n_normal', 'threshold'):
        assert key in result, f"chave '{key}' ausente no resultado"


def test_evaluate_cloud_none_on_few_cracks():
    from gmm_ae import evaluate_cloud
    features, _ = _make_cloud()
    labels = np.zeros(len(features), dtype=np.int64)
    labels[:3] = 1  # só 3 cracks — abaixo do mínimo
    cloud = {'features': features, 'labels': labels, 'filename': 'mock.ply'}
    assert evaluate_cloud(cloud, ae_epochs=5) is None


def test_evaluate_cloud_metrics_in_range():
    from gmm_ae import evaluate_cloud
    features, labels = _make_cloud()
    cloud = {'features': features, 'labels': labels, 'filename': 'mock.ply'}
    r = evaluate_cloud(cloud, ae_epochs=30)
    assert 0.0 <= r['auroc'] <= 1.0
    assert 0.0 <= r['f1']    <= 1.0
    assert 0.0 <= r['ap']    <= 1.0


def test_fit_ae_gmm_crack_idx_resolves_correctly():
    """crack_idx deve apontar para o cluster com menor média sf (crack = sf baixo)."""
    from gmm_ae import fit_ae_gmm
    features, labels = _make_cloud(seed=7)
    torch.manual_seed(42)
    scores = fit_ae_gmm(features, latent_dim=4, epochs=80)
    crack_scores  = scores[labels == 1].mean()
    normal_scores = scores[labels == 0].mean()
    assert crack_scores > normal_scores, (
        f"crack_idx errado: mean score crack={crack_scores:.3f} < normal={normal_scores:.3f}"
    )
