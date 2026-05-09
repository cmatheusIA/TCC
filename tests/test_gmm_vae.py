import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch


def _make_cloud(n_normal=900, n_crack=100, n_features=16, seed=0):
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1, (n_normal + n_crack, n_features)).astype(np.float32)
    features[:n_normal, 9] = rng.normal(200, 15, n_normal)
    features[n_normal:, 9] = rng.normal(40,  10, n_crack)
    labels = np.concatenate([np.zeros(n_normal), np.ones(n_crack)]).astype(np.int64)
    return features, labels


# ── Componentes individuais ───────────────────────────────────────────────────

def test_gumbel_softmax_shape():
    from gmm_vae import gumbel_softmax
    logits = torch.randn(50, 2)
    y = gumbel_softmax(logits, tau=0.5, hard=False)
    assert y.shape == (50, 2)
    assert torch.allclose(y.sum(-1), torch.ones(50), atol=1e-5), "softmax não soma 1"


def test_gumbel_softmax_hard_onehot():
    from gmm_vae import gumbel_softmax
    logits = torch.randn(100, 2)
    y = gumbel_softmax(logits, tau=0.1, hard=True)
    assert y.shape == (100, 2)
    # hard=True: cada linha deve ser one-hot (valores 0 ou 1)
    assert ((y == 0) | (y == 1)).all(), "hard=True deve produzir one-hot"


def test_gmvae_forward_shapes():
    from gmm_vae import GMVAE
    model = GMVAE(in_dim=16, latent_dim=4, K=2)
    x = torch.randn(50, 16)
    x_hat, mu, logvar, logits_c, z = model(x)
    assert x_hat.shape   == (50, 16), f"x_hat shape: {x_hat.shape}"
    assert mu.shape      == (50,  4), f"mu shape: {mu.shape}"
    assert logvar.shape  == (50,  4), f"logvar shape: {logvar.shape}"
    assert logits_c.shape == (50, 2), f"logits_c shape: {logits_c.shape}"
    assert z.shape       == (50,  4), f"z shape: {z.shape}"


def test_gmvae_elbo_finite():
    from gmm_vae import GMVAE
    model = GMVAE(in_dim=16, latent_dim=4, K=2)
    x = torch.randn(100, 16)
    x_hat, mu, logvar, logits_c, z = model(x)
    loss = model.elbo(x, x_hat, mu, logvar, logits_c, beta=1.0)
    assert torch.isfinite(loss), f"ELBO não é finito: {loss.item()}"
    assert loss.item() > 0, "ELBO esperado positivo"


def test_gmvae_elbo_decreases_with_training():
    from gmm_vae import GMVAE
    features, _ = _make_cloud(n_normal=200, n_crack=20)
    x = torch.tensor(features, dtype=torch.float32)
    model = GMVAE(in_dim=16, latent_dim=4, K=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for epoch in range(40):
        beta = min(1.0, epoch / 20)
        x_hat, mu, logvar, logits_c, _ = model(x, tau=max(0.5, 1.0 - epoch / 40))
        loss = model.elbo(x, x_hat, mu, logvar, logits_c, beta=beta)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    import math
    assert all(math.isfinite(l) for l in losses), f"ELBO produziu inf/NaN em algum epoch"
    assert losses[-1] < losses[0], f"ELBO não reduziu: {losses[0]:.4f} → {losses[-1]:.4f}"


# ── gmvae_crack_score ─────────────────────────────────────────────────────────

def test_gmvae_crack_score_range():
    from gmm_vae import GMVAE, gmvae_crack_score
    features, _ = _make_cloud(n_normal=200, n_crack=20)
    # normalizar
    q1 = np.percentile(features, 25, axis=0)
    q3 = np.percentile(features, 75, axis=0)
    features_norm = ((features - q1) / (q3 - q1 + 1e-8)).astype(np.float32)
    model = GMVAE(in_dim=16, latent_dim=4, K=2)
    scores, crack_idx = gmvae_crack_score(model, features_norm, features)
    assert scores.shape == (len(features),)
    assert scores.min() >= 0.0, "score abaixo de 0"
    assert scores.max() <= 1.0, "score acima de 1"
    assert crack_idx in (0, 1)


# ── evaluate_cloud ────────────────────────────────────────────────────────────

def test_evaluate_cloud_returns_expected_keys():
    from gmm_vae import evaluate_cloud
    features, labels = _make_cloud()
    cloud = {'features': features, 'labels': labels, 'filename': 'mock.ply'}
    result = evaluate_cloud(cloud, epochs=20)
    assert result is not None
    for key in ('filename', 'auroc', 'f1', 'ap', 'n_crack', 'n_normal', 'threshold'):
        assert key in result, f"chave '{key}' ausente"


def test_evaluate_cloud_none_on_few_cracks():
    from gmm_vae import evaluate_cloud
    features, _ = _make_cloud()
    labels = np.zeros(len(features), dtype=np.int64)
    labels[:3] = 1
    cloud = {'features': features, 'labels': labels, 'filename': 'mock.ply'}
    assert evaluate_cloud(cloud, epochs=5) is None
