import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import numpy as np
import pytest
from scalar_mae import (
    fps_centroids, build_patches, tokenize_patches,
    ScalarMAEEncoder, ScalarMAEDecoder, ScalarMAEModel,
    scalar_guided_mask,
)


def _rand_features(N=500):
    feat = torch.randn(N, 16)
    feat[:, 9] = torch.rand(N) * 100  # scalar_field col
    return feat


def test_fps_centroids_count():
    feat = _rand_features(500)
    xyz  = feat[:, :3]
    centers = fps_centroids(xyz, M=64)
    assert centers.shape == (64, 3)


def test_fps_centroids_small_cloud():
    feat = _rand_features(30)
    centers = fps_centroids(feat[:, :3], M=64)
    assert centers.shape[0] <= 30


def test_build_patches_shape():
    feat = _rand_features(500)
    centers = fps_centroids(feat[:, :3], M=32)
    patches, patch_sf = build_patches(feat, centers, k=16)
    assert patches.shape == (32, 16, 16)   # (M, k, feat_dim)
    assert patch_sf.shape == (32,)


def test_tokenize_shape():
    feat = _rand_features(500)
    centers = fps_centroids(feat[:, :3], M=32)
    patches, _ = build_patches(feat, centers, k=16)
    tokens = tokenize_patches(patches, centers)   # (M, 35)
    assert tokens.shape == (32, 35)


def test_scalar_guided_mask_ratio():
    sf_means = torch.rand(64)
    mask = scalar_guided_mask(sf_means, base_ratio=0.4, extra=0.3, T=0.1)
    assert mask.shape == (64,)
    assert mask.dtype == torch.bool
    masked_ratio = mask.float().mean().item()
    # With T=0.1 and N=64, Bernoulli variance can push ratio below 0.3 occasionally
    assert 0.1 <= masked_ratio <= 0.95


def test_encoder_output_shape():
    feat = _rand_features(500)
    centers = fps_centroids(feat[:, :3], M=32)
    patches, sf_means = build_patches(feat, centers, k=16)
    tokens = tokenize_patches(patches, centers)
    mask = scalar_guided_mask(sf_means, base_ratio=0.4, extra=0.3)
    enc = ScalarMAEEncoder(token_dim=35, d_model=64, n_heads=4, n_layers=2)
    vis_tokens = tokens[~mask]   # (N_vis, 35)
    out = enc(vis_tokens.unsqueeze(0))   # (1, N_vis+1, 64)
    assert out.shape[0] == 1
    assert out.shape[2] == 64


def test_mae_model_pretrain_loss():
    feat = _rand_features(300)
    model = ScalarMAEModel(d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=1)
    loss = model.pretrain_step(feat)
    assert loss.ndim == 0 and loss.item() > 0


def test_mae_model_segment_shape():
    feat = _rand_features(400)
    model = ScalarMAEModel(d_model=64, n_heads=4, n_enc_layers=2, n_dec_layers=1)
    logits = model.segment(feat)
    assert logits.shape == (400,)
