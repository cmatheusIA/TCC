# tests/test_scalar_field_unsup.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Mock Teacher ──────────────────────────────────────────────────────────────

class MockTeacher(nn.Module):
    """Teacher mínimo para testes sem GPU/checkpoint."""
    def forward(self, x):
        return torch.randn(x.size(0), 512)

    @property
    def feature_adapter(self):
        return nn.Identity()

    @property
    def lfa(self):
        return nn.Identity()

    @property
    def blocks(self):
        return []


# ── Testes EdgeConvBlock ──────────────────────────────────────────────────────

def test_edge_conv_block_output_shape():
    from scalar_field_unsup import EdgeConvBlock
    block = EdgeConvBlock(in_channels=16, out_channels=32, k=5)
    N = 50
    h   = torch.randn(N, 16)
    knn = torch.randint(0, N, (N, 5))
    out = block(h, knn)
    assert out.shape == (N, 32), f"Shape esperado ({N}, 32), got {out.shape}"

def test_edge_conv_block_different_sizes():
    from scalar_field_unsup import EdgeConvBlock
    block = EdgeConvBlock(in_channels=512, out_channels=256, k=10)
    N = 100
    h   = torch.randn(N, 512)
    knn = torch.randint(0, N, (N, 10))
    out = block(h, knn)
    assert out.shape == (N, 256)

def test_edge_conv_block_gradient_flows():
    from scalar_field_unsup import EdgeConvBlock
    block = EdgeConvBlock(in_channels=8, out_channels=8, k=3)
    N = 20
    h   = torch.randn(N, 8, requires_grad=True)
    knn = torch.randint(0, N, (N, 3))
    out = block(h, knn)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None


# ── Testes build_knn_idx ──────────────────────────────────────────────────────

def test_build_knn_idx_shape():
    from scalar_field_unsup import build_knn_idx
    feats = np.random.randn(50, 3).astype(np.float32)
    idx = build_knn_idx(feats, k=5)
    assert idx.shape == (50, 5), f"Shape esperado (50,5), got {idx.shape}"
    assert idx.dtype == torch.int64

def test_build_knn_idx_no_self():
    from scalar_field_unsup import build_knn_idx
    feats = np.random.randn(30, 3).astype(np.float32)
    idx = build_knn_idx(feats, k=4)
    # Nenhum ponto deve ser vizinho de si mesmo
    for i in range(30):
        assert i not in idx[i].tolist(), f"Ponto {i} é vizinho de si mesmo"


# ── Testes ScalarFieldDGCNN ───────────────────────────────────────────────────

def test_dgcnn_forward_shape():
    from scalar_field_unsup import ScalarFieldDGCNN
    model = ScalarFieldDGCNN(input_dim=15, k=5)
    model.teacher = MockTeacher()

    N = 40
    x = torch.randn(N, 15)
    score, recon = model(x)
    assert score.shape == (N,), f"Score shape: {score.shape}"
    assert recon.shape == (N, 512), f"Recon shape: {recon.shape}"
    assert score.min() >= 0.0 and score.max() <= 1.0, "Score deve estar em [0,1]"

def test_dgcnn_no_grad_on_teacher():
    from scalar_field_unsup import ScalarFieldDGCNN
    model = ScalarFieldDGCNN(input_dim=15, k=5)
    model.teacher = MockTeacher()

    for p in model.teacher.parameters():
        assert not p.requires_grad, "Teacher deve ter requires_grad=False"
