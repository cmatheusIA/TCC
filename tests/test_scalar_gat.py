import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import torch
import numpy as np
import pytest
from scalar_gat import ScalarGATConv, ScalarGATModel, focal_loss, dice_loss


def _make_graph(N=100, k=8, dim=16):
    from torch_cluster import knn_graph
    x   = torch.randn(N, dim)
    sf  = x[:, 9]
    xyz = x[:, :3]
    batch = torch.zeros(N, dtype=torch.long)
    edge_index = knn_graph(xyz, k=k, batch=batch, loop=False)
    return x, edge_index, sf


def test_scalar_gat_conv_output_shape_concat():
    x, edge_index, sf = _make_graph(100, k=8, dim=16)
    conv = ScalarGATConv(in_channels=16, out_channels=64, heads=4, concat=True)
    out = conv(x, edge_index, sf)
    assert out.shape == (100, 256)   # 4 heads * 64


def test_scalar_gat_conv_output_shape_mean():
    x, edge_index, sf = _make_graph(100, k=8, dim=256)
    conv = ScalarGATConv(in_channels=256, out_channels=128, heads=4, concat=False)
    out = conv(x, edge_index, sf)
    assert out.shape == (100, 128)


def test_scalar_gat_model_output_shape():
    x, edge_index, sf = _make_graph(150, k=16, dim=16)
    model = ScalarGATModel(in_channels=16, k=16)
    logits = model(x, sf[:, None])   # sf passed as (N,1) slice
    assert logits.shape == (150,)


def test_scalar_gat_model_gradients():
    x, edge_index, sf = _make_graph(80, k=8, dim=16)
    x.requires_grad_(True)
    model = ScalarGATModel(in_channels=16, k=8)
    logits = model(x, sf[:, None])
    labels = torch.zeros(80)
    labels[:10] = 1
    loss = 0.7 * focal_loss(logits, labels) + 0.3 * dice_loss(logits, labels)
    loss.backward()
    assert x.grad is not None


def test_focal_loss_shape():
    logits = torch.randn(200)
    labels = torch.randint(0, 2, (200,)).float()
    loss = focal_loss(logits, labels)
    assert loss.ndim == 0 and loss.item() > 0


def test_dice_loss_all_correct():
    logits = torch.full((100,), 10.0)
    labels = torch.ones(100)
    loss = dice_loss(logits, labels)
    assert loss.item() < 0.05


def test_dice_loss_all_wrong():
    logits = torch.full((100,), -10.0)
    labels = torch.ones(100)
    loss = dice_loss(logits, labels)
    assert loss.item() > 0.9
