import torch
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from crack_gan import DGCNNEncoder, Generator, Discriminator, anomaly_score


K, D_IN, D_Z = 64, 16, 128


def test_encoder_output_shape():
    enc = DGCNNEncoder(in_channels=D_IN, z_dim=D_Z)
    x = torch.randn(K, D_IN)
    z = enc(x)
    assert z.shape == (D_Z,), f"Esperado ({D_Z},), obtido {z.shape}"


def test_generator_output_shape():
    gen = Generator(z_dim=D_Z, out_points=K, out_channels=D_IN)
    z = torch.randn(D_Z)
    x_rec = gen(z)
    assert x_rec.shape == (K, D_IN), f"Esperado ({K},{D_IN}), obtido {x_rec.shape}"


def test_discriminator_output_shape():
    disc = Discriminator(in_channels=D_IN, z_dim=D_Z)
    x = torch.randn(K, D_IN)
    z = torch.randn(D_Z)
    score = disc(x, z)
    assert score.shape == (1,), f"Esperado (1,), obtido {score.shape}"


def test_anomaly_score_range():
    enc  = DGCNNEncoder(in_channels=D_IN, z_dim=D_Z)
    gen  = Generator(z_dim=D_Z, out_points=K, out_channels=D_IN)
    disc = Discriminator(in_channels=D_IN, z_dim=D_Z)
    x = torch.randn(K, D_IN)
    A = anomaly_score(x, enc, gen, disc, kappa=0.1)
    assert isinstance(A, float)
    assert A >= 0.0
