# ============================================================================
# CONFIGURAÇÕES GLOBAIS DO PROJETO
# ============================================================================
# Importado por gan_rachaduras_v5.py e teacher_student_v1.py.
# Alterar BASE_PATH para o caminho correto no seu ambiente.
# ============================================================================

import os
import gc
import math
import time
import json
import random
import logging
import warnings
import psutil
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import open3d as o3d
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from torch.nn.utils import spectral_norm
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, jaccard_score, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve
)
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

import logging

# 1. Configura o "setup" do logging (nível mínimo, formato da mensagem, etc)
logging.basicConfig(
    level=logging.INFO, # Mostra mensagens nível INFO ou acima (WARNING, ERROR)
    format='%(asctime)s - %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)

# 2. Cria o seu log personalizado e chama ele de 'log'
log = logging.getLogger("tcc")


warnings.filterwarnings('ignore')

# ── Caminhos ──────────────────────────────────────────────────────────────────
# Alterar BASE_PATH para o caminho da sua máquina.
# WSL local  : '/home/<usuario>/tcc'
# GCP / Colab: '/content/drive/MyDrive/MEU_TCC/TCC 2'
BASE_PATH       = f'{os.getcwd()}'
DATA_TRAIN      = f'{BASE_PATH}/data/train'
DATA_TEST       = f'{BASE_PATH}/data/test'
PRETRAINED      = f'{BASE_PATH}/pretrained_models'
VIS_PATH        = f'{BASE_PATH}/visualizations_ts'
LOG_PATH        = f'{BASE_PATH}/logs_ts'

KPCONV_WEIGHTS        = f'{PRETRAINED}/kpconv_s3dis_202010091238.pth'
PTRANSF_WEIGHTS       = f'{PRETRAINED}/ptv3_scannet200.pth'         # PTv3 ScanNet200 (tentativa A ou B)
PTRANSF_WEIGHTS_S3DIS = f'{PRETRAINED}/pointtransformer_s3dis_202109241350utc.pth'  # fallback C

# ── Hiperparâmetros compartilhados ────────────────────────────────────────────
INPUT_DIM    = 15      # xyz(3)+rgb(3)+normals(3)+scalar(1)+label(1)+curv(1)+dens(1)+var(1)+sv(1)
D_MODEL      = 128     # dimensão interna do PointTransformer
NUM_HEADS    = 8       # cabeças de atenção
NUM_LAYERS   = 4       # blocos transformer
VOXEL_SIZE   = 0.01   # tamanho do voxel para subsampling (metros)
ANOMALY_PCTL = 85      # percentil de fallback para threshold GMM
# ── ENVS ────────────────────────────────────────────

# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(log_path: str) -> logging.Logger:
    """Configura logger com saída simultânea para arquivo e terminal."""
    os.makedirs(log_path, exist_ok=True)
    ts  = datetime.now().strftime('%Y%m%d_%H%M%S')
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%H:%M:%S')
    log = logging.getLogger('tcc')
    log.setLevel(logging.INFO)
    if not log.handlers:
        fh = logging.FileHandler(f'{log_path}/run_{ts}.log', encoding='utf-8')
        fh.setFormatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        log.addHandler(fh)
        log.addHandler(sh)
    return log


def detect_environment() -> dict:
    """
    Detecta automaticamente o ambiente de execução e retorna
    configurações otimizadas para num_workers e pin_memory.

    Regras:
      - Colab/GCP  : num_workers=2, pin_memory=True
      - WSL local  : num_workers=4, pin_memory=True
      - Sem GPU    : num_workers=0, pin_memory=False
        (pin_memory só tem efeito com CUDA; sem GPU é overhead puro)
      - WSL quirk  : WSL2 tem um bug antigo onde num_workers > 0
        pode travar com DataLoader em alguns kernels.
        Se travar, forçar num_workers=0 manualmente no config.
    """
    has_cuda = torch.cuda.is_available()

    if not has_cuda:
        return {'num_workers': 0, 'pin_memory': False, 'env': 'cpu'}

    # Detectar Colab
    try:
        import google.colab  # noqa
        return {'num_workers': 2, 'pin_memory': True, 'env': 'colab'}
    except ImportError:
        pass

    # Detectar WSL2
    try:
        with open('/proc/version', 'r') as f:
            if 'microsoft' in f.read().lower():
                return {'num_workers': 4, 'pin_memory': True, 'env': 'wsl'}
    except FileNotFoundError:
        pass

    # Linux nativo / GCP
    return {'num_workers': 4, 'pin_memory': True, 'env': 'linux'}


ENV_CONFIG = detect_environment()
