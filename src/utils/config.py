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
from plyfile import PlyData, PlyElement
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
BASE_PATH       = str(Path(__file__).resolve().parents[2])  # src/utils/config.py → TCC/
DATA_TRAIN      = f'{BASE_PATH}/data/train'
DATA_TEST       = f'{BASE_PATH}/data/test'
DATA_TRAIN_ABNT = f'{BASE_PATH}/data_abnt/train'
DATA_TEST_ABNT  = f'{BASE_PATH}/data_abnt/test'
PRETRAINED      = f'{BASE_PATH}/pretrained_models'
VIS_PATH        = f'{BASE_PATH}/visualizations_ts'
LOG_PATH        = f'{BASE_PATH}/logs_ts'

KPCONV_WEIGHTS        = f'{PRETRAINED}/kpconv_s3dis_202010091238.pth'
PTRANSF_WEIGHTS       = f'{PRETRAINED}/ptv3_scannet200.pth'
PTRANSF_WEIGHTS_S3DIS = f'{PRETRAINED}/pointtransformer_s3dis_202109241350utc.pth'

# ── Hiperparâmetros compartilhados ────────────────────────────────────────────
INPUT_DIM    = 18      # xyz(3)+rgb(3)+normals(3)+scalar(1)+curv(1)+dens(1)+var(1)+sv(1)+lum(1)+sat(1)+z_score(1)+gradient_mag(1)
INPUT_DIM_V2 = INPUT_DIM  # alias de compatibilidade
D_MODEL      = 128     # dimensão interna do PointTransformer
NUM_HEADS    = 8       # cabeças de atenção
NUM_LAYERS   = 4       # blocos transformer

# ── Classificação multiclasse ABNT NBR 6118 ──────────────────────────────────
# Labels armazenados como 0-indexed (0–4) após remapeamento de 1–5 do label_converter.
NUM_CLASSES  = 5
CLASS_NAMES  = ['Microfissura', 'Fissura', 'Trinca', 'Rachadura', 'Normal']
# 0=Microfissura 1=Fissura 2=Trinca 3=Rachadura 4=Normal
NORMAL_CLASS = 4
VOXEL_SIZE   = 0.01   # tamanho do voxel para subsampling (metros)
ANOMALY_PCTL = 85      # percentil de fallback para threshold GMM

# ── Detecção binária (crack=1 / normal=0) ────────────────────────────────────
# Fração média de pontos crack nas nuvens avaria (medido nos 35 PLYs)
CRACK_RATIO      = 0.0662          # ~6.62 % dos pontos são crack
POS_WEIGHT_DEFAULT = (1 - CRACK_RATIO) / CRACK_RATIO   # ≈ 14.1  (para BCEWithLogitsLoss)
DATA_TRAIN_BIN   = DATA_TRAIN      # alias explícito — sempre binário

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
