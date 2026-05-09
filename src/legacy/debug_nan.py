"""Script de diagnóstico: localiza exatamente onde o NaN aparece no ScalarFusionPTv3."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from torch.amp import autocast

from utils.config import INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS, NUM_CLASSES
from utils.data import load_ply_file
from utils.architectures import build_teacher
from scalar_gat import compute_class_weights, multiclass_loss
from ptv3_scalar_fusion import ScalarBranch, FusionHead, ScalarFusionPTv3, SF_DIM, GEO_DIM, K_NEIGHBORS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PLY = '/home/cmatheus/projects/TCC/data_abnt/train/avaria_10.ply'

def nan_info(name, t):
    if t is None:
        print(f"  {name}: None")
        return
    has_nan = torch.isnan(t.float()).any().item()
    has_inf = torch.isinf(t.float()).any().item()
    f = t.float()
    print(f"  {name}: shape={tuple(t.shape)}  dtype={t.dtype}  "
          f"NaN={has_nan}  Inf={has_inf}  "
          f"min={f.min():.4f}  max={f.max():.4f}  mean={f.mean():.4f}")

print("=" * 60)
print("DIAGNÓSTICO NaN — ScalarFusionPTv3")
print("=" * 60)

# ── Dados ─────────────────────────────────────────────────────
d = load_ply_file(PLY)
x = torch.tensor(d['features'], dtype=torch.float32).to(DEVICE)
y = torch.tensor(d['labels'],   dtype=torch.long).to(DEVICE)
print(f"\n[1] Input")
nan_info("x", x)
print(f"  y unique: {torch.unique(y).tolist()}")

# ── Modelo ────────────────────────────────────────────────────
teacher     = build_teacher(INPUT_DIM, PTRANSF_WEIGHTS, PTRANSF_WEIGHTS_S3DIS).to(DEVICE)
scalar_br   = ScalarBranch(k=K_NEIGHBORS, out_dim=SF_DIM).to(DEVICE)
fusion_head = FusionHead(in_geo=GEO_DIM, in_sf=SF_DIM, num_classes=NUM_CLASSES).to(DEVICE)
model       = ScalarFusionPTv3(teacher, scalar_br, fusion_head)

# ── train() vs eval() ─────────────────────────────────────────
for mode_name, mode_fn in [('eval', model.eval), ('train', model.train)]:
    mode_fn()
    print(f"\n[2] Componentes em modo {mode_name.upper()}")

    with torch.no_grad() if mode_name == 'eval' else torch.enable_grad():
        try:
            with autocast('cuda'):
                geo = teacher(x)
            nan_info("teacher(x)", geo)
        except Exception as e:
            print(f"  teacher(x): ERRO → {e}")

        try:
            with autocast('cuda'):
                sf = scalar_br(x)
            nan_info("scalar_br(x)", sf)
        except Exception as e:
            print(f"  scalar_br(x): ERRO → {e}")

        try:
            with autocast('cuda'):
                logits = model(x)
            nan_info("model(x) logits", logits)
        except Exception as e:
            print(f"  model(x): ERRO → {e}")
            logits = None

    if logits is not None:
        # weights iguais a 1 para isolar da loss
        weights_ones = torch.ones(NUM_CLASSES, device=DEVICE)
        weights_real = compute_class_weights([d], DEVICE)
        print(f"  class_weights: {[round(float(w), 3) for w in weights_real]}")

        for wname, w in [('weights=1', weights_ones), ('weights_real', weights_real)]:
            try:
                with autocast('cuda'):
                    loss = multiclass_loss(logits, y, w)
                print(f"  loss ({wname}): {loss.item():.6f}  NaN={torch.isnan(loss).item()}")
            except Exception as e:
                print(f"  loss ({wname}): ERRO → {e}")

print("\nDiagnóstico concluído.")
