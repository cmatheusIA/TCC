# ============================================================================
# MÓDULOS AUXILIARES DA ARQUITETURA
# ============================================================================
# LightweightGraphConv, LocalSpatialAttention, MultiScaleAggregation,
# DensityAwareNorm, GatedResidualConnection, _selective_load
# Compartilhado por GAN v5 e Professor-Aluno v1.
# ============================================================================

from utils.config import *

import logging

log = setup_logging(LOG_PATH)


def _knn_idx(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """KNN via cKDTree: O(N log N) e O(N·k) de VRAM, vs O(N²) do torch.cdist."""
    xyz_np = xyz.detach().cpu().numpy()
    _, idx = cKDTree(xyz_np).query(xyz_np, k=k + 1, workers=1)
    return torch.from_numpy(idx[:, 1:]).long().to(xyz.device)


class LightweightGraphConv(nn.Module):
    """
    Graph Convolution leve com K-NN.
    Edge features: [centro, vizinho, diff_espacial] → MLP → max-pool.
    Inspirado em EdgeConv (Wang et al., 2019 — DGCNN).
    """

    def __init__(self, in_channels: int, out_channels: int, k: int = 16):
        super().__init__()
        self.k   = k
        edge_dim = in_channels * 2 + 3
        self.mlp = nn.Sequential(
            nn.Linear(edge_dim, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """x: (N, C)  xyz: (N, 3)"""
        N = x.size(0)
        k = min(self.k, N - 1)

        with torch.no_grad():
            kidx = _knn_idx(xyz, k)

        x_center  = x.unsqueeze(1).expand(-1, k, -1)          # (N, k, C)
        x_neigh   = x[kidx]                                    # (N, k, C)
        xyz_diff  = xyz[kidx] - xyz.unsqueeze(1).expand(-1, k, -1)  # (N, k, 3)

        edge = torch.cat([x_center, x_neigh, xyz_diff], dim=2) # (N, k, 2C+3)
        edge = edge.view(N * k, -1)
        edge = self.mlp(edge).view(N, k, -1)
        return edge.max(dim=1)[0]                               # (N, out_channels)


class LocalSpatialAttention(nn.Module):
    """
    Atenção local espacial com K-NN.
    Processa em chunks para economizar VRAM.
    """

    def __init__(self, in_channels: int, out_channels: int, num_neighbors: int = 16):
        super().__init__()
        self.k  = num_neighbors
        self.q  = nn.Linear(in_channels, out_channels)
        self.kk = nn.Linear(in_channels, out_channels)
        self.v  = nn.Linear(in_channels, out_channels)
        self.pe = nn.Linear(3, out_channels)
        self.scale = out_channels ** -0.5

    def forward(self, x: torch.Tensor, xyz: torch.Tensor,
                chunk_size: int = 512) -> torch.Tensor:
        N = x.size(0)
        k = min(self.k, N - 1)

        with torch.no_grad():
            kidx = _knn_idx(xyz, k)

        Q  = self.q(x)                        # (N, D)
        K_ = self.kk(x)                       # (N, D)
        V_ = self.v(x)                        # (N, D)

        out = torch.zeros_like(Q)
        for s in range(0, N, chunk_size):
            e   = min(s + chunk_size, N)
            idx = kidx[s:e]                           # (chunk, k)
            q_c = Q[s:e].unsqueeze(1)                 # (chunk, 1, D)
            k_c = K_[idx]                             # (chunk, k, D)
            v_c = V_[idx]                             # (chunk, k, D)

            # Positional encoding
            pe  = self.pe(xyz[idx] - xyz[s:e].unsqueeze(1))  # (chunk, k, D)
            k_c = k_c + pe

            scores = (q_c * k_c).sum(-1) * self.scale        # (chunk, k)
            attn   = F.softmax(scores, dim=-1).unsqueeze(-1)  # (chunk, k, 1)
            out[s:e] = (attn * v_c).sum(1)

        return out


class MultiScaleAggregation(nn.Module):
    """Agrega features em 3 escalas de vizinhança (k=8, 16, 32)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.scales = [8, 16, 32]
        self.mlps   = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_channels + 3, out_channels),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
            ) for _ in self.scales
        ])
        self.fusion = nn.Sequential(
            nn.Linear(out_channels * len(self.scales), out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        max_k = min(max(self.scales), N - 1)
        with torch.no_grad():
            kidx_all = _knn_idx(xyz, max_k)  # única chamada à tree para todas as escalas

        outs = []
        for k, mlp in zip(self.scales, self.mlps):
            k_act  = min(k, N - 1)
            kidx   = kidx_all[:, :k_act]
            x_nb   = x[kidx].mean(1)
            xyz_nb = xyz[kidx].mean(1) - xyz
            feat   = torch.cat([x_nb, xyz_nb], dim=1)
            outs.append(mlp(feat))

        return self.fusion(torch.cat(outs, dim=1))


class DensityAwareNorm(nn.Module):
    """BatchNorm modulado por densidade local — adapta à irregularidade da nuvem."""

    def __init__(self, channels: int, k: int = 16):
        super().__init__()
        self.k     = k
        self.bn    = nn.BatchNorm1d(channels)
        self.scale = nn.Sequential(nn.Linear(1, channels), nn.Sigmoid())
        self.bias  = nn.Sequential(nn.Linear(1, channels), nn.Tanh())

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        k = min(self.k, len(xyz) - 1)
        with torch.no_grad():
            kidx    = _knn_idx(xyz, k)
            knn_d   = torch.norm(xyz[kidx] - xyz.unsqueeze(1), dim=-1)  # (N, k)
            density = 1.0 / (knn_d.mean(1, keepdim=True) + 1e-8)
            density = (density - density.min()) / (density.max() - density.min() + 1e-8)
        return self.bn(x) * self.scale(density) + self.bias(density)


class GatedResidualConnection(nn.Module):
    """Conexão residual com gating aprendível."""

    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(channels * 2, channels), nn.Sigmoid())

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([x, residual], dim=-1))
        return g * x + (1 - g) * residual


# ============================================================================
# GENERATOR: KPFCNNInspiredAdvanced
# ============================================================================

def _selective_load(model: nn.Module, sd: dict,
                    new_layer_names: list) -> tuple:
    """
    Carregamento seletivo de pesos pré-treinados em 4 passos:

    1. Correspondência exata por nome (mais confiável)
    2. Correspondência por tokens '.' — remove prefixos 'module.' / 'model.'
    2.5 Correspondência por tokens '.' e '_' — resolve nomes compostos como
        attn_qkv → attn.qkv e attn_proj → attn.proj (PTv3 ↔ PTv3Compatible)
    3. Correspondência por shape como fallback (para layers sem nome mapeável)

    Layers em `new_layer_names` são SEMPRE puladas (treinadas do zero).

    Retorna (loaded, skipped_new, shape_mismatch) para log de aproveitamento.
    """
    import re

    def _tokens(s: str) -> set:
        s = s.replace('module.', '').replace('model.', '')
        return set(re.split(r'[._]', s))

    model_dict = model.state_dict()
    updated    = {}
    loaded     = 0
    skipped_new = 0
    shape_miss  = 0

    # Pré-processar: criar índice {shape_str: [chaves]} para fallback
    shape_index: dict = {}
    for k, v in sd.items():
        key = str(tuple(v.shape))
        shape_index.setdefault(key, []).append(k)

    # Conjunto de chaves do pretrained já usadas (evita reusar o mesmo tensor)
    used_pretrained: set = set()

    for name, param in model.named_parameters():
        # ── Pular layers novas (não têm correspondência no pretrained) ────────
        if any(new in name for new in new_layer_names):
            skipped_new += 1
            continue

        # ── Passo 1: correspondência exata por nome ───────────────────────────
        matched = False
        for k in (name,
                  'module.' + name,
                  'model.'  + name):
            if k in sd and sd[k].shape == param.shape and k not in used_pretrained:
                updated[name] = sd[k]
                used_pretrained.add(k)
                loaded += 1
                matched = True
                break

        if matched:
            continue

        # ── Passo 2: correspondência por partes do nome (split '.') ──────────
        name_parts = set(name.replace('module.', '').replace('model.', '').split('.'))
        for k, v in sd.items():
            if k in used_pretrained:
                continue
            pretrained_parts = set(k.replace('module.', '').replace('model.', '').split('.'))
            overlap = len(name_parts & pretrained_parts)
            if overlap >= 2 and v.shape == param.shape:
                updated[name] = v
                used_pretrained.add(k)
                loaded += 1
                matched = True
                break

        if matched:
            continue

        # ── Passo 2.5: correspondência fuzzy com split em '.' e '_' ──────────
        # Resolve nomes compostos: attn_qkv → {attn, qkv}, attn_proj → {attn, proj}
        # Threshold ≥ 3 para evitar falsos positivos com tokens genéricos ('weight').
        name_tokens = _tokens(name)
        for k, v in sd.items():
            if k in used_pretrained:
                continue
            pretrained_tokens = _tokens(k)
            overlap = len(name_tokens & pretrained_tokens)
            if overlap >= 3 and v.shape == param.shape:
                updated[name] = v
                used_pretrained.add(k)
                loaded += 1
                matched = True
                break

        if matched:
            continue

        # ── Passo 3: fallback por shape ───────────────────────────────────────
        shape_key = str(tuple(param.shape))
        candidates = [k for k in shape_index.get(shape_key, [])
                      if k not in used_pretrained]
        if candidates:
            k = candidates[0]
            updated[name] = sd[k]
            used_pretrained.add(k)
            loaded += 1
        else:
            shape_miss += 1

    # Aplicar apenas os pesos encontrados (strict=False implícito)
    model_dict.update(updated)
    model.load_state_dict(model_dict, strict=False)
    return loaded, skipped_new, shape_miss


