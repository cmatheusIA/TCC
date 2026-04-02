# ============================================================================
# ARQUITETURAS DOS ENCODERS
# ============================================================================
# KPFCNNInspiredAdvanced   — encoder Generator (KPConv + skip connections)
# GANGenerator             — auto-encoder UNet-style
# SpatialPositionalEncoding3D, SpatialMultiHeadSelfAttention,
# GatedFeedForward, PointTransformerBlock, LocalFeatureAggregation
# PointTransformerInspiredAdvanced — encoder Teacher/Discriminator
# GANDiscriminator         — discriminador com spectral norm
# Compartilhado por GAN v5 e Professor-Aluno v1.
# ============================================================================

from utils.config import *
from utils.building_blocks import (
    LightweightGraphConv, LocalSpatialAttention, MultiScaleAggregation,
    DensityAwareNorm, GatedResidualConnection, _selective_load
)


log = setup_logging(LOG_PATH)

class KPFCNNInspiredAdvanced(nn.Module):
    """
    Encoder inspirado no KPConv (Thomas et al., 2019).

    Adições vs versão original do notebook:
      • feature_projection  (15D → 64D → 32D)  ← NOVA — adapta features ao domínio
      • skip connections salvas por camada       ← NOVAS — para o decoder UNet-style
      • _selective_load     (nome + shape)       ← MELHORADO vs only-shape

    Arquitetura de encoder:
      Proj: 15D → 32D
      L1  : 32D → 64   (MLP + LocalSpatialAttention + DensityAwareNorm)
      L2  : 64  → 128  (MLP + LightweightGraphConv  + DensityAwareNorm + GatedResidual)
      L3  : 128 → 256  (MLP + MultiScaleAggregation + DensityAwareNorm + GatedResidual)
      L4  : 256 → 512  (MLP + LightweightGraphConv  + BatchNorm        + GatedResidual)

    Transfer learning: kpconv_s3dis (Thomas et al., 2019) — S3DIS indoor point clouds.
    Superfícies de paredes interiores são morfologicamente análogas às paredes da
    Igreja dos Homens Pretos, justificando o transfer [Kumar et al., 2022].
    """

    # Nomes das layers exclusivamente novas (sem correspondência no pretrained)
    _NEW_LAYERS = ['feature_projection']

    def __init__(self, input_dim: int = INPUT_DIM, checkpoint_path: str = None):
        super().__init__()
        print(f"\n🎨 GENERATOR - KPFCNN Advanced (input={input_dim}D)")

        # ── Feature Projection (NOVA) ─────────────────────────────────────────
        # Mapeia as 15 features específicas do domínio (XYZ, RGB, normals, scalar,
        # curvatura, densidade…) para 32D compatível com o encoder pré-treinado.
        # Treinada do zero — aprende a importância relativa de cada feature.
        self.feature_projection = nn.Sequential(
            nn.Linear(input_dim, 64),   # 15D → 64D
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),          # 64D → 32D  (entrada do encoder)
            nn.BatchNorm1d(32),
        )
        print(f"   ✦ feature_projection: {input_dim}D → 32D (nova, treinada do zero)")

        # ── Encoder layers (com pesos pré-treinados) ──────────────────────────
        # Layer 1: 32 → 64
        self.l1_mlp  = nn.Sequential(nn.Linear(32,  64), nn.ReLU(), nn.Dropout(0.1))
        self.l1_attn = LocalSpatialAttention(64, 64, num_neighbors=16)
        self.l1_norm = DensityAwareNorm(64, k=16)

        # Layer 2: 64 → 128
        self.l2_mlp  = nn.Sequential(nn.Linear(64,  128), nn.ReLU(), nn.Dropout(0.1))
        self.l2_grph = LightweightGraphConv(128, 128, k=16)
        self.l2_norm = DensityAwareNorm(128, k=16)
        self.l2_res  = GatedResidualConnection(128)

        # Layer 3: 128 → 256
        self.l3_mlp  = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.1))
        self.l3_ms   = MultiScaleAggregation(256, 256)
        self.l3_norm = DensityAwareNorm(256, k=16)
        self.l3_res  = GatedResidualConnection(256)

        # Layer 4: 256 → 512
        self.l4_mlp  = nn.Sequential(nn.Linear(256, 512), nn.ReLU())
        self.l4_grph = LightweightGraphConv(512, 512, k=16)
        self.l4_norm = nn.BatchNorm1d(512)
        self.l4_res  = GatedResidualConnection(512)

        self._init_weights()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pretrained(checkpoint_path)
        else:
            print(f"   ⚠️  Pesos pré-treinados não encontrados: {checkpoint_path}")

        total = sum(p.numel() for p in self.parameters())
        print(f"   📊 Parâmetros: {total:,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _load_pretrained(self, path: str):
        """
        Transfer learning seletivo: nome → nome parcial → shape.
        Pula automaticamente feature_projection (layer nova).
        """
        print(f"   📦 Carregando: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

            loaded, skipped_new, shape_miss = _selective_load(
                self, sd, new_layer_names=self._NEW_LAYERS)

            total = sum(1 for _ in self.parameters())
            pct   = loaded / total * 100 if total > 0 else 0
            print(f"   ✅ Transfer: {loaded}/{total} tensores ({pct:.1f}%)")
            print(f"      Novas (skip): {skipped_new} | Shape mismatch: {shape_miss}")
            if pct < 30:
                print(f"   ⚠️  Baixo aproveitamento ({pct:.1f}%) — verifique o checkpoint")
            elif pct > 60:
                print(f"   🏆 Bom aproveitamento!")
        except Exception as e:
            print(f"   ⚠️  Erro no transfer: {e}")

    def freeze(self):
        """Congela APENAS o encoder (pesos pré-treinados). feature_projection permanece treinável."""
        for name, p in self.named_parameters():
            if 'feature_projection' not in name:
                p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        x: (N, input_dim)
        Returns: (x4, [skip1, skip2, skip3])
          x4   : (N, 512)  — bottleneck
          skips : features intermediárias para decoder UNet-style

        Se self.use_checkpoint=True, usa torch.utils.checkpoint nas camadas
        do encoder para reduzir ~40% do uso de VRAM de ativações, ao custo de
        ~30% mais compute (recomputação durante backward).
        Ativado automaticamente pelo loop de treino em caso de OOM.
        """
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        xyz = x[:, :3]
        ck  = getattr(self, 'use_checkpoint', False)

        # Projeção de features (15D → 32D) — sempre sem checkpoint (barata)
        x0 = self.feature_projection(x)

        # ── Encoder — layers custosas com skip connections ──────────────────
        def _l1(inp):
            out = self.l1_mlp(inp)
            out = out + self.l1_attn(out, xyz)
            return self.l1_norm(out, xyz)

        def _l2(inp):
            out = self.l2_mlp(inp)
            return self.l2_res(self.l2_grph(out, xyz), out)

        def _l2_norm(inp):
            return self.l2_norm(inp, xyz)

        def _l3(inp):
            out = self.l3_mlp(inp)
            return self.l3_res(self.l3_ms(out, xyz), out)

        def _l3_norm(inp):
            return self.l3_norm(inp, xyz)

        def _l4(inp):
            out = self.l4_mlp(inp)
            out = self.l4_res(self.l4_grph(out, xyz), out)
            return self.l4_norm(out)

        if ck:
            x1 = ckpt_fn(_l1, x0, use_reentrant=False)
            x2 = ckpt_fn(_l2_norm, ckpt_fn(_l2, x1, use_reentrant=False), use_reentrant=False)
            x3 = ckpt_fn(_l3_norm, ckpt_fn(_l3, x2, use_reentrant=False), use_reentrant=False)
            x4 = ckpt_fn(_l4, x3, use_reentrant=False)
        else:
            x1 = _l1(x0)
            x2 = _l2_norm(_l2(x1))
            x3 = _l3_norm(_l3(x2))
            x4 = _l4(x3)

        return x4, [x1, x2, x3]


class GANGenerator(nn.Module):
    """
    Auto-encoder UNet-style para reconstrução de superfícies normais.

    Encoder: KPFCNNInspiredAdvanced  (15D → 32D[proj] → 512[bottleneck] + skips)
    Decoder: MLP com Skip Connections UNet-style                (512 → 15D)

    Skip Connections (NOVAS):
      skip_conv1: funde skip_3 (256D) com entrada do decoder layer 1
      skip_conv2: funde skip_2 (128D) com entrada do decoder layer 2
      skip_conv3: funde skip_1  (64D) com entrada do decoder layer 3

    Skip connections resolvem o problema de vanishing gradients em redes profundas
    e preservam detalhes geométricos locais (bordas, cantos, texturas de parede)
    que seriam perdidos apenas com o bottleneck [He et al., 2016 — ResNet;
    Ronneberger et al., 2015 — UNet].

    Pontos com rachadura geram alto erro de reconstrução → anomaly score
    [R3D-AD — Zhou et al., 2024].
    """

    def __init__(self, input_dim: int = INPUT_DIM, checkpoint_path: str = None):
        super().__init__()
        self.input_dim = input_dim
        self.encoder   = KPFCNNInspiredAdvanced(input_dim, checkpoint_path)

        # ── Decoder com Skip Connections (NOVAS) ─────────────────────────────
        # Cada layer recebe concatenação: [decoder_feat | skip_i]
        # Dimensões na entrada:
        #   dec_l1: 512 + 256 = 768  → 256
        #   dec_l2: 256 + 128 = 384  → 128
        #   dec_l3: 128 +  64 = 192  →  64
        #   dec_l4:  64        →  32  → input_dim

        self.dec_l1 = nn.Sequential(
            nn.Linear(512 + 256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.1))
        self.dec_l2 = nn.Sequential(
            nn.Linear(256 + 128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1))
        self.dec_l3 = nn.Sequential(
            nn.Linear(128 +  64, 64),  nn.BatchNorm1d(64),  nn.ReLU())
        self.dec_l4 = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU())

        # Camada de reconstrução final (NOVA)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),  # 64D → 15D
        )
        print(f"   ✦ decoder com skip connections: 512→256→128→64→32→{input_dim}D")
        print(f"   ✦ reconstruction_head: 32D → 64D → {input_dim}D (nova)")

    def freeze_encoder(self):
        """Congela encoder pré-treinado, mantém projeção + decoder treináveis."""
        self.encoder.freeze()
        print("   🔒 Encoder congelado (Fase 1 — Transfer Learning)")

    def unfreeze_encoder(self):
        self.encoder.unfreeze()
        print("   🔓 Encoder descongelado (Fase 2 — Fine-tuning)")

    def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        """x: (B, N, D)  →  (B, N, D)"""
        B, N, D = x.shape
        x_flat  = x.reshape(B * N, D)

        # Encoder: bottleneck + skip connections
        bottleneck, skips = self.encoder(x_flat)
        skip3, skip2, skip1 = skips          # (N,256), (N,128), (N,64) — ordem reversa

        # Decoder UNet-style (concatena skip na entrada de cada layer)
        d1 = self.dec_l1(torch.cat([bottleneck, skip3], dim=-1))  # 512+256 → 256
        d2 = self.dec_l2(torch.cat([d1,         skip2], dim=-1))  # 256+128 → 128
        d3 = self.dec_l3(torch.cat([d2,         skip1], dim=-1))  # 128+ 64 →  64
        d4 = self.dec_l4(d3)                                       #  64     →  32

        # Reconstrução final
        out = self.reconstruction_head(d4)   # 32 → input_dim
        return out.view(B, N, D)


# ============================================================================
# DISCRIMINATOR: PointTransformerInspiredAdvanced
# ============================================================================

class SpatialPositionalEncoding3D(nn.Module):
    """Positional encoding 3D: sinusoidal + aprendível."""

    def __init__(self, d_model: int, max_freq: int = 10):
        super().__init__()
        n_freqs    = max(1, d_model // 12)
        freqs      = torch.exp(torch.linspace(0, math.log(max_freq), n_freqs))
        self.register_buffer('freqs', freqs)
        sin_dim    = 3 * 2 * n_freqs
        self.sin_proj = nn.Linear(sin_dim, d_model)
        self.pos_proj = nn.Sequential(
            nn.Linear(3, d_model // 2), nn.LayerNorm(d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """xyz: (N, 3)  →  (N, d_model)"""
        enc = []
        for i in range(3):
            c = xyz[:, i:i+1]
            enc += [torch.sin(c * self.freqs), torch.cos(c * self.freqs)]
        return self.sin_proj(torch.cat(enc, 1)) + self.pos_proj(xyz)


class SpatialMultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention com spatial bias e K-NN local."""

    def __init__(self, d_model: int, num_heads: int = 8,
                 dropout: float = 0.1, k_neighbors: int = 16):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model    = d_model
        self.num_heads  = num_heads
        self.d_head     = d_model // num_heads
        self.k          = k_neighbors
        self.scale      = self.d_head ** -0.5
        self.q   = nn.Linear(d_model, d_model)
        self.k_  = nn.Linear(d_model, d_model)
        self.v   = nn.Linear(d_model, d_model)
        self.sp  = nn.Sequential(nn.Linear(3, num_heads), nn.Tanh())
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, xyz: torch.Tensor,
                pos_enc: torch.Tensor) -> torch.Tensor:
        N  = x.size(0)
        x  = x + pos_enc
        H, Dh = self.num_heads, self.d_head
        Q  = self.q(x).view(N, H, Dh)
        K_ = self.k_(x).view(N, H, Dh)
        V_ = self.v(x).view(N, H, Dh)

        k = min(self.k, N - 1)
        with torch.no_grad():
            dist    = torch.cdist(xyz, xyz)
            _, kidx = torch.topk(dist, k + 1, largest=False, dim=1)
            kidx    = kidx[:, 1:]

        attended = torch.zeros(N, H, Dh, device=x.device)
        chunk = 512
        for s in range(0, N, chunk):
            e      = min(s + chunk, N)
            idx    = kidx[s:e]                             # (c, k)
            q_c    = Q[s:e].unsqueeze(2)                   # (c, H, 1, Dh)
            k_c    = K_[idx].permute(0, 2, 1, 3)           # (c, H, k, Dh)
            v_c    = V_[idx].permute(0, 2, 1, 3)           # (c, H, k, Dh)
            scores = torch.matmul(q_c, k_c.transpose(-2, -1)) * self.scale  # (c, H, 1, k)
            xd     = xyz[idx] - xyz[s:e].unsqueeze(1)      # (c, k, 3)
            sb     = self.sp(xd).permute(0, 2, 1).unsqueeze(2)  # (c, H, 1, k)
            scores = scores + sb
            attn   = self.drop(F.softmax(scores, dim=-1))
            attended[s:e] = torch.matmul(attn, v_c).squeeze(2)  # (c, H, Dh)

        return self.drop(self.out(attended.view(N, self.d_model)))


class GatedFeedForward(nn.Module):
    """Feed-Forward com Gated Linear Unit (GELU)."""

    def __init__(self, d_model: int, d_ff: int = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1  = nn.Linear(d_model, d_ff * 2)
        self.fc2  = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act  = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val, gate = self.fc1(x).chunk(2, dim=-1)
        return self.drop(self.fc2(self.drop(self.act(val * torch.sigmoid(gate)))))


class PointTransformerBlock(nn.Module):
    """Bloco Transformer completo com Pre-LN."""

    def __init__(self, d_model: int, num_heads: int = 8,
                 d_ff: int = None, dropout: float = 0.1, k: int = 16):
        super().__init__()
        self.attn  = SpatialMultiHeadSelfAttention(d_model, num_heads, dropout, k)
        self.ffn   = GatedFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, xyz, pos_enc):
        res = x
        x   = self.norm1(x)
        x   = res + self.drop(self.attn(x, xyz, pos_enc))
        res = x
        x   = self.norm2(x)
        x   = res + self.drop(self.ffn(x))
        return x


class LocalFeatureAggregation(nn.Module):
    """Agregação local com atenção baseada em K-NN."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 16):
        super().__init__()
        self.k   = k
        edge_dim = in_ch * 2 + 3
        self.enc = nn.Sequential(
            nn.Linear(edge_dim, out_ch), nn.LayerNorm(out_ch), nn.ReLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.attn = nn.Sequential(
            nn.Linear(out_ch, out_ch // 4), nn.ReLU(),
            nn.Linear(out_ch // 4, 1), nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        N, C = x.shape
        k    = min(self.k, N - 1)
        with torch.no_grad():
            dist    = torch.cdist(xyz, xyz)
            _, kidx = torch.topk(dist, k + 1, largest=False, dim=1)
            kidx    = kidx[:, 1:]

        xc  = x.unsqueeze(1).expand(-1, k, -1)
        xn  = x[kidx]
        xyd = xyz[kidx] - xyz.unsqueeze(1).expand(-1, k, -1)
        edge = torch.cat([xc, xn, xyd], dim=2).view(-1, xc.size(2) * 2 + 3)
        enc  = self.enc(edge).view(N, k, -1)
        w    = self.attn(enc)
        return (enc * w).sum(1)


class PointTransformerInspiredAdvanced(nn.Module):
    """
    Discriminador baseado em Point Transformer (Zhao et al., 2021).

    Adições vs versão original:
      • feature_adapter  (15D → 64D → 128D)  ← NOVA — adapta features ao domínio
        Separa a responsabilidade: adapter aprende importância das features do
        domínio; transformer blocks (pré-treinados) capturam geometria global.
      • _selective_load  (nome → nome parcial → shape)  ← MELHORADO

    Arquitetura:
      Adapter: 15D → 64D → 128D              (nova)
      Pos enc: sinusoidal + aprendível        (pré-treinado)
      LFA    : Local Feature Aggregation k=16 (pré-treinado)
      4× PointTransformerBlock                (pré-treinados)
      Proj   : 128D → 256D → 512D            (pré-treinado / novo)
    """

    # Layers exclusivamente novas (sem correspondência no pretrained)
    _NEW_LAYERS = ['feature_adapter']

    def __init__(self, input_dim: int = INPUT_DIM, d_model: int = D_MODEL,
                 num_heads: int = NUM_HEADS, num_layers: int = NUM_LAYERS,
                 dropout: float = 0.2, checkpoint_path: str = None):
        super().__init__()
        print(f"\n🎭 DISCRIMINATOR - PointTransformer Advanced")

        self.d_model = d_model

        # ── Feature Adapter (NOVA) ────────────────────────────────────────────
        # 15D → 64D → 128D (d_model)
        # Motivação: o embed original (Linear 15→128) é muito abrupto —
        # uma camada intermediária melhora o mapeamento para o espaço aprendido.
        self.feature_adapter = nn.Sequential(
            nn.Linear(input_dim, 64),     # 15D → 64D
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, d_model),       # 64D → 128D
            nn.Dropout(dropout),
        )
        print(f"   ✦ feature_adapter: {input_dim}D → 64D → {d_model}D (nova)")

        # ── Módulos pré-treinados ─────────────────────────────────────────────
        self.pos_enc = SpatialPositionalEncoding3D(d_model)
        self.lfa     = LocalFeatureAggregation(d_model, d_model, k=16)
        self.blocks  = nn.ModuleList([
            PointTransformerBlock(d_model, num_heads, d_model * 4, dropout, k=16)
            for _ in range(num_layers)
        ])
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, 512),
        )

        self._init_weights()

        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_pretrained(checkpoint_path)
        else:
            print(f"   ⚠️  Pesos pré-treinados não encontrados: {checkpoint_path}")

        print(f"   📊 Parâmetros: {sum(p.numel() for p in self.parameters()):,}")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def _load_pretrained(self, path: str):
        """
        Transfer learning seletivo: nome → nome parcial → shape.
        Pula automaticamente feature_adapter (layer nova).
        """
        print(f"   📦 Carregando: {os.path.basename(path)}")
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            sd   = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))

            loaded, skipped_new, shape_miss = _selective_load(
                self, sd, new_layer_names=self._NEW_LAYERS)

            total = sum(1 for _ in self.parameters())
            pct   = loaded / total * 100 if total > 0 else 0
            print(f"   ✅ Transfer: {loaded}/{total} tensores ({pct:.1f}%)")
            print(f"      Novas (skip): {skipped_new} | Shape mismatch: {shape_miss}")
        except Exception as e:
            print(f"   ⚠️  Erro no transfer: {e}")

    def freeze(self):
        """Congela transformer blocks (pré-treinados). feature_adapter permanece treinável."""
        for name, p in self.named_parameters():
            if 'feature_adapter' not in name:
                p.requires_grad_(False)

    def unfreeze(self):
        for p in self.parameters(): p.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, input_dim)  →  (N, 512)

        Se self.use_checkpoint=True, aplica gradient checkpointing nos
        transformer blocks (os mais pesados em memória por causa da atenção).
        """
        from torch.utils.checkpoint import checkpoint as ckpt_fn

        xyz = x[:, :3]
        x   = self.feature_adapter(x)
        pos = self.pos_enc(xyz)
        x   = x + self.lfa(x, xyz)

        ck = getattr(self, 'use_checkpoint', False)
        for blk in self.blocks:
            if ck:
                x = ckpt_fn(blk, x, xyz, pos, use_reentrant=False)
            else:
                x = blk(x, xyz, pos)

        return self.proj(x)


class GANDiscriminator(nn.Module):
    """
    Discriminador completo:
      PointTransformerInspiredAdvanced  (N, D) → (N, 512)
      AdaptiveMaxPool → global feature
      Classifier com spectral normalization (Miyato et al., 2018)
    """

    def __init__(self, input_dim: int = INPUT_DIM, checkpoint_path: str = None):
        super().__init__()
        self.features = PointTransformerInspiredAdvanced(
            input_dim=input_dim, checkpoint_path=checkpoint_path)
        self.pool     = nn.AdaptiveMaxPool1d(1)
        self.clf      = nn.Sequential(
            spectral_norm(nn.Linear(512, 256)),
            nn.LayerNorm(256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            spectral_norm(nn.Linear(256, 128)),
            nn.LayerNorm(128), nn.LeakyReLU(0.2), nn.Dropout(0.2),
            spectral_norm(nn.Linear(128,  64)), nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear( 64,   1)),
        )

    def freeze_features(self):
        """
        Congela transformer blocks (pré-treinados).
        feature_adapter permanece treinável — aprende o domínio das paredes.
        """
        self.features.freeze()
        print("   🔒 Discriminator: transformer blocks congelados (feature_adapter livre)")

    def unfreeze_features(self):
        self.features.unfreeze()
        print("   🔓 Discriminator: tudo descongelado (Fase 2 — fine-tuning)")

    def forward(self, x: torch.Tensor, masks=None) -> torch.Tensor:
        """x: (B, N, D)  →  (B, 1)"""
        B, N, D   = x.shape
        x_flat    = x.reshape(B * N, D)
        feats     = self.features(x_flat).view(B, N, -1)  # (B, N, 512)
        pooled    = self.pool(feats.permute(0, 2, 1)).squeeze(-1)  # (B, 512)
        return self.clf(pooled)


# ============================================================================
# EARLY STOPPING
# ============================================================================

