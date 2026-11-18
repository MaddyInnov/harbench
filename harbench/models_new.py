"""New-model candidates for the 4-model comparison (all windowed classifiers).

- TinyHAR_GRKAN: TinyHAR encoder + group-rational KAN head (rational/fractional
  KAN candidate; clean ablation vs the MLP head).
- GatedDeltaNetHAR: pure-PyTorch Gated DeltaNet (Yang et al., ICLR 2025)
  encoder. Mathematically the paper's recurrence (scalar forget gate +
  delta-rule state update); the chunkwise parallel form is not needed at
  128-step windows. No fla/triton dependency.
- RWKVTSHAR: RWKV-TS style encoder (Hou et al., 2024): patching +
  token-shift time mixing (multi-head WKV recurrence) + channel mixing.
- MomentHAR: MOMENT-1-small (40M) time-series foundation model fine-tuned
  end-to-end for classification (lazy import; needs `pip install momentfm`).

All take x: (B, C, T) and return logits: (B, n_classes).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Group-rational KAN head
# ---------------------------------------------------------------------------

class GroupRationalKAN(nn.Module):
    """KAN-style layer with group-shared rational univariate functions.

    y_j = sum_i w_ji * R_{g(i)}(x_i) + b_j
    R_g(x) = P_g(x) / (1 + |Q_g(x)|), deg(P) = m, deg(Q) = n (no const term).
    Initialized to R_g(x) ~= x for stable training.
    """

    def __init__(self, in_f, out_f, n_groups=8, deg_p=5, deg_q=4):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        self.n_groups = n_groups
        self.deg_p, self.deg_q = deg_p, deg_q
        self.a = nn.Parameter(torch.zeros(n_groups, deg_p + 1))
        self.b = nn.Parameter(torch.zeros(n_groups, deg_q))
        with torch.no_grad():
            self.a[:, 1] = 1.0
        self.weight = nn.Parameter(
            torch.randn(out_f, in_f) * (1.0 / math.sqrt(in_f)))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.register_buffer("group_ids",
                             torch.arange(in_f) % n_groups)

    def rational(self, x):
        g = self.group_ids
        a = self.a[g]                      # (in_f, m+1)
        b = self.b[g]                      # (in_f, n)
        dev = x.device
        pw_p = torch.arange(self.deg_p + 1, device=dev, dtype=x.dtype)
        num = (x.unsqueeze(-1).pow(pw_p) * a).sum(-1)
        pw_q = torch.arange(1, self.deg_q + 1, device=dev, dtype=x.dtype)
        den = 1.0 + (x.unsqueeze(-1).pow(pw_q) * b).sum(-1).abs()
        return num / den

    def forward(self, x):
        return self.rational(x) @ self.weight.t() + self.bias


class TinyHAR_GRKAN(nn.Module):
    """TinyHAR encoder with a group-rational KAN classifier head."""

    def __init__(self, n_channels=9, n_classes=6, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1, n_groups=8):
        super().__init__()
        from .models import TinyHAR
        base = TinyHAR(n_channels=n_channels, n_classes=n_classes,
                       d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                       dropout=dropout)
        self.channel_conv = base.channel_conv
        self.fusion = base.fusion
        self.transformer = base.transformer
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            GroupRationalKAN(d_model, n_classes, n_groups=n_groups),
        )

    def forward(self, x):
        h = torch.relu(self.channel_conv(x))
        h = self.fusion(h)
        h = h.permute(0, 2, 1)
        h = self.transformer(h)
        h = h.mean(dim=1)
        return self.head(h)


# ---------------------------------------------------------------------------
# 2. Gated DeltaNet (pure PyTorch recurrence)
# ---------------------------------------------------------------------------

class GatedDeltaNetLayer(nn.Module):
    """One Gated DeltaNet layer (Yang et al., ICLR 2025), recurrent form.

    Per head and step t:
        S_t = g_t * S_{t-1} @ (I - b_t k_t k_t^T) + b_t (v_t k_t^T)
        o_t = S_t^T q_t
    with L2-normalized q, k; g_t = forget gate, b_t = delta-rule rate.
    Output gated by swish(input gate), then projected.
    """

    def __init__(self, d_model, n_heads=4, expand_k=1.0, expand_v=2.0,
                 conv_size=4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        self.kd = max(8, int(d_model * expand_k // n_heads))
        self.vd = max(8, int(d_model * expand_v // n_heads))
        H, kd, vd = n_heads, self.kd, self.vd
        self.q_proj = nn.Linear(d_model, H * kd, bias=False)
        self.k_proj = nn.Linear(d_model, H * kd, bias=False)
        self.v_proj = nn.Linear(d_model, H * vd, bias=False)
        self.beta_proj = nn.Linear(d_model, H, bias=True)
        self.gate_proj = nn.Linear(d_model, H, bias=True)
        self.out_gate = nn.Linear(d_model, H * vd, bias=True)
        self.o_proj = nn.Linear(H * vd, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        # causal depthwise short convolution on q/k/v (as in the paper)
        self.qkv_conv = nn.Conv1d(2 * H * kd + H * vd, 2 * H * kd + H * vd,
                                  kernel_size=conv_size, padding=conv_size - 1,
                                  groups=2 * H * kd + H * vd, bias=False)
        self.conv_size = conv_size
        self.register_buffer("eye_k", torch.eye(kd))

    def forward(self, x):
        # x: (B, T, d)
        B, T, _ = x.shape
        H, kd, vd = self.n_heads, self.kd, self.vd
        h = self.norm(x)
        q = self.q_proj(h).view(B, T, H, kd)
        k = self.k_proj(h).view(B, T, H, kd)
        v = self.v_proj(h).view(B, T, H, vd)
        qkv = torch.cat([q, k, v], dim=-1)          # (B,T,H,2kd+vd)
        qkv = qkv.permute(0, 2, 3, 1).reshape(B, H * (2 * kd + vd), T)
        qkv = self.qkv_conv(qkv)[:, :, :T]           # causal: keep first T
        qkv = qkv.view(B, H, 2 * kd + vd, T).permute(0, 3, 1, 2)
        q, k, v = qkv.split([kd, kd, vd], dim=-1)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        beta = torch.sigmoid(self.beta_proj(h)).view(B, T, H, 1, 1)
        gate = torch.sigmoid(self.gate_proj(h)).view(B, T, H, 1, 1)

        S = torch.zeros(B, H, vd, kd, device=x.device, dtype=x.dtype)
        eye = self.eye_k.to(x.dtype)
        outs = []
        for t in range(T):
            kt = k[:, t]            # (B,H,kd)
            vt = v[:, t]            # (B,H,vd)
            qt = q[:, t]            # (B,H,kd)
            kk = kt.unsqueeze(-1) @ kt.unsqueeze(-2)   # (B,H,kd,kd)
            vk = vt.unsqueeze(-1) @ kt.unsqueeze(-2)   # (B,H,vd,kd)
            S = gate[:, t] * (S @ (eye - beta[:, t] * kk)) + beta[:, t] * vk
            o = torch.einsum("bhcd,bhd->bhc", S, qt)  # S_t^T q_t -> (B,H,vd)
            outs.append(o)
        o = torch.stack(outs, dim=1)                   # (B,T,H,vd)
        o = o.reshape(B, T, H * vd)
        y = o * F.silu(self.out_gate(x))
        y = self.o_proj(y)
        return x + y


class GatedDeltaNetHAR(nn.Module):
    """GatedDeltaNet encoder + mean pool + linear head for HAR."""

    def __init__(self, n_channels=9, n_classes=6, d_model=128, n_heads=4,
                 n_layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv1d(n_channels, d_model, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.layers = nn.ModuleList(
            [GatedDeltaNetLayer(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))

    def forward(self, x):
        # x: (B, C, T)
        h = self.embed(x).permute(0, 2, 1)   # (B, T, d)
        for lyr in self.layers:
            h = lyr(h)
        h = self.norm(h).mean(dim=1)
        return self.head(h)


# ---------------------------------------------------------------------------
# 3. RWKV-TS encoder
# ---------------------------------------------------------------------------

class RWKVTSBlock(nn.Module):
    """One RWKV-TS block: token-shift time mixing (multi-head WKV) +
    token-shift channel mixing."""

    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model, self.n_heads = d_model, n_heads
        assert d_model % n_heads == 0
        self.hd = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        # time mixing
        self.mu_r = nn.Parameter(torch.zeros(d_model))
        self.mu_k = nn.Parameter(torch.zeros(d_model))
        self.mu_v = nn.Parameter(torch.zeros(d_model))
        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.w_decay = nn.Parameter(torch.zeros(n_heads, self.hd))
        # channel mixing
        self.mu_cr = nn.Parameter(torch.zeros(d_model))
        self.mu_ck = nn.Parameter(torch.zeros(d_model))
        self.W_cr = nn.Linear(d_model, d_model, bias=False)
        self.W_ck = nn.Linear(d_model, d_model, bias=False)
        self.W_cv = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def _shift(x, mu):
        # token shift: mu * x_t + (1 - mu) * x_{t-1}
        x_prev = F.pad(x[:, :-1], (0, 0, 1, 0))
        return mu * x + (1 - mu) * x_prev

    def _time_mix(self, x):
        B, T, _ = x.shape
        H, hd = self.n_heads, self.hd
        xr = self._shift(x, torch.sigmoid(self.mu_r))
        xk = self._shift(x, torch.sigmoid(self.mu_k))
        xv = self._shift(x, torch.sigmoid(self.mu_v))
        r = self.W_r(xr).view(B, T, H, hd)
        k = self.W_k(xk).view(B, T, H, hd).clamp(-8, 8)
        v = self.W_v(xv).view(B, T, H, hd)
        decay = torch.exp(-torch.exp(self.w_decay))      # (H, hd)
        num = torch.zeros(B, H, hd, device=x.device, dtype=x.dtype)
        den = torch.zeros(B, H, hd, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(T):
            kt = torch.exp(k[:, t])
            num = decay * num + kt * v[:, t]
            den = decay * den + kt
            wkv = num / den.clamp_min(1e-6)
            outs.append(torch.sigmoid(r[:, t]) * wkv)
        wkv = torch.stack(outs, dim=1).reshape(B, T, H * hd)
        return self.W_o(wkv)

    def _channel_mix(self, x):
        xr = self._shift(x, torch.sigmoid(self.mu_cr))
        xk = self._shift(x, torch.sigmoid(self.mu_ck))
        r = torch.sigmoid(self.W_cr(xr))
        k = self.W_ck(xk)
        v = self.W_cv(torch.relu(k) ** 2)
        return r * v

    def forward(self, x):
        x = x + self._time_mix(self.ln1(x))
        x = x + self._channel_mix(self.ln2(x))
        return x


class RWKVTSHAR(nn.Module):
    """RWKV-TS encoder (patching + RWKV blocks) + head for HAR."""

    def __init__(self, n_channels=9, n_classes=6, d_model=128, n_heads=4,
                 n_layers=2, patch_len=8, dropout=0.1):
        super().__init__()
        self.patch_len = patch_len
        self.patch_embed = nn.Linear(patch_len * n_channels, d_model)
        self.blocks = nn.ModuleList(
            [RWKVTSBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Dropout(dropout),
                                  nn.Linear(d_model, n_classes))

    def forward(self, x):
        # x: (B, C, T) -> patch over time
        B, C, T = x.shape
        P = self.patch_len
        n_patches = T // P
        x = x[:, :, :n_patches * P].permute(0, 2, 1)       # (B, T', C)
        x = x.reshape(B, n_patches, P * C)                  # (B, nP, P*C)
        h = self.patch_embed(x)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h).mean(dim=1)
        return self.head(h)


# ---------------------------------------------------------------------------
# 4. MOMENT foundation-model fine-tuning
# ---------------------------------------------------------------------------

class MomentHAR(nn.Module):
    """MOMENT-1-small fine-tuned end-to-end for HAR classification."""

    def __init__(self, n_channels=9, n_classes=6, freeze_backbone=False):
        super().__init__()
        from momentfm import MOMENTPipeline
        self.pipe = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-small",
            model_kwargs={"task_name": "classification",
                          "n_channels": n_channels,
                          "num_class": n_classes},
        )
        self.pipe.init()
        if freeze_backbone:
            for name, p in self.pipe.named_parameters():
                if not name.startswith("head"):
                    p.requires_grad = False

    def forward(self, x):
        # x: (B, C, T)
        out = self.pipe(x_enc=x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        return logits


# ---------------------------------------------------------------------------
# 5. InceptionKAN (faithful re-implementation of the user's MTech paper,
#    Sahoo & Ramanujam: "InceptionKAN: Multi-Scale Temporal
#    Kolmogorov-Arnold Networks", equations + Table 4 of the manuscript)
#
#    Block: 1x1 bottleneck (NF) -> five parallel branches (k = 1, 3, 5, 11
#    and MaxPool(3,1)+1x1), each NF filters with BN + GELU -> concat
#    (5*NF) -> SE attention (reduction r = 4) -> temporal MaxPool(2).
#    Block 1 NF = 24 (120 ch), block 2 NF = 32 (160 ch). Fusion conv
#    k = 3 -> 128 ch + ReLU, global average pooling, then a two-layer
#    RBF-KAN head 128 -> 64 -> K with dropout 0.3 between layers.
#    KAN edge: phi(x) = wb * SiLU(x) + sum_g a_g * exp(-((x - c_g) * s_g)^2)
#    with centers/widths shared per input feature and per-edge amplitudes
#    (FastKAN-style). G = 10 in layer 1 and G = 8 in layer 2 reproduce the
#    paper's Table 4 KAN parameter counts exactly (92,736 and, for K = 6,
#    4,486).
# ---------------------------------------------------------------------------

class RBFKANLayer(nn.Module):
    def __init__(self, in_f, out_f, n_gauss=10):
        super().__init__()
        self.in_f, self.out_f, self.G = in_f, out_f, n_gauss
        self.centers = nn.Parameter(
            torch.linspace(-2.0, 2.0, n_gauss).repeat(in_f, 1))
        self.inv_widths = nn.Parameter(torch.full((in_f, n_gauss), 2.0))
        self.amp = nn.Parameter(torch.randn(in_f, out_f, n_gauss)
                                * (1.0 / math.sqrt(in_f * n_gauss)))
        self.base_w = nn.Parameter(torch.randn(in_f, out_f)
                                   * (1.0 / math.sqrt(in_f)))
        self.bias = nn.Parameter(torch.zeros(out_f))

    def forward(self, x):  # (B, in_f) -> (B, out_f)
        d = (x.unsqueeze(-1) - self.centers) * self.inv_widths
        rbf = torch.exp(-d * d)                       # (B, in, G)
        out = torch.einsum("big,iog->bo", rbf, self.amp)
        return out + F.silu(x) @ self.base_w + self.bias


class _KANInceptionBlock(nn.Module):
    def __init__(self, in_ch, nf):
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, nf, 1)

        def conv_branch(k):
            return nn.Sequential(nn.Conv1d(nf, nf, k, padding=k // 2),
                                 nn.BatchNorm1d(nf), nn.GELU())

        self.b1 = conv_branch(1)
        self.b2 = conv_branch(3)
        self.b3 = conv_branch(5)
        self.b4 = conv_branch(11)
        self.b5 = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                nn.Conv1d(nf, nf, 1), nn.BatchNorm1d(nf),
                                nn.GELU())
        c = 5 * nf
        self.se_fc1 = nn.Linear(c, c // 4)
        self.se_fc2 = nn.Linear(c // 4, c)
        self.pool = nn.MaxPool1d(2, stride=2)

    def forward(self, x):
        h = self.bottleneck(x)
        y = torch.cat([self.b1(h), self.b2(h), self.b3(h), self.b4(h),
                       self.b5(h)], dim=1)
        z = y.mean(dim=2)
        s = torch.sigmoid(self.se_fc2(F.relu(self.se_fc1(z))))
        return self.pool(y * s.unsqueeze(-1))


class InceptionKAN(nn.Module):
    def __init__(self, n_channels=9, n_classes=6):
        super().__init__()
        self.block1 = _KANInceptionBlock(n_channels, 24)
        self.block2 = _KANInceptionBlock(120, 32)
        self.fusion = nn.Sequential(nn.Conv1d(160, 128, 3, padding=1),
                                    nn.ReLU())
        self.kan1 = RBFKANLayer(128, 64, n_gauss=10)
        self.drop = nn.Dropout(0.3)
        self.kan2 = RBFKANLayer(64, n_classes, n_gauss=8)

    def forward(self, x):  # (B, C, T) -> (B, n_classes)
        x = self.block2(self.block1(x))
        x = self.fusion(x).mean(dim=2)
        return self.kan2(self.drop(self.kan1(x)))


def build_new_model(name, n_channels, n_classes, seed=42):
    torch.manual_seed(seed)
    if name == "tinyhar-grkan":
        return TinyHAR_GRKAN(n_channels=n_channels, n_classes=n_classes)
    if name == "gated-deltanet":
        return GatedDeltaNetHAR(n_channels=n_channels, n_classes=n_classes)
    if name == "rwkv-ts":
        return RWKVTSHAR(n_channels=n_channels, n_classes=n_classes)
    if name == "moment":
        return MomentHAR(n_channels=n_channels, n_classes=n_classes)
    if name == "inceptionkan":
        return InceptionKAN(n_channels=n_channels, n_classes=n_classes)
    raise ValueError(f"Unknown new model: {name}")
