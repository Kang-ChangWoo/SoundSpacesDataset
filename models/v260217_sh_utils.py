"""
Spherical Harmonics utilities for ERP depth estimation (v260217).

Extends v260120 with bidirectional cross-attention fusion:
  Pass 1 (Feature→SH): feature pixels query SH tokens
  Pass 2 (SH→Feature): SH tokens query feature pixels

Adapted from HUSH (https://github.com/vision3d-lab/HUSH).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import sph_harm


def build_real_sh_basis(degree, height, width):
    """
    Build real spherical harmonics basis for an ERP grid.

    Uses HUSH convention: only non-negative m values (m >= 0), giving
    degree*(degree+1)/2 basis functions.

    Args:
        degree: Maximum SH degree (l ranges from 0 to degree-1)
        height: ERP image height (e.g., 256)
        width: ERP image width (e.g., 512)

    Returns:
        basis: FloatTensor of shape (N_coeffs, height, width)
               where N_coeffs = degree*(degree+1)//2
    """
    theta = np.linspace(0, np.pi, height, endpoint=False) + np.pi / (2 * height)
    phi = np.linspace(0, 2 * np.pi, width, endpoint=False) + np.pi / width

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')

    basis_list = []
    for l in range(degree):
        for m in range(l + 1):
            Y_complex = sph_harm(m, l, phi_grid, theta_grid)
            if m == 0:
                Y_real = Y_complex.real
            else:
                Y_real = np.sqrt(2) * Y_complex.real
            basis_list.append(Y_real.astype(np.float32))

    basis = np.stack(basis_list, axis=0)
    return torch.from_numpy(basis)


class SHCoeffExtractor(nn.Module):
    """
    Extract SH coefficients from bottleneck features.

    3-layer MLP with BatchNorm for better gradient flow.
    """

    def __init__(self, in_channels, n_coeffs, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(True),
            nn.Linear(hidden_dim, n_coeffs),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) feature map
        Returns:
            coeffs: (B, N_coeffs)
        """
        return self.net(x)


class SHBidirectionalFusionModule(nn.Module):
    """
    Bidirectional cross-attention fusion between decoder features and SH tokens.

    Pass 1 (Feature→SH): Feature pixels query SH tokens.
        Q = features (B, HW, C),  K,V = sh_tokens (B, N_sh, C)
        Attention shape: (B, heads, HW, N_sh) — efficient since N_sh << HW.
        Features learn geometric priors from SH basis.

    Pass 2 (SH→Feature): SH tokens query (enriched) feature pixels.
        Q = sh_tokens (B, N_sh, C),  K,V = feat_seq (B, HW, C)
        Attention shape: (B, heads, N_sh, HW)
        SH tokens absorb spatial context from features.
        Enriched SH tokens are projected back to feature space via a final
        cross-attention readout.

    Both passes use pre-norm and residual connections.
    """

    def __init__(self, feature_channels, n_sh, num_heads=4):
        super().__init__()
        self.feature_channels = feature_channels
        self.n_sh = n_sh
        self.num_heads = num_heads
        assert feature_channels % num_heads == 0, \
            f"feature_channels ({feature_channels}) must be divisible by num_heads ({num_heads})"
        self.head_dim = feature_channels // num_heads
        self.scale = self.head_dim ** -0.5

        # --- SH token creation ---
        self.sh_pool = nn.AdaptiveAvgPool2d(4)               # (B, N_sh, 4, 4)
        self.sh_token_proj = nn.Sequential(
            nn.Linear(16, feature_channels),                  # 4*4=16 -> C
            nn.ReLU(True),
        )

        # --- Pass 1: Feature→SH (features query SH tokens) ---
        self.norm_q1 = nn.LayerNorm(feature_channels)
        self.norm_kv1 = nn.LayerNorm(feature_channels)
        self.to_q1 = nn.Linear(feature_channels, feature_channels)
        self.to_kv1 = nn.Linear(feature_channels, feature_channels * 2)
        self.to_out1 = nn.Linear(feature_channels, feature_channels)

        # --- Pass 2: SH→Feature (SH tokens query features) ---
        self.norm_q2 = nn.LayerNorm(feature_channels)
        self.norm_kv2 = nn.LayerNorm(feature_channels)
        self.to_q2 = nn.Linear(feature_channels, feature_channels)
        self.to_kv2 = nn.Linear(feature_channels, feature_channels * 2)
        self.to_out2 = nn.Linear(feature_channels, feature_channels)

        # --- Final readout: enriched SH → features ---
        self.norm_q3 = nn.LayerNorm(feature_channels)
        self.norm_kv3 = nn.LayerNorm(feature_channels)
        self.to_q3 = nn.Linear(feature_channels, feature_channels)
        self.to_kv3 = nn.Linear(feature_channels, feature_channels * 2)
        self.to_out3 = nn.Linear(feature_channels, feature_channels)

    def _multi_head_attn(self, q, k, v, n_q, n_kv):
        """Shared multi-head scaled dot-product attention.

        Args:
            q: (B, n_q, C)
            k: (B, n_kv, C)
            v: (B, n_kv, C)
            n_q: query sequence length
            n_kv: key/value sequence length

        Returns:
            out: (B, n_q, C)
        """
        B = q.shape[0]
        C = self.feature_channels
        heads = self.num_heads
        d = self.head_dim

        q = q.view(B, n_q, heads, d).transpose(1, 2)    # (B, heads, n_q, d)
        k = k.view(B, n_kv, heads, d).transpose(1, 2)   # (B, heads, n_kv, d)
        v = v.view(B, n_kv, heads, d).transpose(1, 2)   # (B, heads, n_kv, d)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)                   # (B, heads, n_q, n_kv)

        out = torch.matmul(attn, v)                      # (B, heads, n_q, d)
        out = out.transpose(1, 2).reshape(B, n_q, C)     # (B, n_q, C)
        return out

    def forward(self, features, sh_map):
        """
        Args:
            features: (B, C, H, W) decoder features
            sh_map:   (B, N_sh, H_full, W_full) multi-channel SH map
        Returns:
            fused: (B, C, H, W) features enriched by bidirectional SH attention
        """
        B, C, H, W = features.shape
        N = self.n_sh
        HW = H * W

        # --- Build SH tokens: (B, N_sh, C) ---
        sh_resized = F.interpolate(
            sh_map, size=(H, W), mode='bilinear', align_corners=True
        )                                                    # (B, N_sh, H, W)
        sh_pooled = self.sh_pool(sh_resized)                 # (B, N_sh, 4, 4)
        sh_tokens = sh_pooled.flatten(2)                     # (B, N_sh, 16)
        sh_tokens = self.sh_token_proj(sh_tokens)            # (B, N_sh, C)

        # --- Flatten features to sequence ---
        feat_seq = features.permute(0, 2, 3, 1).reshape(B, HW, C)  # (B, HW, C)

        # ============================================================
        # Pass 1: Feature→SH  (features learn from SH geometry)
        # Q = feat_seq (B, HW, C),  K,V = sh_tokens (B, N_sh, C)
        # ============================================================
        q1 = self.to_q1(self.norm_q1(feat_seq))              # (B, HW, C)
        kv1 = self.norm_kv1(sh_tokens)
        k1, v1 = self.to_kv1(kv1).chunk(2, dim=-1)          # each (B, N_sh, C)
        out1 = self._multi_head_attn(q1, k1, v1, HW, N)     # (B, HW, C)
        feat_seq = feat_seq + self.to_out1(out1)             # residual

        # ============================================================
        # Pass 2: SH→Feature  (SH tokens learn from enriched features)
        # Q = sh_tokens (B, N_sh, C),  K,V = feat_seq (B, HW, C)
        # ============================================================
        q2 = self.to_q2(self.norm_q2(sh_tokens))             # (B, N_sh, C)
        kv2 = self.norm_kv2(feat_seq)
        k2, v2 = self.to_kv2(kv2).chunk(2, dim=-1)          # each (B, HW, C)
        out2 = self._multi_head_attn(q2, k2, v2, N, HW)     # (B, N_sh, C)
        sh_enriched = sh_tokens + self.to_out2(out2)         # residual

        # ============================================================
        # Final readout: features attend to enriched SH tokens
        # Q = feat_seq (B, HW, C),  K,V = sh_enriched (B, N_sh, C)
        # ============================================================
        q3 = self.to_q3(self.norm_q3(feat_seq))              # (B, HW, C)
        kv3 = self.norm_kv3(sh_enriched)
        k3, v3 = self.to_kv3(kv3).chunk(2, dim=-1)          # each (B, N_sh, C)
        out3 = self._multi_head_attn(q3, k3, v3, HW, N)     # (B, HW, C)
        feat_seq = feat_seq + self.to_out3(out3)             # residual

        # --- Reshape back to spatial ---
        out = feat_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return out
