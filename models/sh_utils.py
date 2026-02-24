"""
Spherical Harmonics utilities for ERP depth estimation.

Adapted from HUSH (https://github.com/vision3d-lab/HUSH).
SH provides a natural basis for functions on the sphere, making it
ideal for ERP (equirectangular projection) images which map the full
sphere to a 2D grid.

Key idea:
- ERP pixel (i, j) maps to spherical coordinates (theta, phi)
- Real SH basis functions Y_l^m(theta, phi) form an orthonormal basis
- Depth on the sphere can be decomposed into SH coefficients
- Low-order SH captures global room structure (walls, floor, ceiling)
- Higher-order SH captures finer geometric detail
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
    # ERP spherical coordinate grid
    # theta (colatitude): 0 to pi, top to bottom
    theta = np.linspace(0, np.pi, height, endpoint=False) + np.pi / (2 * height)
    # phi (azimuth): 0 to 2*pi, left to right
    phi = np.linspace(0, 2 * np.pi, width, endpoint=False) + np.pi / width

    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')  # (H, W)

    basis_list = []
    for l in range(degree):
        for m in range(l + 1):
            # scipy.special.sph_harm(m, l, phi, theta) -> complex SH
            Y_complex = sph_harm(m, l, phi_grid, theta_grid)

            if m == 0:
                Y_real = Y_complex.real
            else:
                # Real SH for m > 0: sqrt(2) * Re(Y_l^m)
                Y_real = np.sqrt(2) * Y_complex.real

            basis_list.append(Y_real.astype(np.float32))

    basis = np.stack(basis_list, axis=0)  # (N_coeffs, H, W)
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


class SHFusionModule(nn.Module):
    """
    Cross-attention fusion: decoder feature pixels attend to SH tokens.

    Inspired by HUSH's SH_Attention_Module. Creates N_sh compact tokens
    from the multi-channel SH map, then each feature pixel queries all
    SH tokens via multi-head cross-attention.

    Attention matrix shape: (B, heads, H*W, N_sh) — efficient because
    N_sh (15–55) is much smaller than H*W.
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
        # Pool each SH channel to a 4x4 spatial summary, then project to C dims.
        # Result: N_sh tokens each of dimension feature_channels.
        self.sh_pool = nn.AdaptiveAvgPool2d(4)          # (B, N_sh, 4, 4)
        self.sh_token_proj = nn.Sequential(
            nn.Linear(16, feature_channels),             # 4*4=16 -> C
            nn.ReLU(True),
        )

        # --- Cross-attention (pre-norm) ---
        self.norm_q = nn.LayerNorm(feature_channels)
        self.norm_kv = nn.LayerNorm(feature_channels)
        self.to_q = nn.Linear(feature_channels, feature_channels, bias=True)
        self.to_kv = nn.Linear(feature_channels, feature_channels * 2, bias=True)
        self.to_out = nn.Linear(feature_channels, feature_channels)

    def forward(self, features, sh_map):
        """
        Args:
            features: (B, C, H, W) decoder features
            sh_map:   (B, N_sh, H_full, W_full) multi-channel SH map
        Returns:
            fused: (B, C, H, W) attention-fused features (same shape as input)
        """
        B, C, H, W = features.shape

        # --- Build SH tokens: (B, N_sh, C) ---
        sh_resized = F.interpolate(
            sh_map, size=(H, W), mode='bilinear', align_corners=True
        )                                                # (B, N_sh, H, W)
        sh_pooled = self.sh_pool(sh_resized)             # (B, N_sh, 4, 4)
        sh_tokens = sh_pooled.flatten(2)                 # (B, N_sh, 16)
        sh_tokens = self.sh_token_proj(sh_tokens)        # (B, N_sh, C)

        # --- Flatten features to sequence: (B, HW, C) ---
        feat_seq = features.permute(0, 2, 3, 1).reshape(B, H * W, C)

        # --- Pre-norm ---
        q = self.to_q(self.norm_q(feat_seq))             # (B, HW, C)
        kv_normed = self.norm_kv(sh_tokens)              # (B, N_sh, C)
        k, v = self.to_kv(kv_normed).chunk(2, dim=-1)   # each (B, N_sh, C)

        # --- Multi-head reshape ---
        q = q.view(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, self.n_sh, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, self.n_sh, self.num_heads, self.head_dim).transpose(1, 2)
        # q: (B, heads, HW, d), k/v: (B, heads, N_sh, d)

        # --- Scaled dot-product attention ---
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)                   # (B, heads, HW, N_sh)

        out = torch.matmul(attn, v)                      # (B, heads, HW, d)
        out = out.transpose(1, 2).reshape(B, H * W, C)   # (B, HW, C)
        out = self.to_out(out)

        # --- Reshape back to spatial + residual ---
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)
        return features + out
