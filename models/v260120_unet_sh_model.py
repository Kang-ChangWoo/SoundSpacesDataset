"""
UNet with Spherical Harmonics (SH) for ERP depth estimation.

This module implements a non-recursive UNet encoder-decoder with an
SH branch that extracts spherical harmonic coefficients from the
bottleneck features and uses them to modulate decoder features via
a learned gating mechanism.

Architecture:
    Input -> Encoder -> Bottleneck -> SH Coefficient Extractor
                                          |
                        Decoder <--- SH Fusion (gate)
                           |
                        Depth Output

The SH branch provides global spherical geometry priors that are
especially effective for ERP images, which represent the full sphere.

Adapted from HUSH (https://github.com/vision3d-lab/HUSH).
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v260120_sh_utils import build_real_sh_basis, SHCoeffExtractor, SHFusionModule


class UnetSHGenerator(nn.Module):
    """
    Non-recursive UNet with Spherical Harmonics branch for ERP depth.

    Compared to the recursive UnetGenerator, this version:
    1. Stores encoder features explicitly for skip connections
    2. Adds SH coefficient extraction from the bottleneck
    3. Fuses SH-based geometric prior into decoder features via cross-attention
    """

    def __init__(self, cfg, input_nc, output_nc, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False,
                 sh_degree=10, image_size=(256, 512)):
        super().__init__()

        self.num_downs = num_downs
        self.depth_norm = cfg.dataset.depth_norm

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # ---- Pre-compute SH basis ----
        n_sh = sh_degree * (sh_degree + 1) // 2
        sh_basis = build_real_sh_basis(sh_degree, image_size[0], image_size[1])
        self.register_buffer('sh_basis', sh_basis)  # (N_sh, H, W)

        # ---- Encoder ----
        # Level 0 (outermost): no activation before, no norm
        self.enc0 = nn.Conv2d(input_nc, ngf, 4, 2, 1)

        # Levels 1 to num_downs-2
        encoder_layers = []
        in_ch = ngf
        for i in range(1, num_downs - 1):
            out_ch = min(ngf * (2 ** i), ngf * 8)
            encoder_layers.append(nn.Sequential(
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=use_bias),
                norm_layer(out_ch),
            ))
            in_ch = out_ch
        self.encoders = nn.ModuleList(encoder_layers)

        # Innermost encoder (no norm)
        self.enc_inner = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_ch, ngf * 8, 4, 2, 1),
        )

        # ---- SH Branch ----
        self.sh_extractor = SHCoeffExtractor(
            in_channels=ngf * 8,
            n_coeffs=n_sh,
            hidden_dim=ngf * 4,
        )
        self.sh_fusion_mid = SHFusionModule(
            feature_channels=ngf * 4,  # after second-to-last skip: ngf*2 + ngf*2 = ngf*4
            n_sh=n_sh,
            num_heads=4,
        )
        self.sh_fusion_late = SHFusionModule(
            feature_channels=ngf * 2,  # after last skip: ngf + ngf = ngf*2
            n_sh=n_sh,
            num_heads=4,
        )

        # ---- Decoder ----
        decoder_layers = []

        # Innermost decoder: input is bottleneck (ngf*8), no skip
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 8),
        ))

        # Middle decoders: input = skip(ngf*8) + prev(ngf*8) = ngf*16
        for _ in range(num_downs - 5):
            layers = [
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=use_bias),
                norm_layer(ngf * 8),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            decoder_layers.append(nn.Sequential(*layers))

        # Transition decoders: reducing channels
        # trans1: skip(ngf*8) + prev(ngf*8) = ngf*16 -> ngf*4
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
        ))
        # trans2: skip(ngf*4) + prev(ngf*4) = ngf*8 -> ngf*2
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
        ))
        # trans3: skip(ngf*2) + prev(ngf*2) = ngf*4 -> ngf
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
        ))

        self.decoders = nn.ModuleList(decoder_layers)

        # Outermost decoder: skip(ngf) + prev(ngf) = ngf*2 -> output_nc
        if self.depth_norm:
            self.dec_outer = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.dec_outer = nn.Sequential(
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
                nn.ReLU(),
            )

    def forward(self, x):
        # ---- Encode ----
        enc_features = []
        h = self.enc0(x)
        enc_features.append(h)

        for enc in self.encoders:
            h = enc(h)
            enc_features.append(h)

        bottleneck = self.enc_inner(h)

        # ---- SH coefficient extraction ----
        sh_coeffs = self.sh_extractor(bottleneck)  # (B, N_sh)
        sh_map = torch.einsum('bn,nhw->bnhw', sh_coeffs, self.sh_basis)  # (B, N_sh, H, W)

        # ---- Decode ----
        enc_reversed = enc_features[::-1]  # [e_{n-2}, ..., e1, e0]

        h = self.decoders[0](bottleneck)  # Innermost decoder

        for i in range(len(self.decoders) - 2):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)

        # ---- SH Fusion (mid-level) ----
        # Second-to-last skip: ngf*2 + ngf*2 = ngf*4 channels
        h = torch.cat([enc_reversed[len(self.decoders) - 2], h], dim=1)
        h = self.sh_fusion_mid(h, sh_map)
        h = self.decoders[-1](h)

        # ---- SH Fusion (late) ----
        # Final skip connection (with e0): ngf + ngf = ngf*2 channels
        h = torch.cat([enc_reversed[-1], h], dim=1)
        h = self.sh_fusion_late(h, sh_map)

        # ---- Output ----
        depth = self.dec_outer(h)
        return depth, sh_map
