"""
UNet with Spherical Harmonics (SH) v2 for ERP depth estimation.

Extends v260120 with bidirectional cross-attention fusion:
  Pass 1: Feature pixels query SH tokens (learn geometry)
  Pass 2: SH tokens query feature pixels (learn spatial context)
  Final:  Features attend to enriched SH tokens

Architecture is otherwise identical to v260120_unet_sh_model.py:
  Non-recursive encoder/decoder with skip connections,
  SH coefficient extraction from bottleneck,
  SH fusion at mid and late decoder levels.

Adapted from HUSH (https://github.com/vision3d-lab/HUSH).
"""

import functools
import torch
import torch.nn as nn
import torch.nn.functional as F

from .v260217_sh_utils import build_real_sh_basis, SHCoeffExtractor, SHBidirectionalFusionModule


class UnetSHv2Generator(nn.Module):
    """
    Non-recursive UNet with bidirectional SH cross-attention for ERP depth.

    Compared to UnetSHGenerator (v260120):
    - SHFusionModule replaced by SHBidirectionalFusionModule
    - Each fusion point performs 3-pass attention: Feature→SH, SH→Feature, readout
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
        self.enc0 = nn.Conv2d(input_nc, ngf, 4, 2, 1)

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
        self.sh_fusion_mid = SHBidirectionalFusionModule(
            feature_channels=ngf * 4,
            n_sh=n_sh,
            num_heads=4,
        )
        self.sh_fusion_late = SHBidirectionalFusionModule(
            feature_channels=ngf * 2,
            n_sh=n_sh,
            num_heads=4,
        )

        # ---- Decoder ----
        decoder_layers = []

        # Innermost decoder
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 8),
        ))

        # Middle decoders
        for _ in range(num_downs - 5):
            layers = [
                nn.ReLU(True),
                nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, 4, 2, 1, bias=use_bias),
                norm_layer(ngf * 8),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            decoder_layers.append(nn.Sequential(*layers))

        # Transition decoders
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
        ))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
        ))
        decoder_layers.append(nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2 * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
        ))

        self.decoders = nn.ModuleList(decoder_layers)

        # Outermost decoder
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
        enc_reversed = enc_features[::-1]

        h = self.decoders[0](bottleneck)  # Innermost decoder

        for i in range(len(self.decoders) - 2):
            h = torch.cat([enc_reversed[i], h], dim=1)
            h = self.decoders[i + 1](h)

        # ---- SH Bidirectional Fusion (mid-level) ----
        h = torch.cat([enc_reversed[len(self.decoders) - 2], h], dim=1)
        h = self.sh_fusion_mid(h, sh_map)
        h = self.decoders[-1](h)

        # ---- SH Bidirectional Fusion (late) ----
        h = torch.cat([enc_reversed[-1], h], dim=1)
        h = self.sh_fusion_late(h, sh_map)

        # ---- Output ----
        depth = self.dec_outer(h)
        return depth, sh_map
