import torch.nn as nn
from .unetbaseline_model import UnetGenerator


class OracleERPGenerator(UnetGenerator):
    """Oracle model: Panoramic (ERP) RGB image -> ERP depth estimation.

    Uses the same UNet architecture as the baseline but takes an equi-
    rectangular panoramic RGB image (3 channels) as input instead of audio.
    This serves as an upper-bound reference for audio-based depth estimation
    methods.
    """

    def __init__(self, cfg, output_nc=1, num_downs=8, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        input_nc = 3  # RGB
        super().__init__(cfg, input_nc, output_nc, num_downs, ngf,
                         norm_layer=norm_layer, use_dropout=use_dropout)
