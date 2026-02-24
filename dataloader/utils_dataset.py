
import torch
import torchvision.transforms as transforms

import os, glob
import pandas as pd
import numpy as np

# Try to import InterpolationMode (available in newer torchvision versions)
try:
    from torchvision.transforms import InterpolationMode
    USE_INTERPOLATION_MODE = True
except ImportError:
    USE_INTERPOLATION_MODE = False
    # For older versions, use integer: 0 = nearest, 1 = bilinear, 2 = bicubic


def get_transform(cfg, convert=False, depth_norm=False):
    # Create list of transform to apply to data
    transform_list = []

    if convert:
        # Convert data to Tensor type
        transform_list += [transforms.ToTensor()]

    # Parse target size (shared between resize and crop modes)
    images_size = cfg.dataset.images_size
    # Handle OmegaConf ListConfig and regular list/tuple
    if hasattr(images_size, '__iter__') and not isinstance(images_size, (str, bytes)):
        # Convert to tuple (handles list, tuple, ListConfig, etc.)
        try:
            target_size = tuple(int(x) for x in images_size)  # (H, W)
        except (TypeError, ValueError):
            # Fallback: try to convert single value
            target_size = (int(images_size), int(images_size))
    else:
        # Single value (square)
        target_size = (int(images_size), int(images_size))

    if 'crop' in cfg.dataset.preprocess:
        # Resize maintaining aspect ratio (shorter edge to max(target_size)),
        # then center crop to exact target_size.
        # This avoids aspect ratio distortion (no pixel stretching/squishing).
        max_dim = max(target_size)
        if USE_INTERPOLATION_MODE:
            transform_list.append(transforms.Resize(
                max_dim,
                interpolation=InterpolationMode.NEAREST
            ))
        else:
            transform_list.append(transforms.Resize(
                max_dim,
                interpolation=0
            ))
        transform_list.append(transforms.CenterCrop(target_size))

    elif 'resize' in cfg.dataset.preprocess:
        # Resize to specified image size (e.g., 192x384 or 256x256)
        # Use nearest neighbor interpolation (sampling) instead of bilinear interpolation
        # Nearest neighbor just picks the closest pixel value without blending
        if USE_INTERPOLATION_MODE:
            transform_list.append(transforms.Resize(
                target_size,
                interpolation=InterpolationMode.NEAREST  # Nearest neighbor = sampling, no interpolation
            ))
        else:
            # Fallback for older torchvision: 0 = nearest neighbor
            transform_list.append(transforms.Resize(
                target_size,
                interpolation=0  # 0 = nearest neighbor (sampling)
            ))

    # Note: depth_norm is ignored if MinMaxNorm is removed
    # Depth normalization should be handled separately if needed

    return transforms.Compose(transform_list)
