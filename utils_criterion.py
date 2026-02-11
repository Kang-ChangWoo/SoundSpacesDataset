
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BerHuLoss(nn.Module):
    """
    Reverse Huber (BerHu) loss for depth estimation.

    For small errors (|e| <= delta): L1 loss
    For large errors (|e| > delta): L2-like loss

    This is better than pure L1 for depth because it:
    - Preserves sharpness for small errors (like L1)
    - Penalizes large errors more strongly (like L2)
    - Automatically adapts the threshold per batch

    From HUSH (https://github.com/vision3d-lab/HUSH).
    """

    def __init__(self, threshold=0.2):
        super().__init__()
        self.threshold = threshold

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted depth (masked to valid pixels only)
            target: Ground truth depth (masked to valid pixels only)
        """
        diff = torch.abs(target - pred)
        delta = self.threshold * torch.max(diff).item()

        # L1 part: |e| <= delta
        part1 = -F.threshold(-diff, -delta, 0.)
        # L2 part: |e| > delta -> (e^2 + delta^2) / (2*delta)
        part2 = F.threshold(diff ** 2 + delta ** 2, 2.0 * delta ** 2, 0.)
        part2 = part2 / (2. * delta + 1e-8)

        loss = (part1 + part2).mean()
        return loss


class GradientLoss(nn.Module):
    """
    Gradient loss that penalizes differences in spatial gradients.

    Encourages depth maps with sharp edges matching the ground truth,
    preventing over-smoothed predictions.

    From HUSH (https://github.com/vision3d-lab/HUSH).
    """

    def __init__(self):
        super().__init__()
        x_kernel = torch.FloatTensor([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
        y_kernel = torch.FloatTensor([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
        self.register_buffer('weight_x', x_kernel.unsqueeze(0).unsqueeze(0))
        self.register_buffer('weight_y', y_kernel.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target, valid_mask):
        """
        Args:
            pred: (B, 1, H, W) predicted depth
            target: (B, 1, H, W) ground truth depth
            valid_mask: (B, 1, H, W) boolean mask for valid depth
        """
        diff = (pred - target) * valid_mask.float()
        gx = F.conv2d(diff, self.weight_x, padding=1)
        gy = F.conv2d(diff, self.weight_y, padding=1)
        loss = (torch.abs(gx) + torch.abs(gy)).mean()
        return loss


class SHAuxiliaryLoss(nn.Module):
    """
    Auxiliary loss for direct SH map supervision.

    SH can only represent low-frequency structure, so we match the SH map
    against a smoothed version of the ground truth depth. This forces the
    SH branch to learn meaningful depth coefficients rather than relying
    solely on the final depth loss for gradient signal.
    """

    def __init__(self, pool_size=16):
        super().__init__()
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size)
        self.upsample_mode = 'bilinear'

    def forward(self, sh_map, gt_depth):
        """
        Args:
            sh_map: (B, N_sh, H, W) multi-channel SH map (or (B, 1, H, W) legacy)
            gt_depth: (B, 1, H, W) ground truth depth
        Returns:
            loss: scalar L1 loss between summed SH map and smoothed GT on valid pixels
        """
        # Sum SH channels to reconstruct scalar depth-like signal
        if sh_map.shape[1] > 1:
            sh_reconstructed = sh_map.sum(dim=1, keepdim=True)  # (B, 1, H, W)
        else:
            sh_reconstructed = sh_map

        # Smooth the ground truth to match SH's low-frequency nature
        smoothed_gt = self.pool(gt_depth)
        smoothed_gt = F.interpolate(
            smoothed_gt, size=gt_depth.shape[2:],
            mode=self.upsample_mode, align_corners=True
        )
        # Resize SH map to match GT if needed
        if sh_reconstructed.shape[2:] != gt_depth.shape[2:]:
            sh_reconstructed = F.interpolate(
                sh_reconstructed, size=gt_depth.shape[2:],
                mode=self.upsample_mode, align_corners=True
            )
        valid = get_valid_depth_mask(gt_depth)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=sh_map.device, requires_grad=True)
        return F.l1_loss(sh_reconstructed[valid], smoothed_gt[valid])


def get_valid_depth_mask(depth, epsilon=1e-6):
    """
    Create a mask for valid depth values (non-zero and positive).
    Masks out 0 values which represent invalid/missing depth measurements.
    
    Args:
        depth: Depth tensor/array (torch.Tensor or np.ndarray)
        epsilon: Small value to handle floating point precision (default: 1e-6)
    
    Returns:
        mask: Boolean mask where True indicates valid depth values
    """
    if isinstance(depth, torch.Tensor):
        # For torch tensors: mask out 0 and negative values
        return depth > epsilon
    else:
        # For numpy arrays: mask out 0 and negative values
        return depth > epsilon


def compute_errors(gt, pred, scene_id=None, step_idx=None):
    """Computation of error metrics between predicted and ground truth depths
        Taken from Beyond Image to Depth Github repository
    
    Note: Zero values in depth maps represent invalid/missing depth and are masked out.
    Returns zeros for all metrics if no valid depth pixels are found.
    
    Args:
        gt: Ground truth depth array
        pred: Predicted depth array
        scene_id: Optional scene identifier for debugging (printed if empty array detected)
        step_idx: Optional step index for debugging (printed if empty array detected)
    """
    # Select only the values that are greater than zero (valid depth)
    # Use epsilon to handle floating point precision issues
    mask = get_valid_depth_mask(gt)
    pred = pred[mask]
    gt = gt[mask]
    
    # Check if there are any valid pixels after masking
    if len(gt) == 0 or len(pred) == 0:
        # No valid depth pixels, print warning with scene info if available
        if scene_id is not None and step_idx is not None:
            print(f"WARNING: Empty depth array detected - Scene: {scene_id}, Step: {step_idx:03d}")
        elif scene_id is not None:
            print(f"WARNING: Empty depth array detected - Scene: {scene_id}")
        elif step_idx is not None:
            print(f"WARNING: Empty depth array detected - Step: {step_idx:03d}")
        else:
            print("WARNING: Empty depth array detected (no valid depth pixels after masking)")
        # Return zeros for all metrics
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    if rmse != rmse:
        rmse = 0.0
    if a1 != a1:
        a1=0.0
    if a2 != a2:
        a2=0.0
    if a3 != a3:
        a3=0.0
    
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    log_10 = (np.abs(np.log10(gt)-np.log10(pred))).mean()
    mae = (np.abs(gt-pred)).mean()
    if abs_rel != abs_rel:
        abs_rel=0.0
    if log_10 != log_10:
        log_10=0.0
    if mae != mae:
        mae=0.0
    
    return abs_rel, rmse, a1, a2, a3, log_10, mae