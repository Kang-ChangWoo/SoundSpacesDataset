from dataloader.SoundSpaces_Dataset import SoundSpacesDataset

from models.utils_models import *

from models.unetbaseline_model import *

from utils_tensorboard import *
from utils_criterion import compute_errors, get_valid_depth_mask, BerHuLoss

import time
import os 
import numpy as np 
import math
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from PIL import Image
import torchaudio
import torchaudio.transforms as T

import torch
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")  
def main(cfg):
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    
    if cfg.mode.mode != 'test':
        raise Exception('This script is for test only. Please run train.py for training')

    # ------------ GPU config ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print("{} {} device is used".format(n_GPU, device))

    batch_size = cfg.mode.batch_size
    
    # ------------ Create dataset -----------
    
    # Use SoundSpaces dataset class
    if cfg.dataset.name == 'soundspaces':
        eval_on = getattr(cfg.mode, 'eval_on', 'test')  # Default to 'test' if not in struct
        if eval_on == 'val':
            eval_set = SoundSpacesDataset(cfg, split='val')
        else:
            eval_set = SoundSpacesDataset(cfg, split='test')
    else:
        raise Exception('Test can be done only on soundspaces dataset')

    print(f'Eval Dataset of {len(eval_set)} instances')
    num_workers = cfg.mode.num_threads
    use_pin = torch.cuda.is_available()
    eval_loader = DataLoader(
        eval_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin, persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ---------- Load Model ----------
    # Determine input channels based on input type
    input_type = getattr(cfg.dataset, 'input_type', 'audio')
    if input_type == 'rgb':
        input_nc = 3  # RGB has 3 channels
        print('Input type: RGB (3 channels)')
    else:
        input_nc = 2  # Audio spectrogram has 2 channels (stereo)
        print('Input type: Audio (2 channels)')
    
    model = define_G(cfg, input_nc=input_nc, output_nc=1, ngf=64, netG=cfg.model.generator, norm='batch',
                     use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[device])
    print('Network used:', cfg.model.generator)
    
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
    elif cfg.mode.criterion == 'BerHu':
        criterion = BerHuLoss(threshold=0.2).to(device)

    if cfg.mode.checkpoints is None:
        raise AttributeError('In test mode, a checkpoint needs to be loaded.')
    else:
        load_epoch = cfg.mode.checkpoints
        
        # Construct experiment name the same way as train.py
        # Allow override with mode.experiment_name_full if provided
        if hasattr(cfg.mode, 'experiment_name_full') and cfg.mode.experiment_name_full:
            experiment_name = cfg.mode.experiment_name_full
        else:
            # Construct from components (same as train.py)
            experiment_name = (cfg.model.generator + '_' + cfg.dataset.name + '_' + 
                             'BS' + str(cfg.mode.batch_size) + '_' + 
                             'Lr' + str(cfg.mode.learning_rate) + '_' + 
                             cfg.mode.optimizer + '_' + cfg.mode.experiment_name)
        
        if str(load_epoch) == 'best':
            checkpoint_path = './checkpoints/' + experiment_name + '/best_model.pth'
        else:
            checkpoint_path = './checkpoints/' + experiment_name + '/checkpoint_' + str(load_epoch) + '.pth'
        print(f'Loading checkpoint from: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        if str(load_epoch) == 'best':
            print('Loaded best model (val loss: {:.6f})'.format(checkpoint.get('best_val_loss', float('nan'))))
        else:
            print('Epoch loaded:', str(load_epoch))
    

    # ------ Eval ---------
    model.eval()  # eval mode

    # Create output directory for saving results (use the same experiment_name)
    eval_on = getattr(cfg.mode, 'eval_on', 'test')  # Default to 'test' if not in struct
    output_dir = os.path.join('./outputs', experiment_name, f'epoch_{load_epoch}', eval_on)
    os.makedirs(output_dir, exist_ok=True)
    max_vis_samples = getattr(cfg.mode, 'max_vis_samples', 500)
    if max_vis_samples < 0:
        max_vis_samples = float('inf')  # Visualize all
    print(f'Results will be saved to: {output_dir}')
    print(f'Visualization: first {max_vis_samples} samples (set mode.max_vis_samples=-1 for all)')

    loss_list = []  # per-batch losses
    errors = []  # per-sample error tuples: (abs_rel, rmse, a1, a2, a3, log10, mae)
    sample_scene_ids = []  # per-sample scene info
    sample_step_idxs = []
    no_depth_ratios = []  # per-sample no-depth ratio

    with torch.no_grad():

        for batch_idx, (input_data, depthgt) in enumerate(eval_loader):
            # input_data can be audio (spectrogram) or RGB image depending on input_type
            input_data = input_data.to(device)
            depthgt = depthgt.to(device)        

            output = model(input_data)
            if isinstance(output, tuple):
                depth_pred = output[0]
            else:
                depth_pred = output

            # compute test loss - mask out invalid depth values (0 = missing/invalid depth)
            valid_mask = get_valid_depth_mask(depthgt)
            if valid_mask.sum() > 0:  # Only compute loss if there are valid pixels
                loss_test = criterion(depth_pred[valid_mask], depthgt[valid_mask])
            else:
                # If no valid pixels, set loss to 0 (shouldn't happen in practice)
                loss_test = torch.tensor(0.0, device=device) 
            loss_list.append(loss_test.cpu().item())

            for idx in range(depth_pred.shape[0]):
                # Calculate actual dataset index and get scene info
                dataset_idx = batch_idx * batch_size + idx
                if dataset_idx >= len(eval_set.samples):
                    break  # Last batch may be padded
                scene_id, step_idx = eval_set.samples[dataset_idx]
                
                # Get numpy arrays
                gt_np = depthgt[idx].detach().cpu().numpy()
                pred_np = depth_pred[idx].detach().cpu().numpy()
                input_np = input_data[idx].detach().cpu().numpy()
                
                # Unscale if normalized
                if cfg.dataset.depth_norm:
                    unscaledgt = gt_np * cfg.dataset.max_depth
                    unscaledpred = pred_np * cfg.dataset.max_depth
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(unscaledgt, unscaledpred, scene_id=scene_id, step_idx=step_idx)
                else:   
                    unscaledgt = gt_np
                    unscaledpred = pred_np
                    abs_rel, rmse, a1, a2, a3, log_10, mae = compute_errors(gt_np, pred_np, scene_id=scene_id, step_idx=step_idx)
                
                errors.append((abs_rel, rmse, a1, a2, a3, log_10, mae))
                sample_scene_ids.append(scene_id)
                sample_step_idxs.append(step_idx)
                
                # Count no-depth regions (where depth == 0)
                no_depth_gt = np.sum(unscaledgt == 0)
                no_depth_total = unscaledgt.size
                no_depth_ratio_gt = no_depth_gt / no_depth_total if no_depth_total > 0 else 0
                no_depth_ratios.append(float(no_depth_ratio_gt))
                
                # Save images (only for first max_vis_samples)
                if dataset_idx < max_vis_samples:
                    save_sample_results(output_dir, dataset_idx, input_np, unscaledgt, unscaledpred, 
                                      cfg, scene_id, step_idx, depth_type=getattr(cfg.dataset, 'depth_type', 'pinhole'))
            
            if (batch_idx + 1) % 10 == 0:
                total_samples = min((batch_idx + 1) * batch_size, len(eval_set))
                print(f'Processed {batch_idx + 1}/{len(eval_loader)} batches, {total_samples} samples')


        errors_np = np.array(errors)
        mean_errors = errors_np.mean(0)
        
        # Calculate no-depth statistics
        avg_no_depth_ratio = np.mean(no_depth_ratios) if no_depth_ratios else 0
        
        print('=' * 50)
        print('Test Results:')
        print('=' * 50)
        print('abs rel: {:.3f}'.format(mean_errors[0])) 
        print('RMSE: {:.3f}'.format(mean_errors[1])) 
        print('Delta1: {:.3f}'.format(mean_errors[2])) 
        print('Delta2: {:.3f}'.format(mean_errors[3])) 
        print('Delta3: {:.3f}'.format(mean_errors[4])) 
        print('Log10: {:.3f}'.format(mean_errors[5])) 
        print('MAE: {:.3f}'.format(mean_errors[6]))
        print('-' * 50)
        print('No-Depth Region Statistics:')
        print('Average no-depth ratio: {:.3f}%'.format(avg_no_depth_ratio * 100))
        print('Max depth (clipped at): {}m'.format(cfg.dataset.max_depth))
        print('=' * 50)
    
    # Save evaluation statistics - all per-sample arrays have the same length
    d = {
        'scene_id': sample_scene_ids,
        'step_idx': sample_step_idxs,
        'abs_rel': errors_np[:, 0].tolist(),
        'rmse': errors_np[:, 1].tolist(),
        'delta1': errors_np[:, 2].tolist(),
        'delta2': errors_np[:, 3].tolist(),
        'delta3': errors_np[:, 4].tolist(),
        'log10': errors_np[:, 5].tolist(),
        'mae': errors_np[:, 6].tolist(),
        'no_depth_ratio': no_depth_ratios,
    }

    stats_df = pd.DataFrame(data=d)
    
    # Create stats directory if it doesn't exist
    eval_on = getattr(cfg.mode, 'eval_on', 'test')  # Default to 'test' if not in struct
    stats_dir = os.path.join(cfg.mode.stat_dir, cfg.dataset.name, eval_on)
    os.makedirs(stats_dir, exist_ok=True)
    
    if eval_on == 'test':
        stats_df.to_pickle(os.path.join(stats_dir, 'stats_on_' + cfg.dataset.name + '_test_set_' + experiment_name + '_epoch_' + str(load_epoch) + ".pkl"))
    else:
        stats_df.to_pickle(os.path.join(stats_dir, 'stats_on_' + cfg.dataset.name + '_val_set_' + experiment_name + '_epoch_' + str(load_epoch) + ".pkl"))
    
    print(f'Statistics saved to: {stats_dir}')
    print(f'Visualization results saved to: {output_dir}')


def load_rgb_image(cfg, scene_id, step_idx, depth_type='pinhole'):
    """Load RGB image corresponding to the depth type"""
    scene_dir = os.path.join(cfg.dataset.dataset_dir, scene_id)
    
    if depth_type == 'erp':
        erp_rgb_dir = os.path.join(scene_dir, 'erp_rgb')
        if os.path.exists(erp_rgb_dir):
            rgb_path = os.path.join(erp_rgb_dir, f'erp_{step_idx:03d}.png')
            if os.path.exists(rgb_path):
                return np.array(Image.open(rgb_path))
    else:  # pinhole
        pinhole_rgb_dir = os.path.join(scene_dir, 'pinhole_rgb')
        if os.path.exists(pinhole_rgb_dir):
            rgb_path = os.path.join(pinhole_rgb_dir, f'pinhole_{step_idx:03d}.png')
            if os.path.exists(rgb_path):
                return np.array(Image.open(rgb_path))
    
    return None  # RGB image not found


def save_sample_results(output_dir, sample_idx, input_np, gt_depth, pred_depth, cfg, scene_id, step_idx, depth_type='pinhole'):
    """
    Save visualization results for a single sample:
    - Input RGB image (if available)
    - Input spectrogram (if audio input)
    - Ground truth depth
    - Predicted depth
    """
    input_type = getattr(cfg.dataset, 'input_type', 'audio')
    
    # Load RGB image (if not already the input)
    rgb_image = None
    if input_type != 'rgb':
        rgb_image = load_rgb_image(cfg, scene_id, step_idx, depth_type)
    
    # Save input visualization
    if input_type == 'rgb':
        # Input is RGB, save it
        rgb_path = os.path.join(output_dir, f'sample_{sample_idx:05d}_input_rgb.png')
        # Convert from (C, H, W) to (H, W, C) for PIL
        rgb_vis = np.transpose(input_np, (1, 2, 0))
        # Denormalize if needed (assuming values are in [0, 1])
        if rgb_vis.max() <= 1.0:
            rgb_vis = (rgb_vis * 255).astype(np.uint8)
        Image.fromarray(rgb_vis).save(rgb_path)
        rgb_image = rgb_vis  # Use for comparison plot
    else:
        # Input is audio, save spectrogram
        spec_path = os.path.join(output_dir, f'sample_{sample_idx:05d}_spectrogram.png')
        save_spectrogram(input_np, spec_path, scene_id=scene_id, step_idx=step_idx)
    
    # Save depth maps
    gt_path = os.path.join(output_dir, f'sample_{sample_idx:05d}_gt_depth.png')
    pred_path = os.path.join(output_dir, f'sample_{sample_idx:05d}_pred_depth.png')
    
    save_depth_map(gt_depth, gt_path, cfg.dataset.max_depth, 
                   title=f'Ground Truth Depth | Scene: {scene_id} | Step: {step_idx:03d}')
    save_depth_map(pred_depth, pred_path, cfg.dataset.max_depth, 
                   title=f'Predicted Depth | Scene: {scene_id} | Step: {step_idx:03d}')
    
    # Save side-by-side comparison
    comparison_path = os.path.join(output_dir, f'sample_{sample_idx:05d}_comparison.png')
    save_comparison(input_np, gt_depth, pred_depth, comparison_path, cfg.dataset.max_depth, depth_type, rgb_image, 
                   scene_id=scene_id, step_idx=step_idx, input_type=input_type)


def save_spectrogram(audio_np, save_path, scene_id=None, step_idx=None):
    """Save spectrogram visualization"""
    # audio_np shape: (2, H, W) for stereo channels
    if len(audio_np.shape) == 3 and audio_np.shape[0] == 2:
        spec_vis = audio_np[0]  # Left channel
    else:
        spec_vis = audio_np
    
    # Convert to log scale for better visualization
    spec_vis = np.log10(spec_vis + 1e-10)
    
    plt.figure(figsize=(10, 6))
    plt.imshow(spec_vis, aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Log Magnitude')
    
    # Build title with scene_id and step_idx if provided
    title = 'Input Spectrogram'
    if scene_id is not None and step_idx is not None:
        title += f' | Scene: {scene_id} | Step: {step_idx:03d}'
    elif scene_id is not None:
        title += f' | Scene: {scene_id}'
    elif step_idx is not None:
        title += f' | Step: {step_idx:03d}'
    
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _squeeze_depth(depth):
    """Helper function to convert depth array to 2D"""
    if len(depth.shape) == 3:
        # If 3D, squeeze out channel dimension if it's 1, otherwise take first channel
        if depth.shape[0] == 1:
            return depth.squeeze(0)
        else:
            return depth[0]  # Take first channel
    elif len(depth.shape) == 2:
        return depth
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}. Expected 2D or 3D array.")


def save_depth_map(depth, save_path, max_depth, title='Depth Map'):
    """Save depth map visualization"""
    # Convert to 2D array
    depth = _squeeze_depth(depth)
    
    plt.figure(figsize=(8, 8))
    # Mask zero values for better visualization
    depth_masked = np.ma.masked_where(depth == 0, depth)
    im = plt.imshow(depth_masked, cmap='jet', vmin=0, vmax=max_depth)
    plt.colorbar(im, label='Depth (meters)')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _squeeze_depth(depth):
    """Helper function to convert depth array to 2D"""
    if len(depth.shape) == 3:
        # If 3D, squeeze out channel dimension if it's 1, otherwise take first channel
        if depth.shape[0] == 1:
            return depth.squeeze(0)
        else:
            return depth[0]  # Take first channel
    elif len(depth.shape) == 2:
        return depth
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}. Expected 2D or 3D array.")


def save_comparison(input_np, gt_depth, pred_depth, save_path, max_depth, depth_type, rgb_image=None, scene_id=None, step_idx=None, input_type='audio'):
    """Save side-by-side comparison of all outputs"""
    # Convert depth arrays to 2D
    gt_depth = _squeeze_depth(gt_depth)
    pred_depth = _squeeze_depth(pred_depth)
    
    # Determine if we should show RGB
    show_rgb = rgb_image is not None or input_type == 'rgb'
    if input_type == 'rgb' and rgb_image is None:
        # Input is RGB, convert from (C, H, W) to (H, W, C)
        rgb_image = np.transpose(input_np, (1, 2, 0))
        if rgb_image.max() <= 1.0:
            rgb_image = (rgb_image * 255).astype(np.uint8)
    
    if show_rgb:
        # 2x3 grid if RGB is available
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Row 0: RGB, Spectrogram, GT Depth
        axes[0, 0].imshow(rgb_image)
        axes[0, 0].set_title('Input RGB Image')
        axes[0, 0].axis('off')
        
        # Input visualization (spectrogram or RGB)
        if input_type == 'rgb':
            # RGB input - already shown in first subplot
            axes[0, 1].axis('off')
        else:
            # Audio spectrogram
            if len(input_np.shape) == 3 and input_np.shape[0] == 2:
                spec_vis = input_np[0]
            else:
                spec_vis = input_np
            spec_vis = np.log10(spec_vis + 1e-10)
            axes[0, 1].imshow(spec_vis, aspect='auto', origin='lower', cmap='viridis')
            axes[0, 1].set_title('Input Spectrogram')
            axes[0, 1].axis('off')
        
        # Ground truth depth
        gt_masked = np.ma.masked_where(gt_depth == 0, gt_depth)
        im1 = axes[0, 2].imshow(gt_masked, cmap='jet', vmin=0, vmax=max_depth)
        axes[0, 2].set_title(f'Ground Truth Depth ({depth_type})')
        axes[0, 2].axis('off')
        plt.colorbar(im1, ax=axes[0, 2], label='Depth (m)')
        
        # Row 1: Empty, Predicted Depth, Error
        axes[1, 0].axis('off')
        
        # Predicted depth
        pred_masked = np.ma.masked_where(pred_depth == 0, pred_depth)
        im2 = axes[1, 1].imshow(pred_masked, cmap='jet', vmin=0, vmax=max_depth)
        axes[1, 1].set_title('Predicted Depth')
        axes[1, 1].axis('off')
        plt.colorbar(im2, ax=axes[1, 1], label='Depth (m)')
        
        # Error map
        error_map = np.abs(gt_depth - pred_depth)
        error_masked = np.ma.masked_where(gt_depth == 0, error_map)
        im3 = axes[1, 2].imshow(error_masked, cmap='hot', vmin=0, vmax=max_depth * 0.3)
        axes[1, 2].set_title('Absolute Error')
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], label='Error (m)')
    else:
        # 2x2 grid if no RGB
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Input visualization (spectrogram or RGB)
        if input_type == 'rgb':
            # RGB input
            rgb_vis = np.transpose(input_np, (1, 2, 0))
            if rgb_vis.max() <= 1.0:
                rgb_vis = (rgb_vis * 255).astype(np.uint8)
            axes[0, 0].imshow(rgb_vis)
            axes[0, 0].set_title('Input RGB Image')
            axes[0, 0].axis('off')
        else:
            # Audio spectrogram
            if len(input_np.shape) == 3 and input_np.shape[0] == 2:
                spec_vis = input_np[0]
            else:
                spec_vis = input_np
            spec_vis = np.log10(spec_vis + 1e-10)
            axes[0, 0].imshow(spec_vis, aspect='auto', origin='lower', cmap='viridis')
            axes[0, 0].set_title('Input Spectrogram')
            axes[0, 0].axis('off')
        
        # Ground truth depth
        gt_masked = np.ma.masked_where(gt_depth == 0, gt_depth)
        im1 = axes[0, 1].imshow(gt_masked, cmap='jet', vmin=0, vmax=max_depth)
        axes[0, 1].set_title(f'Ground Truth Depth ({depth_type})')
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], label='Depth (m)')
        
        # Predicted depth
        pred_masked = np.ma.masked_where(pred_depth == 0, pred_depth)
        im2 = axes[1, 0].imshow(pred_masked, cmap='jet', vmin=0, vmax=max_depth)
        axes[1, 0].set_title('Predicted Depth')
        axes[1, 0].axis('off')
        plt.colorbar(im2, ax=axes[1, 0], label='Depth (m)')
        
        # Error map
        error_map = np.abs(gt_depth - pred_depth)
        error_masked = np.ma.masked_where(gt_depth == 0, error_map)
        im3 = axes[1, 1].imshow(error_masked, cmap='hot', vmin=0, vmax=max_depth * 0.3)
        axes[1, 1].set_title('Absolute Error')
        axes[1, 1].axis('off')
        plt.colorbar(im3, ax=axes[1, 1], label='Error (m)')
    
    # Build main title with scene_id and step_idx
    title = f'Max Depth: {max_depth}m | Depth Type: {depth_type}'
    if scene_id is not None and step_idx is not None:
        title += f' | Scene: {scene_id} | Step: {step_idx:03d}'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Exception happened during test")
        import traceback
        traceback.print_exc()
