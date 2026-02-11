from dataloader.SoundSpaces_Dataset import SoundSpacesDataset

from models.utils_models import *

from models.unetbaseline_model import *

from utils_criterion import compute_errors, get_valid_depth_mask, BerHuLoss, GradientLoss, SHAuxiliaryLoss

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

import torch
from torch.utils.data import DataLoader
import wandb

import hydra
from omegaconf import DictConfig, OmegaConf


def print_statistics(name, data, mask=None):
    """
    Print statistics (min, max, average) for a data tensor/array
    
    Args:
        name: Name/label for the data
        data: Data tensor or numpy array
        mask: Optional mask to apply before computing statistics
    """
    if isinstance(data, torch.Tensor):
        data_np = data.detach().cpu().numpy()
    else:
        data_np = data
    
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask_np = mask.detach().cpu().numpy()
        else:
            mask_np = mask
        data_np = data_np[mask_np]
    
    if data_np.size == 0:
        print(f"{name}: No valid data (empty after masking)")
        return
    
    print(f"{name}:")
    print(f"  Min: {np.min(data_np):.6f}, Max: {np.max(data_np):.6f}, Mean: {np.mean(data_np):.6f}")
    print(f"  Shape: {data_np.shape}, Valid pixels: {data_np.size}")


@hydra.main(version_base=None, config_path="conf", config_name="config")  
def main(cfg: DictConfig) -> None:
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    #print(OmegaConf.to_yaml(cfg))
    
    if cfg.mode.mode != 'train':
        raise Exception('This script is for training only. Please run test.py for evaluation')
    if cfg.model.name != 'unet_baseline':
        raise Exception('This script is for training on unet model only')

    # ------------ GPU config ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print("{} {} device is used".format(n_GPU, device))

    batch_size = cfg.mode.batch_size
    
    # ------------ Create dataset -----------
    
    # Use SoundSpaces dataset class
    if cfg.dataset.name == 'soundspaces':
        train_set = SoundSpacesDataset(cfg, split='train')
        if cfg.mode.validation:
            val_set = SoundSpacesDataset(cfg, split='val')
    else:
        raise Exception('Training can be done only on soundspaces dataset')

    print(f'Train Dataset of {len(train_set)} instances')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=cfg.mode.shuffle, num_workers=cfg.mode.num_threads) 

    if cfg.mode.validation:
        print(f'Validation Dataset of {len(val_set)} instances')
        # Validation should not shuffle to maintain consistent indexing for visualization
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=cfg.mode.num_threads)


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
    print('Model used:', cfg.model.generator)
   
    # ---------- Criterion & Optimizers ----------
    if cfg.mode.criterion == 'L1':
        criterion = nn.L1Loss().to(device)
    elif cfg.mode.criterion == 'BerHu':
        criterion = BerHuLoss(threshold=0.2).to(device)

    # Gradient loss (optional, used alongside primary criterion)
    use_grad_loss = getattr(cfg.mode, 'use_grad_loss', False)
    grad_loss_weight = getattr(cfg.mode, 'grad_loss_weight', 0.5)
    if use_grad_loss:
        grad_criterion = GradientLoss().to(device)
        print(f'Using gradient loss with weight {grad_loss_weight}')

    # SH auxiliary loss (optional, for SH model variants)
    use_sh_aux_loss = getattr(cfg.mode, 'use_sh_aux_loss', False)
    sh_aux_loss_weight = getattr(cfg.mode, 'sh_aux_loss_weight', 0.1)
    if use_sh_aux_loss:
        sh_aux_criterion = SHAuxiliaryLoss().to(device)
        print(f'Using SH auxiliary loss with weight {sh_aux_loss_weight}')

    learning_rate = cfg.mode.learning_rate

    if cfg.mode.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    elif cfg.mode.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # ---------- TensorBoard ---------- 
    
    experiment_name = cfg.model.generator + '_' + cfg.dataset.name + '_' + 'BS' + str(cfg.mode.batch_size) + '_' + 'Lr' + str(cfg.mode.learning_rate) + '_' + cfg.mode.optimizer + '_' + cfg.mode.experiment_name

    # Initialize Weights & Biases
    wandb.init(
        project="soundspaces-depth",
        name=experiment_name,
        config={
            "batch_size": cfg.mode.batch_size,
            "epochs": cfg.mode.epochs,
            "learning_rate": cfg.mode.learning_rate,
            "optimizer": cfg.mode.optimizer,
            "criterion": cfg.mode.criterion,
            "model": cfg.model.generator,
            "dataset": cfg.dataset.name,
            "depth_type": getattr(cfg.dataset, 'depth_type', 'pinhole'),
            "input_type": getattr(cfg.dataset, 'input_type', 'audio'),
            "use_grad_loss": getattr(cfg.mode, 'use_grad_loss', False),
            "grad_loss_weight": getattr(cfg.mode, 'grad_loss_weight', 0.5),
            "use_sh_aux_loss": getattr(cfg.mode, 'use_sh_aux_loss', False),
            "sh_aux_loss_weight": getattr(cfg.mode, 'sh_aux_loss_weight', 0.1),
            "sh_degree": getattr(cfg.model, 'sh_degree', None),
            "images_size": list(cfg.dataset.images_size),
            "max_depth": cfg.dataset.max_depth,
        }
    )

    # Create output directory for validation plots
    val_output_dir = os.path.join('./outputs', experiment_name, 'validation_plots')
    os.makedirs(val_output_dir, exist_ok=True)
    print(f'Validation plots will be saved to: {val_output_dir}')

    log_dir = './logs/' + experiment_name + '/'
    os.makedirs(log_dir, exist_ok=True)
    file = open(log_dir + "architecture.txt","w")

    # dataloader parameters
    file.write("Dataset name: {}\n".format(cfg.dataset.name))
    file.write("Batch size: {}\n".format(batch_size))
    file.write("Image processing: {}\n".format(cfg.dataset.preprocess))
    file.write("Image resize: {}\n".format(cfg.dataset.images_size))
    file.write("Depth norm: {}\n".format(cfg.dataset.depth_norm))
    input_type = getattr(cfg.dataset, 'input_type', 'audio')
    file.write("Input type: {}\n".format(input_type))
    if input_type == 'audio':
        file.write("Audio type used for training: {}\n".format(cfg.dataset.audio_format))
    
    # parameters
    file.write("Learning rate: {}\n".format(cfg.mode.learning_rate))
    file.write("Optimizer used: {}\n".format(cfg.mode.optimizer))

    # net architecture
    file.write("Generator: {}\n".format(cfg.model.generator))
    
    file.write(str(model))
    file.close()


    if cfg.mode.checkpoints is None:
        checkpoint_epoch = 1
    else:
        load_epoch = cfg.mode.checkpoints
        checkpoint = torch.load('./checkpoints/' + experiment_name + '/checkpoint_' + str(load_epoch) + '.pth')
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_epoch = checkpoint["epoch"] + 1 

    nb_epochs = cfg.mode.epochs

    # Best model tracking
    best_val_loss = float('inf')
    best_epoch = -1

    for param_group in optimizer.param_groups:
        print("Learning rate used: {}".format(param_group['lr']))

    train_iter = 0
    for epoch in range(checkpoint_epoch, nb_epochs + 1):

        t0 = time.time()

        batch_loss = [] 
        batch_loss_val = [] 

        # ------ Training ---------
        model.train()  

        for i, (input_data, gtdepth) in enumerate(train_loader):
            # input_data can be audio (spectrogram) or RGB image depending on input_type
            input_data = input_data.to(device)
            gtdepth = gtdepth.to(device)   

            # Print statistics for first batch of first epoch
            if epoch == checkpoint_epoch and i == 0:
                print("\n" + "="*60)
                print("TRAINING DATA STATISTICS (First Batch)")
                print("="*60)
                input_type = getattr(cfg.dataset, 'input_type', 'audio')
                if input_type == 'rgb':
                    print_statistics("RGB Input", input_data)
                else:
                    print_statistics("Audio Input", input_data)
                print_statistics("Ground Truth Depth", gtdepth)
                valid_mask = get_valid_depth_mask(gtdepth)
                print_statistics("Ground Truth Depth (Valid Only)", gtdepth, valid_mask)

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            output = model(input_data)
            if isinstance(output, tuple):
                depth_pred, sh_map = output
            else:
                depth_pred = output
                sh_map = None

            # Print network output statistics for first batch of first epoch
            if epoch == checkpoint_epoch and i == 0:
                print_statistics("Network Output (Predicted Depth)", depth_pred)
                valid_mask = get_valid_depth_mask(gtdepth)
                print_statistics("Network Output (Valid Regions Only)", depth_pred, valid_mask)
                print("="*60 + "\n")
            
            # compute loss - mask out invalid depth values (0 = missing/invalid depth)
            valid_mask = get_valid_depth_mask(gtdepth)
            if valid_mask.sum() > 0:  # Only compute loss if there are valid pixels
                loss = criterion(depth_pred[valid_mask], gtdepth[valid_mask])
                # Add gradient loss if enabled
                if use_grad_loss:
                    loss = loss + grad_loss_weight * grad_criterion(depth_pred, gtdepth, valid_mask)
                # Add SH auxiliary loss if enabled
                if use_sh_aux_loss and sh_map is not None:
                    loss = loss + sh_aux_loss_weight * sh_aux_criterion(sh_map, gtdepth)
            else:
                # If no valid pixels, set loss to 0 (shouldn't happen in practice)
                loss = torch.tensor(0.0, device=device, requires_grad=True)
            batch_loss.append(loss.item())

            # optimize
            loss.backward()  # backward-pass
            optimizer.step()  # update weights
            
            train_iter += 1
            
            # Print progress periodically for large datasets
            total_batches = len(train_loader)
            if (i + 1) % max(1, total_batches // 10) == 0 or (i + 1) == total_batches:
                # Print progress every 10% or at the end
                progress = (i + 1) / total_batches * 100
                avg_loss_so_far = np.mean(batch_loss)
                print(f'Epoch {epoch} - Progress: {i+1}/{total_batches} batches ({progress:.1f}%) - Avg Loss: {avg_loss_so_far:.4f}', end='\r')
            
        wandb.log({'Train/Loss': np.mean(batch_loss), 'epoch': epoch})

        epoch_time = time.time()-t0
        # Clear the progress line and print final epoch summary
        print(' ' * 100, end='\r')  # Clear progress line
        print('Epoch {} - Loss: {:.4f} - epoch time: {:.1f}s - Total batches: {}'.format(
            epoch, np.mean(batch_loss), epoch_time, len(train_loader)))

        wandb.log({'Epoch_time': epoch_time, 'epoch': epoch})

        # ------- Validation ------------
        if cfg.mode.validation and epoch % cfg.mode.validation_iter == 0:
            model.eval() 
            errors = []
            no_depth_counts = []  # Track no-depth regions
            sample_idx = 0  # Counter for saving validation samples
            
            with torch.no_grad():
                total_val_batches = len(val_loader)
                for batch_idx, (input_val, gtdepth_val) in enumerate(val_loader):
                    input_val = input_val.to(device)
                    gtdepth_val = gtdepth_val.to(device)        

                    output_val = model(input_val)
                    if isinstance(output_val, tuple):
                        depth_pred_val = output_val[0]
                    else:
                        depth_pred_val = output_val

                    # Print validation statistics for first batch
                    if batch_idx == 0:
                        print("\n" + "="*60)
                        print(f"VALIDATION DATA STATISTICS (Epoch {epoch})")
                        print("="*60)
                        input_type = getattr(cfg.dataset, 'input_type', 'audio')
                        if input_type == 'rgb':
                            print_statistics("RGB Input", input_val)
                        else:
                            print_statistics("Audio Input", input_val)
                        print_statistics("Ground Truth Depth", gtdepth_val)
                        valid_mask_val = get_valid_depth_mask(gtdepth_val)
                        print_statistics("Ground Truth Depth (Valid Only)", gtdepth_val, valid_mask_val)
                        print_statistics("Network Output (Predicted Depth)", depth_pred_val)
                        print_statistics("Network Output (Valid Regions Only)", depth_pred_val, valid_mask_val)
                        print("="*60 + "\n")
                    
                    # Print validation progress periodically (after first batch stats)
                    if batch_idx > 0 and ((batch_idx + 1) % max(1, total_val_batches // 5) == 0 or (batch_idx + 1) == total_val_batches):
                        progress = (batch_idx + 1) / total_val_batches * 100
                        print(f'Validation Epoch {epoch} - Progress: {batch_idx+1}/{total_val_batches} batches ({progress:.1f}%)', end='\r')
                    
                    # compute validation loss - mask out invalid depth values (0 = missing/invalid depth)
                    valid_mask_val = get_valid_depth_mask(gtdepth_val)
                    if valid_mask_val.sum() > 0:  # Only compute loss if there are valid pixels
                        loss_val = criterion(depth_pred_val[valid_mask_val], gtdepth_val[valid_mask_val])
                    else:
                        # If no valid pixels, set loss to 0 (shouldn't happen in practice)
                        loss_val = torch.tensor(0.0, device=device)
                    batch_loss_val.append(loss_val.item()) 

                    for idx in range(depth_pred_val.shape[0]):
                        # Get dataset index to retrieve scene_id and step_idx
                        dataset_idx = batch_idx * batch_size + idx
                        scene_id, step_idx = val_set.samples[dataset_idx]
                        
                        # Get numpy arrays
                        gt_np = gtdepth_val[idx].cpu().numpy()
                        pred_np = depth_pred_val[idx].cpu().numpy()
                        input_np = input_val[idx].cpu().numpy()
                        
                        # Unscale if normalized
                        if cfg.dataset.depth_norm:
                            unscaledgt = gt_np * cfg.dataset.max_depth
                            unscaledpred = pred_np * cfg.dataset.max_depth
                            errors.append(compute_errors(unscaledgt, unscaledpred, scene_id=scene_id, step_idx=step_idx))
                        else:
                            unscaledgt = gt_np
                            unscaledpred = pred_np
                            errors.append(compute_errors(gt_np, pred_np, scene_id=scene_id, step_idx=step_idx))
                        
                        # Print unscaled depth statistics for first sample of first batch
                        if batch_idx == 0 and idx == 0:
                            print("UNSCALED DEPTH STATISTICS (Actual meters):")
                            print_statistics("Ground Truth Depth (Unscaled)", unscaledgt)
                            valid_mask_unscaled = get_valid_depth_mask(unscaledgt)
                            print_statistics("Ground Truth Depth (Valid Only, Unscaled)", unscaledgt, valid_mask_unscaled)
                            print_statistics("Predicted Depth (Unscaled)", unscaledpred)
                            print_statistics("Predicted Depth (Valid Regions Only, Unscaled)", unscaledpred, valid_mask_unscaled)
                            print("="*60 + "\n")
                        
                        # Count no-depth regions
                        no_depth_gt = np.sum(unscaledgt == 0)
                        no_depth_total = unscaledgt.size
                        no_depth_ratio_gt = no_depth_gt / no_depth_total if no_depth_total > 0 else 0
                        no_depth_counts.append({
                            'epoch': epoch,
                            'sample_idx': sample_idx,
                            'no_depth_pixels_gt': int(no_depth_gt),
                            'total_pixels': int(no_depth_total),
                            'no_depth_ratio_gt': float(no_depth_ratio_gt)
                        })
                        
                        # Save plots for validation samples every epoch (limit to first few samples per epoch to avoid too many files)
                        if sample_idx < getattr(cfg.mode, 'val_plot_samples', 100):
                            # scene_id and step_idx already retrieved above
                            depth_type = getattr(cfg.dataset, 'depth_type', 'pinhole')
                            input_image_type = getattr(cfg.dataset, 'input_image_type', None)
                            if input_image_type is None:
                                input_image_type = depth_type
                            
                            # Create epoch-specific directory
                            epoch_output_dir = os.path.join(val_output_dir, f'epoch_{epoch:05d}')
                            os.makedirs(epoch_output_dir, exist_ok=True)
                            
                            # Reload depth from file to ensure alignment with plot
                            # This ensures the GT depth in the plot matches the actual file for this scene_id and step_idx
                            # Load with transform applied (resize to match model output size)
                            gt_depth_from_file = load_depth_from_file(cfg, scene_id, step_idx, depth_type, apply_transform=True)
                            if gt_depth_from_file is not None:
                                # load_depth_from_file returns normalized depth if depth_norm is True
                                # Unscale to get actual depth in meters for visualization
                                if cfg.dataset.depth_norm:
                                    gt_depth_for_plot = gt_depth_from_file * cfg.dataset.max_depth
                                else:
                                    gt_depth_for_plot = gt_depth_from_file
                            else:
                                # Fallback to the depth from DataLoader if file not found
                                print(f"Warning: Could not reload depth file for scene {scene_id}, step {step_idx}. Using DataLoader depth.")
                                gt_depth_for_plot = unscaledgt
                            
                            # Save validation plots
                            save_validation_sample(epoch_output_dir, sample_idx, input_np, gt_depth_for_plot, unscaledpred,
                                                 cfg, scene_id, step_idx, depth_type, epoch, input_image_type=input_image_type)
                        
                        sample_idx += 1
                
                # Clear validation progress line
                print(' ' * 100, end='\r')
	
                mean_errors = np.array(errors).mean(0)
                avg_no_depth_ratio = np.mean([x['no_depth_ratio_gt'] for x in no_depth_counts]) if no_depth_counts else 0
                
                print('Validation - RMSE: {:.3f}, ABS_REL: {:.3f}, DELTA1: {:.3f}'.format(mean_errors[1], mean_errors[0], mean_errors[2]))
                if no_depth_counts:
                    print('Validation - Avg No-Depth Ratio: {:.3f}%'.format(avg_no_depth_ratio * 100))
                
                val_errors = {}
                val_errors['ABS_REL'], val_errors['RMSE'] = mean_errors[0], mean_errors[1]
                val_errors['DELTA1'] = mean_errors[2] 
                val_errors['DELTA2'] = mean_errors[3]
                val_errors['DELTA3'] = mean_errors[4]

                wandb.log({
                    'Val/Loss': np.mean(batch_loss_val),
                    'Val/RMSE': val_errors['RMSE'],
                    'Val/ABS_REL': val_errors['ABS_REL'],
                    'Val/DELTA1': val_errors['DELTA1'],
                    'Val/DELTA2': val_errors['DELTA2'],
                    'Val/DELTA3': val_errors['DELTA3'],
                    'Val/NoDepthRatio': avg_no_depth_ratio,
                    'epoch': epoch
                })

                # Save best model based on validation loss
                current_val_loss = np.mean(batch_loss_val)
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch
                    best_state = {
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    }
                    path_check = './checkpoints/' + experiment_name + '/'
                    os.makedirs(path_check, exist_ok=True)
                    torch.save(best_state, path_check + 'best_model.pth')
                    print(f'*** New best model saved at epoch {epoch} with val loss {best_val_loss:.6f} ***')

                # Log validation images to wandb
                wandb.log({
                    'Val/Pred_Depth': [wandb.Image(depth_pred_val[j, 0].cpu().numpy(), caption=f'pred_{j}') for j in range(min(4, depth_pred_val.shape[0]))],
                    'Val/GT_Depth': [wandb.Image(gtdepth_val[j, 0].cpu().numpy(), caption=f'gt_{j}') for j in range(min(4, gtdepth_val.shape[0]))],
                    'epoch': epoch
                })

        # ------- Log training images to wandb periodically ------------
        if epoch % cfg.mode.print_tensorboard == 0:
            wandb.log({
                'Train/Pred_Depth': [wandb.Image(depth_pred[j, 0].detach().cpu().numpy(), caption=f'pred_{j}') for j in range(min(4, depth_pred.shape[0]))],
                'Train/GT_Depth': [wandb.Image(gtdepth[j, 0].cpu().numpy(), caption=f'gt_{j}') for j in range(min(4, gtdepth.shape[0]))],
                'epoch': epoch
            })

        # ------- Save ------------
        if epoch % cfg.mode.saving_checkpoints == 0:
            print('Save network')
            state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            path_check = './checkpoints/' + experiment_name + '/'
            isExist = os.path.exists(path_check)
            if not isExist:
                os.makedirs(path_check)
            torch.save(state, './checkpoints/' + experiment_name + '/checkpoint_' + str(epoch) + '.pth')

    # Print best model summary
    if best_epoch > 0:
        print(f'\n*** Training complete. Best model: epoch {best_epoch}, val loss {best_val_loss:.6f} ***')
        print(f'*** Best model saved at: ./checkpoints/{experiment_name}/best_model.pth ***')

    wandb.finish()


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
        else:
            # Try alternative locations
            pinhole_dir_rgb = os.path.join(scene_dir, 'pinhole', 'rgb')
            if os.path.exists(pinhole_dir_rgb):
                rgb_path = os.path.join(pinhole_dir_rgb, f'pinhole_{step_idx:03d}.png')
                if os.path.exists(rgb_path):
                    return np.array(Image.open(rgb_path))
            rgb_dir = os.path.join(scene_dir, 'rgb')
            if os.path.exists(rgb_dir):
                rgb_path = os.path.join(rgb_dir, f'pinhole_{step_idx:03d}.png')
                if os.path.exists(rgb_path):
                    return np.array(Image.open(rgb_path))
    
    return None  # RGB image not found


def load_depth_from_file(cfg, scene_id, step_idx, depth_type='pinhole', apply_transform=True):
    """Load depth file directly from disk to ensure alignment with plot
    
    Args:
        cfg: Configuration object
        scene_id: Scene identifier
        step_idx: Step index
        depth_type: 'pinhole' or 'erp'
        apply_transform: If True, apply the same transform as dataset (resize, etc.)
    
    Returns:
        depth: Depth array (numpy array, unscaled if depth_norm is True)
    """
    from dataloader.utils_dataset import get_transform
    from PIL import Image
    import torch
    
    scene_dir = os.path.join(cfg.dataset.dataset_dir, scene_id)
    depth = None
    
    if depth_type == 'erp':
        erp_depth_dir = os.path.join(scene_dir, 'erp_depth')
        if os.path.exists(erp_depth_dir):
            depth_path = os.path.join(erp_depth_dir, f'erp_depth_{step_idx:03d}.npy')
            if os.path.exists(depth_path):
                depth = np.load(depth_path).astype(np.float32)
    else:  # pinhole
        pinhole_depth_dir = os.path.join(scene_dir, 'pinhole_depth')
        if os.path.exists(pinhole_depth_dir):
            depth_path = os.path.join(pinhole_depth_dir, f'pinhole_depth_{step_idx:03d}.npy')
            if os.path.exists(depth_path):
                depth = np.load(depth_path).astype(np.float32)
        else:
            pinhole_dir_depth = os.path.join(scene_dir, 'pinhole', 'depth')
            if os.path.exists(pinhole_dir_depth):
                depth_path = os.path.join(pinhole_dir_depth, f'pinhole_depth_{step_idx:03d}.npy')
                if os.path.exists(depth_path):
                    depth = np.load(depth_path).astype(np.float32)
    
    if depth is None:
        return None
    
    # Apply same preprocessing as in dataset
    if cfg.dataset.max_depth:
        depth[depth > cfg.dataset.max_depth] = cfg.dataset.max_depth
    depth[depth < 0.0] = 0.0
    
    # Apply transform (resize) if requested
    if apply_transform:
        # Resize depth to match model output size using torch
        # Convert to tensor and add channel dimension: (H, W) -> (1, H, W)
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        # Resize using nearest neighbor interpolation (same as dataset)
        import torch.nn.functional as F
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
        
        depth_tensor = F.interpolate(
            depth_tensor.unsqueeze(0),  # Add batch dimension: (1, H, W) -> (1, 1, H, W)
            size=target_size,
            mode='nearest',
            align_corners=None
        )
        # Remove batch and channel dimensions: (1, 1, H, W) -> (H, W)
        depth = depth_tensor.squeeze(0).squeeze(0).numpy()
    
    # Apply normalization if needed (after transform)
    if cfg.dataset.depth_norm and cfg.dataset.max_depth:
        depth = depth / cfg.dataset.max_depth
    
    return depth


def save_validation_sample(output_dir, sample_idx, input_np, gt_depth, pred_depth, cfg, scene_id, step_idx, depth_type, epoch, input_image_type=None):
    """
    Save validation visualization results for a single sample
    """
    input_type = getattr(cfg.dataset, 'input_type', 'audio')
    
    # Load RGB image (if not already the input)
    rgb_image = None
    if input_type != 'rgb':
        # Use input_image_type if specified, otherwise use depth_type
        if input_image_type is None:
            input_image_type = getattr(cfg.dataset, 'input_image_type', None)
            if input_image_type is None:
                input_image_type = depth_type
        rgb_image = load_rgb_image(cfg, scene_id, step_idx, input_image_type)
    
    # Save input visualization
    if input_type == 'rgb':
        # Input is RGB, save it
        rgb_path = os.path.join(output_dir, f'val_sample_{sample_idx:05d}_input_rgb.png')
        # Convert from (C, H, W) to (H, W, C) for PIL
        rgb_vis = np.transpose(input_np, (1, 2, 0))
        # Denormalize if needed (assuming values are in [0, 1])
        if rgb_vis.max() <= 1.0:
            rgb_vis = (rgb_vis * 255).astype(np.uint8)
        Image.fromarray(rgb_vis).save(rgb_path)
        rgb_image = rgb_vis  # Use for comparison plot
    else:
        # Input is audio, save spectrogram
        spec_path = os.path.join(output_dir, f'val_sample_{sample_idx:05d}_spectrogram.png')
        save_spectrogram(input_np, spec_path, scene_id=scene_id, step_idx=step_idx)
    
    # Save depth maps
    gt_path = os.path.join(output_dir, f'val_sample_{sample_idx:05d}_gt_depth.png')
    pred_path = os.path.join(output_dir, f'val_sample_{sample_idx:05d}_pred_depth.png')
    
    save_depth_map(gt_depth, gt_path, cfg.dataset.max_depth, 
                   title=f'Ground Truth Depth | Scene: {scene_id} | Step: {step_idx:03d} | Epoch: {epoch}')
    save_depth_map(pred_depth, pred_path, cfg.dataset.max_depth, 
                   title=f'Predicted Depth | Scene: {scene_id} | Step: {step_idx:03d} | Epoch: {epoch}')
    
    # Save side-by-side comparison
    comparison_path = os.path.join(output_dir, f'val_sample_{sample_idx:05d}_comparison.png')
    save_comparison(input_np, gt_depth, pred_depth, comparison_path, cfg.dataset.max_depth, depth_type, rgb_image, epoch, scene_id, step_idx, input_type)


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


def save_comparison(input_np, gt_depth, pred_depth, save_path, max_depth, depth_type, rgb_image=None, epoch=None, scene_id=None, step_idx=None, input_type='audio'):
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
    if epoch is not None:
        title += f' | Epoch: {epoch}'
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Exception happened during training")
        import traceback
        traceback.print_exc()
