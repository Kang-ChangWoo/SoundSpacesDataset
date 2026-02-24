"""
Evaluation script - similar to test.py but can be used for detailed evaluation
"""
from dataloader.SoundSpaces_Dataset import SoundSpacesDataset

from models.utils_models import *
from models.unetbaseline_model import *
from utils_criterion import compute_errors, get_valid_depth_mask, BerHuLoss

import os 
import numpy as np 
import pandas as pd
import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")  
def main(cfg):
    working_dir = os.getcwd()
    print(f"The current working directory is {working_dir}")
    
    # ------------ GPU config ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_GPU = torch.cuda.device_count()
    print("{} {} device is used".format(n_GPU, device))

    batch_size = cfg.mode.batch_size
    
    # ------------ Create dataset -----------
    if cfg.dataset.name == 'soundspaces':
        # Can evaluate on train, val, or test split
        split = getattr(cfg.mode, 'eval_split', 'test')
        eval_set = SoundSpacesDataset(cfg, split=split)
    else:
        raise Exception('Eval can be done only on soundspaces dataset')

    print(f'Evaluation Dataset of {len(eval_set)} instances (split: {split})')
    eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False, num_workers=cfg.mode.num_threads) 

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
        raise AttributeError('In eval mode, a checkpoint needs to be loaded.')
    else:
        load_epoch = cfg.mode.checkpoints
        checkpoint_path = os.path.join('./checkpoints', cfg.mode.experiment_name, 'checkpoint_' + str(load_epoch) + '.pth')
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        print(f'Loaded checkpoint from epoch: {load_epoch}')

    # ------ Eval ---------
    model.eval()

    all_errors = []
    all_losses = []

    print('Starting evaluation...')
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
            # compute eval loss - mask out invalid depth values (0 = missing/invalid depth)
            valid_mask = get_valid_depth_mask(depthgt)
            if valid_mask.sum() > 0:  # Only compute loss if there are valid pixels
                loss_test = criterion(depth_pred[valid_mask], depthgt[valid_mask])
            else:
                # If no valid pixels, set loss to 0 (shouldn't happen in practice)
                loss_test = torch.tensor(0.0, device=device)
            all_losses.append(loss_test.cpu().item())

            for idx in range(depth_pred.shape[0]):
                # Get dataset index and scene info
                dataset_idx = batch_idx * batch_size + idx
                scene_id, step_idx = eval_set.samples[dataset_idx]
                
                if cfg.dataset.depth_norm:
                    unscaledgt = depthgt[idx].detach().cpu().numpy() * cfg.dataset.max_depth
                    unscaledpred = depth_pred[idx].detach().cpu().numpy() * cfg.dataset.max_depth
                    errors = compute_errors(unscaledgt, unscaledpred, scene_id=scene_id, step_idx=step_idx)
                else:
                    errors = compute_errors(depthgt[idx].cpu().numpy(), depth_pred[idx].cpu().numpy(), scene_id=scene_id, step_idx=step_idx)
                
                all_errors.append(errors)

            if (batch_idx + 1) % 10 == 0:
                print(f'Processed {batch_idx + 1}/{len(eval_loader)} batches')

    # Compute statistics
    mean_errors = np.array(all_errors).mean(0)
    std_errors = np.array(all_errors).std(0)
    
    print('=' * 60)
    print(f'Evaluation Results ({split} split):')
    print('=' * 60)
    print('Metric        | Mean      | Std')
    print('-' * 60)
    print('ABS_REL       | {:.4f}    | {:.4f}'.format(mean_errors[0], std_errors[0]))
    print('RMSE          | {:.4f}    | {:.4f}'.format(mean_errors[1], std_errors[1]))
    print('Delta1        | {:.4f}    | {:.4f}'.format(mean_errors[2], std_errors[2]))
    print('Delta2        | {:.4f}    | {:.4f}'.format(mean_errors[3], std_errors[3]))
    print('Delta3        | {:.4f}    | {:.4f}'.format(mean_errors[4], std_errors[4]))
    print('Log10         | {:.4f}    | {:.4f}'.format(mean_errors[5], std_errors[5]))
    print('MAE           | {:.4f}    | {:.4f}'.format(mean_errors[6], std_errors[6]))
    print('Loss          | {:.4f}    | {:.4f}'.format(np.mean(all_losses), np.std(all_losses)))
    print('=' * 60)

    # Save detailed results
    results_dict = {
        'metrics': {
            'ABS_REL': {'mean': mean_errors[0], 'std': std_errors[0]},
            'RMSE': {'mean': mean_errors[1], 'std': std_errors[1]},
            'Delta1': {'mean': mean_errors[2], 'std': std_errors[2]},
            'Delta2': {'mean': mean_errors[3], 'std': std_errors[3]},
            'Delta3': {'mean': mean_errors[4], 'std': std_errors[4]},
            'Log10': {'mean': mean_errors[5], 'std': std_errors[5]},
            'MAE': {'mean': mean_errors[6], 'std': std_errors[6]},
            'Loss': {'mean': np.mean(all_losses), 'std': np.std(all_losses)}
        },
        'per_sample_errors': all_errors,
        'per_sample_losses': all_losses,
        'config': dict(cfg),
        'checkpoint_epoch': load_epoch,
        'split': split
    }

    # Save results
    results_dir = os.path.join(cfg.mode.stat_dir, cfg.dataset.name, split)
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(results_dir, 
                                f'eval_{cfg.dataset.name}_{split}_{cfg.mode.experiment_name}_epoch_{load_epoch}.pkl')
    pd.to_pickle(results_dict, results_path)
    print(f'Results saved to: {results_path}')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Exception happened during evaluation")
        import traceback
        traceback.print_exc()
