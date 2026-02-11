"""
Script to plot training results, evaluation metrics, and visualize predictions.
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import argparse
import json

def plot_training_curves(log_dir, save_dir=None):
    """Plot training curves from TensorBoard logs or checkpoint files."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    if save_dir is None:
        save_dir = os.path.join(log_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    
    # Try to load TensorBoard logs
    ea = EventAccumulator(log_dir)
    ea.Reload()
    
    scalars = ea.Tags()['scalars']
    
    # Plot training loss
    if 'Train/Loss' in scalars:
        train_loss = ea.Scalars('Train/Loss')
        epochs = [s.step for s in train_loss]
        values = [s.value for s in train_loss]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, label='Train Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'train_loss.png'), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(save_dir, 'train_loss.png')}")
    
    # Plot validation loss
    if 'Val/Loss' in scalars:
        val_loss = ea.Scalars('Val/Loss')
        epochs = [s.step for s in val_loss]
        values = [s.value for s in val_loss]
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, values, label='Val Loss', linewidth=2, color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'val_loss.png'), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(save_dir, 'val_loss.png')}")
    
    # Plot validation metrics
    if 'Val/RMSE' in scalars:
        rmse = ea.Scalars('Val/RMSE')
        abs_rel = ea.Scalars('Val/ABS_REL') if 'Val/ABS_REL' in scalars else None
        delta1 = ea.Scalars('Val/DELTA1') if 'Val/DELTA1' in scalars else None
        
        epochs = [s.step for s in rmse]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        axes[0].plot(epochs, [s.value for s in rmse], label='RMSE', linewidth=2, color='red')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Validation RMSE')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if abs_rel:
            axes[1].plot(epochs, [s.value for s in abs_rel], label='ABS_REL', linewidth=2, color='blue')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('ABS_REL')
            axes[1].set_title('Validation ABS_REL')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        if delta1:
            axes[2].plot(epochs, [s.value for s in delta1], label='Delta1', linewidth=2, color='green')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Delta1')
            axes[2].set_title('Validation Delta1')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'val_metrics.png'), dpi=150)
        plt.close()
        print(f"Saved: {os.path.join(save_dir, 'val_metrics.png')}")

def plot_evaluation_results(eval_file, save_dir=None):
    """Plot evaluation results from pickle file."""
    if not os.path.exists(eval_file):
        print(f"Evaluation file not found: {eval_file}")
        return
    
    if save_dir is None:
        save_dir = os.path.dirname(eval_file)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load evaluation results
    results = pd.read_pickle(eval_file)
    
    # Extract metrics
    metrics = {
        'RMSE': results['rmse'] if 'rmse' in results else None,
        'ABS_REL': results['abs_rel'] if 'abs_rel' in results else None,
        'Delta1': results['delta1'] if 'delta1' in results else None,
        'Delta2': results['delta2'] if 'delta2' in results else None,
        'Delta3': results['delta3'] if 'delta3' in results else None,
        'Log10': results['log10'] if 'log10' in results else None,
        'MAE': results['mae'] if 'mae' in results else None,
    }
    
    # Filter out None values
    metrics = {k: v for k, v in metrics.items() if v is not None}
    
    if not metrics:
        print("No metrics found in evaluation file")
        return
    
    # Plot distribution of metrics
    n_metrics = len(metrics)
    cols = min(4, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (name, values) in enumerate(metrics.items()):
        ax = axes[idx]
        ax.hist(values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(values):.4f}')
        ax.set_xlabel(name)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'eval_metrics_distribution.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Evaluation Metrics Summary:")
    print("="*60)
    for name, values in metrics.items():
        print(f"{name:15s}: Mean={np.mean(values):.4f}, Std={np.std(values):.4f}, "
              f"Min={np.min(values):.4f}, Max={np.max(values):.4f}")
    print("="*60)

def plot_predictions(eval_file, num_samples=4, save_dir=None):
    """Plot sample predictions vs ground truth."""
    if not os.path.exists(eval_file):
        print(f"Evaluation file not found: {eval_file}")
        return
    
    if save_dir is None:
        save_dir = os.path.dirname(eval_file)
    os.makedirs(save_dir, exist_ok=True)
    
    # Load evaluation results
    results = pd.read_pickle(eval_file)
    
    if 'gt_images' not in results or 'pred_imgs' not in results:
        print("Prediction images not found in evaluation file")
        return
    
    gt_images = results['gt_images']
    pred_images = results['pred_imgs']
    
    num_samples = min(num_samples, len(gt_images))
    
    # Select random samples
    indices = np.random.choice(len(gt_images), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(5*num_samples, 10))
    
    for i, idx in enumerate(indices):
        # Ground truth
        gt = gt_images[idx]
        if len(gt.shape) == 3:
            gt = gt[0]  # Take first channel if multi-channel
        
        axes[0, i].imshow(gt, cmap='jet')
        axes[0, i].set_title(f'GT Sample {idx}')
        axes[0, i].axis('off')
        
        # Prediction
        pred = pred_images[idx]
        if len(pred.shape) == 3:
            pred = pred[0]  # Take first channel if multi-channel
        
        axes[1, i].imshow(pred, cmap='jet')
        axes[1, i].set_title(f'Pred Sample {idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'sample_predictions.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Plot training and evaluation results')
    parser.add_argument('--log_dir', type=str, help='Path to TensorBoard log directory')
    parser.add_argument('--eval_file', type=str, help='Path to evaluation pickle file')
    parser.add_argument('--checkpoint_dir', type=str, help='Path to checkpoint directory (to find logs)')
    parser.add_argument('--experiment_name', type=str, help='Experiment name (to find logs and eval files)')
    parser.add_argument('--eval_dir', type=str, default='./eval', help='Directory containing evaluation files')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of prediction samples to plot')
    
    args = parser.parse_args()
    
    # Determine paths
    if args.checkpoint_dir or args.experiment_name:
        if args.checkpoint_dir:
            base_dir = args.checkpoint_dir
        else:
            base_dir = './logs'
        
        if args.experiment_name:
            log_dir = os.path.join(base_dir, args.experiment_name)
            if os.path.exists(log_dir):
                print(f"Plotting training curves from: {log_dir}")
                plot_training_curves(log_dir)
    
    if args.log_dir:
        print(f"Plotting training curves from: {args.log_dir}")
        plot_training_curves(args.log_dir)
    
    if args.eval_file:
        print(f"Plotting evaluation results from: {args.eval_file}")
        plot_evaluation_results(args.eval_file)
        plot_predictions(args.eval_file, num_samples=args.num_samples)
    
    if args.experiment_name and args.eval_dir:
        # Try to find evaluation files
        eval_base = os.path.join(args.eval_dir, 'soundspaces')
        if os.path.exists(eval_base):
            for split in ['test', 'val']:
                split_dir = os.path.join(eval_base, split)
                if os.path.exists(split_dir):
                    pkl_files = [f for f in os.listdir(split_dir) if f.endswith('.pkl') and args.experiment_name in f]
                    for pkl_file in pkl_files:
                        eval_file = os.path.join(split_dir, pkl_file)
                        print(f"Plotting evaluation results from: {eval_file}")
                        plot_evaluation_results(eval_file)
                        plot_predictions(eval_file, num_samples=args.num_samples)

if __name__ == '__main__':
    main()
