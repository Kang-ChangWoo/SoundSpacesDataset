# How to Test a Trained Checkpoint

## Quick Start

To test your checkpoint at:
```
/home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset/checkpoints/unet_256_soundspaces_BS16_Lr0.001_AdamW_my_experiment/checkpoint_50.pth
```

### Option 1: Using Full Experiment Name (Easiest)

```bash
cd /home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset

python test.py \
    mode=test \
    +mode.experiment_name_full=unet_256_soundspaces_BS16_Lr0.001_AdamW_my_experiment \
    mode.checkpoints=50 \
    mode.eval_on=test \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```

**Note:** Use `+mode.experiment_name_full` (with `+` prefix) since this field doesn't exist in the config by default.

### Option 2: Using Component Parameters

```bash
cd /home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset

python test.py \
    mode=test \
    mode.experiment_name=my_experiment \
    mode.checkpoints=50 \
    mode.batch_size=16 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    mode.eval_on=test \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```

## Parameters Explained

- **mode.experiment_name_full**: Full checkpoint folder name (use this if you know it)
- **mode.experiment_name**: Base experiment name (will be combined with model/dataset params)
- **mode.checkpoints**: Epoch number of the checkpoint (50 in your case)
- **mode.eval_on**: Which split to evaluate on (`test` or `val`)
- **dataset.dataset_dir**: Path to your dataset

## Output Locations

After running the test, results will be saved to:

1. **Visualization Images**: 
   ```
   ./outputs/{experiment_name}/epoch_50/{eval_on}/
   ```
   - Input RGB images
   - Input spectrograms
   - Ground truth depth maps
   - Predicted depth maps
   - Comparison images with error maps

2. **Statistics**:
   ```
   ./eval/{dataset_name}/{eval_on}/stats_on_*.pkl
   ```

## Testing with Different Configurations

### Test on Validation Set
```bash
python test.py \
    mode=test \
    +mode.experiment_name_full=unet_256_soundspaces_BS16_Lr0.001_AdamW_my_experiment \
    mode.checkpoints=50 \
    mode.eval_on=val
```

### Test with ERP Depth
```bash
python test.py \
    mode=test \
    +mode.experiment_name_full=unet_256_soundspaces_BS16_Lr0.001_AdamW_my_experiment \
    mode.checkpoints=50 \
    dataset.depth_type=erp
```

### Test Different Epoch
```bash
python test.py \
    mode=test \
    +mode.experiment_name_full=unet_256_soundspaces_BS16_Lr0.001_AdamW_my_experiment \
    mode.checkpoints=40  # Change to desired epoch
```

## What Gets Saved

For each test sample, the following files are saved:
- `sample_XXXXX_input_rgb.png` - Input RGB image (if available)
- `sample_XXXXX_spectrogram.png` - Input spectrogram
- `sample_XXXXX_gt_depth.png` - Ground truth depth map
- `sample_XXXXX_pred_depth.png` - Predicted depth map
- `sample_XXXXX_comparison.png` - Side-by-side comparison with error map

Statistics include:
- Loss, RMSE, ABS_REL, Delta1/2/3, Log10, MAE
- No-depth region counts and ratios

## Troubleshooting

If you get an error about checkpoint not found:
1. Check that the checkpoint path exists
2. Verify the experiment_name_full matches the folder name exactly
3. Make sure the epoch number (checkpoints) matches an existing checkpoint file
