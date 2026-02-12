# SoundSpaces Dataset Loader

This folder contains a complete training, testing, and evaluation framework adapted for the SoundSpaces dataset organized by scenes, following the same structure as UNetSoundOnly.

## Folder Structure

The dataloader expects the following folder structure with scene directories:

```
dataset_dir/
├── scene_id_1/
│   ├── audio_wav/          # or audio/
│   │   ├── audio_000.wav
│   │   ├── audio_001.wav
│   │   └── ...
│   └── pinhole_depth/      # or pinhole/depth/
│       ├── pinhole_depth_000.npy
│       ├── pinhole_depth_001.npy
│       └── ...
├── scene_id_2/
│   └── ...
└── ...
```

The dataset supports both folder structures:
- `audio_wav/` or `audio/` for audio files
- `pinhole_depth/` or `pinhole/depth/` for depth files

## Installation

### Quick Start with Conda (Recommended)

```bash
cd /home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset

# Create and activate conda environment
conda env create -f environment.yml
conda activate soundspaces_dataset
```

### Alternative: Manual Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions including:
- Conda environment setup
- Pip/virtual environment setup
- CUDA troubleshooting
- Dependency versions

## Configuration

1. **Update dataset path**: either set `SOUNDSPACES_DATASET_DIR` (recommended) or edit `conf/dataset/soundspaces.yaml`.

   - Using env var (server example: dataset in `/root/storage/matterport3d`):
     ```bash
     export SOUNDSPACES_DATASET_DIR=/root/storage/matterport3d
     ```

   - Using YAML (fallback):
   ```yaml
   dataset_dir: /home/rvi-lab/workspace/sound-spaces/dataset
   ```

2. **Training configuration**: Edit `conf/mode/train.yaml` to set:
   - `experiment_name`: Name for your experiment
   - `batch_size`: Batch size for training
   - `epochs`: Number of training epochs
   - `learning_rate`: Learning rate
   - `optimizer`: Optimizer (Adam, AdamW, or SGD)

3. **Model configuration**: Edit `conf/model/unet_baseline.yaml` to set:
   - `generator`: `unet_128` or `unet_256` (depending on image size)

## Dataset Splitting

**Scene-based splitting**: The dataset splits **scenes** (not individual samples) into train/val/test with a **6:2:2 ratio**:
- **Train**: 60% of scenes
- **Val**: 20% of scenes
- **Test**: 20% of scenes

This ensures that samples from the same scene stay in the same split, preventing data leakage. The split is deterministic (seed=42) for reproducibility.

All samples within each scene are included in that split. For example, if a scene has 400 samples and is assigned to train, all 400 samples are in the training set.

## Usage

### Training

Train the model:
```bash
python train.py mode.experiment_name=my_experiment
```

If you're on a server and your dataset is under `/root/storage/matterport3d`, you can use the provided server wrappers:
```bash
bash train_erp_sh_server.sh
```

Or update the config files and run:
```bash
python train.py
```

The training script will:
- Automatically split scenes into train (60%), val (20%), and test (20%)
- Load all samples from each scene in the respective split
- Save checkpoints in `./checkpoints/{experiment_name}/`
- Save logs to `./logs/{experiment_name}/`
- Display training progress in TensorBoard

### Testing

Test the model on test or validation set:
```bash
python test.py mode.experiment_name=my_experiment mode.checkpoints=50 mode.eval_on=test
```

Server wrapper (evaluates the v2 experiments; default uses `EVAL_EPOCH=best`):
```bash
bash eval_erp_sh_server.sh
```

Results will be saved in `./eval/{dataset_name}/{eval_on}/`

### Evaluation

Run detailed evaluation:
```bash
python eval.py mode.experiment_name=my_experiment mode.checkpoints=50 mode.eval_split=test
```

This provides:
- Mean and standard deviation for all metrics
- Per-sample error statistics
- Detailed results saved as pickle file

## Dataset Features

- **Scene-based splitting**: Splits by scenes (6:2:2) to prevent data leakage
- **Flexible folder structure**: Supports multiple folder naming conventions
- **Audio formats**: Supports `spectrogram`, `mel_spectrogram`, or `waveform`
- **Depth normalization**: Optional depth normalization using `max_depth`
- **Image preprocessing**: Optional resizing to specified size
- **Automatic validation**: Verifies file existence and filters invalid samples

## Differences from BatvisionV2

- **Scene-based organization**: Organizes data by scene directories instead of flat CSV files
- **Scene-level splitting**: Splits entire scenes rather than individual samples
- **Multiple folder structures**: Handles both `audio_wav/` and `audio/`, `pinhole_depth/` and `pinhole/depth/`
- **Automatic sample discovery**: Finds all samples within each scene automatically

## Output Structure

```
SoundSpacesDataset/
├── checkpoints/
│   └── {experiment_name}/
│       └── checkpoint_{epoch}.pth
├── logs/
│   └── {experiment_name}/
│       ├── events.out.tfevents...
│       └── architecture.txt
└── eval/
    └── soundspaces/
        ├── test/
        │   └── stats_on_*.pkl
        └── val/
            └── stats_on_*.pkl
```

## Evaluation Metrics

The framework computes the following metrics:
- **ABS_REL**: Absolute relative error
- **RMSE**: Root mean square error
- **Delta1/2/3**: Accuracy with thresholds 1.25, 1.25², 1.25³
- **Log10**: Log10 error
- **MAE**: Mean absolute error

## Example

If you have 100 scenes in your dataset:
- **Train**: 60 scenes (all samples from these scenes)
- **Val**: 20 scenes (all samples from these scenes)
- **Test**: 20 scenes (all samples from these scenes)

Each scene may have different numbers of samples, and all samples from a scene will be in the same split.
