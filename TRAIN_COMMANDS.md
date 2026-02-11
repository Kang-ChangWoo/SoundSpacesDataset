# Training Commands and Plotting Results

## Training Command

### Basic Training

```bash
cd /home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset

python train.py \
    mode.experiment_name=my_experiment \
    mode.batch_size=16 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```

### Training with Custom Config

1. Edit `conf/mode/train.yaml`:
```yaml
mode: train
experiment_name: my_experiment
batch_size: 16
epochs: 100
learning_rate: 0.001
optimizer: AdamW
```

2. Edit `conf/dataset/soundspaces.yaml`:
```yaml
dataset_dir: /home/rvi-lab/workspace/sound-spaces/dataset
```

3. Run:
```bash
python train.py
```

### Training with Checkpoint Resume

```bash
python train.py \
    mode.experiment_name=my_experiment \
    mode.checkpoints=50
```

## Testing Command

```bash
python test.py \
    mode.experiment_name=my_experiment \
    mode.checkpoints=50 \
    mode.eval_on=test \
    mode.batch_size=16
```

## Evaluation Command

```bash
python eval.py \
    mode.experiment_name=my_experiment \
    mode.checkpoints=50 \
    mode.eval_split=test
```

## Plotting Results

### Plot Training Curves

From TensorBoard logs:
```bash
python plot_results.py --log_dir ./logs/my_experiment
```

Or specify experiment name:
```bash
python plot_results.py --experiment_name my_experiment
```

### Plot Evaluation Results

```bash
python plot_results.py \
    --eval_file ./eval/soundspaces/test/stats_on_soundspaces_test_set_my_experiment_epoch_50.pkl
```

### Plot Everything for an Experiment

```bash
python plot_results.py \
    --experiment_name my_experiment \
    --eval_dir ./eval \
    --num_samples 8
```

### View TensorBoard

```bash
tensorboard --logdir=./logs/my_experiment
```

Then open http://localhost:6006 in your browser.

## Output Structure After Training

```
SoundSpacesDataset/
├── logs/
│   └── my_experiment/
│       ├── events.out.tfevents...  # TensorBoard logs
│       ├── architecture.txt        # Model architecture
│       └── plots/                  # Generated plots (if using plot_results.py)
│           ├── train_loss.png
│           ├── val_loss.png
│           └── val_metrics.png
├── checkpoints/
│   └── my_experiment/
│       ├── checkpoint_10.pth
│       ├── checkpoint_20.pth
│       └── ...
└── eval/
    └── soundspaces/
        ├── test/
        │   └── stats_on_soundspaces_test_set_my_experiment_epoch_50.pkl
        │   └── eval_soundspaces_test_my_experiment_epoch_50.pkl
        │   └── eval_metrics_distribution.png (after plotting)
        │   └── sample_predictions.png (after plotting)
        └── val/
            └── ...
```

## Quick Start Example

```bash
# 1. Train model
python train.py mode.experiment_name=test_run dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset

# 2. Test model (after training completes or manually stop and specify epoch)
python test.py mode.experiment_name=test_run mode.checkpoints=50 mode.eval_on=test

# 3. Plot results
python plot_results.py --experiment_name test_run --eval_dir ./eval

# 4. View TensorBoard (in another terminal)
tensorboard --logdir=./logs/test_run
```

## Notes

- The dataset automatically splits scenes 6:2:2 (train/val/test)
- Checkpoints are saved every `saving_checkpoints` epochs (default: 10)
- Validation is performed every `validation_iter` epochs (default: 2)
- All plots will be saved in the respective directories
