# Quick Start - Training Command

## Fixed Command

The config now defaults to `train` mode. Use this command:

```bash
python train.py \
    mode.experiment_name=my_experiment \
    mode.batch_size=16 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```

## Alternative: If you still get struct errors

Use the `+` prefix or override mode explicitly:

```bash
# Option 1: Explicitly set mode=train first
python train.py \
    mode=train \
    mode.experiment_name=my_experiment \
    mode.batch_size=16 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset

# Option 2: Use + to force add (if needed)
python train.py \
    mode=train \
    mode.experiment_name=my_experiment \
    mode.batch_size=16 \
    +mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```

## Simplest Approach

Just edit `conf/mode/train.yaml` with your settings, then run:

```bash
python train.py dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```
