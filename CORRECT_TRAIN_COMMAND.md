# Correct Training Command

## Issue
Hydra struct mode prevents overriding keys that don't exist. Use proper syntax.

## Correct Command

```bash
python train.py \
    mode=train \
    mode.experiment_name=my_experiment \
    mode.batch_size=16 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```

## Key Points

1. **First override the mode**: `mode=train` (not `mode.mode=train`)
2. **Then override other parameters**: All other `mode.*` parameters will work

## Alternative: Edit Config File

Or just edit `conf/mode/train.yaml` and run:
```bash
python train.py mode=train dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset
```

## If Still Getting Errors

Use `+` prefix to add new keys (if needed):
```bash
python train.py \
    mode=train \
    +mode.epochs=100
```
