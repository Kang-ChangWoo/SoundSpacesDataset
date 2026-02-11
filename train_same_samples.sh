#!/bin/bash
# Training script using same scene and same samples for all splits (train/val/test)
# This is useful for verification - all splits use identical data

cd "$(dirname "$0")"

echo "Starting training with SAME samples for all splits (verification mode)..."
echo "=============================================="

# Use same scene and same samples for train/val/test
python train.py \
    dataset.use_same_samples_all_splits=True \
    dataset.single_scene_name=null \
    mode.experiment_name=same_samples_verification \
    mode.batch_size=8 \
    mode.epochs=10 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW

# Or specify a specific scene:
# python train.py \
#     dataset.use_same_samples_all_splits=True \
#     dataset.single_scene_name=your_scene_name \
#     mode.experiment_name=same_samples_verification \
#     mode.batch_size=8 \
#     mode.epochs=10

echo "Training completed!"
