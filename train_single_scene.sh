#!/bin/bash
# Training script for single scene test mode
# This uses one scene and splits samples 8:1:1 (train:val:test)

cd "$(dirname "$0")"

echo "Starting training in Single Scene Test Mode..."
echo "=============================================="

# Example 1: Use first available scene (auto-select)
python train.py \
    dataset.single_scene_test_mode=True \
    mode.experiment_name=single_scene_test \
    mode.batch_size=8 \
    mode.epochs=10 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW

# Example 2: Use specific scene (uncomment and modify scene name)
# python train.py \
#     dataset.single_scene_test_mode=True \
#     dataset.single_scene_name=your_scene_name \
#     mode.experiment_name=single_scene_test \
#     mode.batch_size=8 \
#     mode.epochs=10 \
#     mode.learning_rate=0.001 \
#     mode.optimizer=AdamW

echo "Training completed!"
