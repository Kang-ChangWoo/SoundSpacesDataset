#!/bin/bash
# Training script using RGB images as input instead of audio

cd "$(dirname "$0")"

echo "Starting training with RGB input (single scene for quick test)..."
echo "=============================================="

#    dataset.single_scene_test_mode=True \
#    dataset.single_scene_name=jtcxE69GiFV \
# Train with RGB images as input - using single scene for quick testing
python train.py \
    dataset.input_type=rgb \
    mode.experiment_name=rgb_input_test \
    mode.batch_size=8 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    dataset.depth_type=pinhole
    experiment_name=rgb_input_test

echo "Training completed!"
