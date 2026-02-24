#!/bin/bash
# Default training script for SoundSpacesDataset
# This script runs training with default configuration

cd "$(dirname "$0")"

echo "Starting training with default configuration..."
echo "=============================================="

python train.py \
    mode.experiment_name=soundspaces_train \
    mode.batch_size=16 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
    dataset.audio_format=spectrogram \
    dataset.depth_norm=True \
    dataset.max_depth=30.0 \
    dataset.depth_type=pinhole

echo "Training completed!"
