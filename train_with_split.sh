#!/bin/bash
# Training script for simple UNet to predict depth from audio input
# Uses sound-spaces/dataset directory with train/val/test split by ratio (8:1:1)

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==========================================
# (1) Audio -> Pinhole depth
# ==========================================
echo "=========================================="
echo "Training: Audio -> Pinhole depth (unet_256)"
echo "=========================================="

python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_pinhole_unet \
    mode.batch_size=16 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    mode.criterion=L1 \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    mode.print_tensorboard=50 \
    mode.shuffle=True \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
    dataset.use_same_samples_all_splits=False \
    dataset.single_scene_test_mode=False \
    dataset.input_type=audio \
    dataset.audio_format=spectrogram \
    dataset.preprocess=resize \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    dataset.min_depth=0.01 \
    dataset.max_depth=10.0 \
    dataset.depth_type=pinhole \
    dataset.use_augmentation=False \
    model.generator=unet_256

echo "Audio -> Pinhole depth training completed!"

# ==========================================
# (2) Audio -> ERP depth
# ==========================================
echo "=========================================="
echo "Training: Audio -> ERP depth (unet_256)"
echo "=========================================="

python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_unet \
    mode.batch_size=16 \
    mode.epochs=100 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    mode.criterion=L1 \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    mode.print_tensorboard=50 \
    mode.shuffle=True \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
    dataset.use_same_samples_all_splits=False \
    dataset.single_scene_test_mode=False \
    dataset.input_type=audio \
    dataset.audio_format=spectrogram \
    dataset.preprocess=resize \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    dataset.min_depth=0.01 \
    dataset.max_depth=10.0 \
    dataset.depth_type=erp \
    dataset.use_augmentation=False \
    model.generator=unet_256

echo "Audio -> ERP depth training completed!"
echo "=========================================="
echo "All training completed!"
echo "=========================================="
