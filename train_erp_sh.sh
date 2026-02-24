#!/bin/bash
# Training script for UNet + Spherical Harmonics (SH) on ERP depth estimation
# Input: Audio spectrogram (2-channel binaural)
# Output: ERP depth map (256x512)
#
# How SH works in this pipeline:
#   1. Encoder processes audio spectrogram -> bottleneck (encoded audio features)
#   2. SH branch: bottleneck -> 3-layer MLP with BN -> N_sh SH coefficients
#   3. SH map = einsum(coeffs, fixed SH basis) -> (B, N_sh, H, W) multi-channel
#   4. SH map fused into decoder via multi-head cross-attention at two levels
#      - Feature pixels (Q) attend to N_sh SH tokens (KV)
#      - Efficient: attention shape (B, heads, HW, N_sh), N_sh << HW
#
# Changes from v1 (report fixes applied):
#   - sh_degree 5 -> 10 (55 coefficients for finer geometry)
#   - Cross-attention fusion instead of additive gating
#   - Multi-channel SH map (B, N_sh, H, W) instead of (B, 1, H, W)
#   - SHCoeffExtractor: 3-layer MLP with BatchNorm
#   - lr 0.001 -> 0.0003 (stabilize SH branch training)
#   - sh_aux_loss_weight 0.1 -> 0.2
#   - Fair comparison: both SH and baseline use BerHu + GradientLoss
#   - epochs 100 -> 150
#
# Adapted from HUSH (https://github.com/vision3d-lab/HUSH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Dataset path: server vs local
if [ -d "/root/storage/matterport3d" ]; then
    DATASET_DIR="/root/storage/matterport3d"
    echo "[ENV] Server detected -> dataset: $DATASET_DIR"
else
    DATASET_DIR="/home/rvi-lab/workspace/sound-spaces/dataset"
    echo "[ENV] Local detected -> dataset: $DATASET_DIR"
fi

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
if [ -d "/root/storage/matterport3d" ]; then
    conda activate ss
else
    conda activate soundspaces_dataset
fi

# ==========================================
# (1) UNet+SH -> ERP depth (BerHu + Gradient + SH aux)
# ==========================================
echo "=========================================="
echo "Training UNet+SH with Audio Spectrogram input"
echo "Output: ERP depth map"
echo "Loss: BerHu + GradientLoss + SH Auxiliary"
echo "SH: degree=10, 55 coeffs, cross-attention fusion"
echo "LR: 0.0003"
echo "=========================================="

python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_20260211 \
    mode.batch_size=16 \
    mode.epochs=150 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
    mode.criterion=BerHu \
    mode.use_grad_loss=True \
    mode.grad_loss_weight=0.5 \
    mode.use_sh_aux_loss=True \
    mode.sh_aux_loss_weight=0.2 \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    mode.print_tensorboard=50 \
    mode.shuffle=True \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.dataset_dir=${DATASET_DIR} \
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
    dataset.use_augmentation=True \
    model.generator=unet_256_sh \
    model.sh_degree=10

echo "=========================================="
echo "UNet+SH training completed!"
echo "=========================================="

# ==========================================
# (2) Baseline UNet -> Pinhole depth (BerHu + Gradient, fair comparison)
# ==========================================
echo ""
echo "=========================================="
echo "Training: Baseline UNet Audio -> Pinhole depth"
echo "Loss: BerHu + GradientLoss (same as SH for fair comparison)"
echo "=========================================="

python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_pinhole_baseline_v2_20260211 \
    mode.batch_size=16 \
    mode.epochs=150 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
    mode.criterion=BerHu \
    mode.use_grad_loss=True \
    mode.grad_loss_weight=0.5 \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    mode.print_tensorboard=50 \
    mode.shuffle=True \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.dataset_dir=${DATASET_DIR} \
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
    dataset.use_augmentation=True \
    model.generator=unet_256

echo "Baseline Pinhole training completed!"

# ==========================================
# (3) Baseline UNet -> ERP depth (BerHu + Gradient, fair comparison)
# ==========================================
echo ""
echo "=========================================="
echo "Training: Baseline UNet Audio -> ERP depth"
echo "Loss: BerHu + GradientLoss (same as SH for fair comparison)"
echo "=========================================="

python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_baseline_v2_20260211 \
    mode.batch_size=16 \
    mode.epochs=150 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
    mode.criterion=BerHu \
    mode.use_grad_loss=True \
    mode.grad_loss_weight=0.5 \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    mode.print_tensorboard=50 \
    mode.shuffle=True \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.dataset_dir=${DATASET_DIR} \
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
    dataset.use_augmentation=True \
    model.generator=unet_256

echo "Baseline ERP training completed!"

# ==========================================
# (4) Oracle: Pinhole RGB -> Pinhole depth (upper bound)
# ==========================================
echo ""
echo "=========================================="
echo "Training: Oracle Pinhole RGB -> Pinhole depth"
echo "Input: Pinhole RGB image (3 channels)"
echo "Output: Pinhole depth map"
echo "Loss: BerHu + GradientLoss"
echo "=========================================="

python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_oracle_pinhole_v2_20260211 \
    mode.batch_size=16 \
    mode.epochs=150 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
    mode.criterion=BerHu \
    mode.use_grad_loss=True \
    mode.grad_loss_weight=0.5 \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    mode.print_tensorboard=50 \
    mode.shuffle=True \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.dataset_dir=${DATASET_DIR} \
    dataset.use_same_samples_all_splits=False \
    dataset.single_scene_test_mode=False \
    dataset.input_type=rgb \
    dataset.input_image_type=pinhole \
    dataset.preprocess=resize \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    dataset.min_depth=0.01 \
    dataset.max_depth=10.0 \
    dataset.depth_type=pinhole \
    dataset.use_augmentation=True \
    model.generator=oracle_pinhole_256

echo "Oracle Pinhole training completed!"

# ==========================================
# (5) Oracle: ERP RGB -> ERP depth (upper bound)
# ==========================================
echo ""
echo "=========================================="
echo "Training: Oracle ERP RGB -> ERP depth"
echo "Input: Panoramic (ERP) RGB image (3 channels)"
echo "Output: ERP depth map"
echo "Loss: BerHu + GradientLoss"
echo "=========================================="

python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_oracle_erp_v2_20260211 \
    mode.batch_size=16 \
    mode.epochs=150 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
    mode.criterion=BerHu \
    mode.use_grad_loss=True \
    mode.grad_loss_weight=0.5 \
    mode.validation=True \
    mode.validation_iter=2 \
    mode.saving_checkpoints=10 \
    mode.print_tensorboard=50 \
    mode.shuffle=True \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.dataset_dir=${DATASET_DIR} \
    dataset.use_same_samples_all_splits=False \
    dataset.single_scene_test_mode=False \
    dataset.input_type=rgb \
    dataset.input_image_type=erp \
    dataset.preprocess=resize \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    dataset.min_depth=0.01 \
    dataset.max_depth=10.0 \
    dataset.depth_type=erp \
    dataset.use_augmentation=True \
    model.generator=oracle_erp_256

echo "Oracle ERP training completed!"
echo "=========================================="
echo "All training completed!"
echo "=========================================="
