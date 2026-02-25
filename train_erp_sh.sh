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
# All 5 experiments run in parallel on separate GPUs (GPU 0-4)
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

# Disable Python output buffering so logs appear in real-time
export PYTHONUNBUFFERED=1

# Log directory
LOG_DIR="${SCRIPT_DIR}/server_logs/$(date +%Y%m%d)_train"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Launching 5 experiments in parallel (GPU 0-4)"
echo "Logs: ${LOG_DIR}/"
echo "=========================================="

# ==========================================
# (1) UNet+SH -> ERP depth (BerHu + Gradient + SH aux) [GPU 0]
# ==========================================
echo "[GPU 0] UNet+SH Audio -> ERP depth"
CUDA_VISIBLE_DEVICES=4 python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_20260224 \
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
    mode.print_train_images=50 \
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
    model.sh_degree=10 \
    > "${LOG_DIR}/exp1_unet_sh_erp.log" 2>&1 &
PID1=$!

# ==========================================
# (2) Baseline UNet -> Pinhole depth (BerHu + Gradient, fair comparison) [GPU 1]
# ==========================================
echo "[GPU 1] Baseline UNet Audio -> Pinhole depth"
CUDA_VISIBLE_DEVICES=5 python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_pinhole_baseline_v2_20260224 \
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
    mode.print_train_images=50 \
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
    model.generator=unet_256 \
    > "${LOG_DIR}/exp2_baseline_pinhole.log" 2>&1 &
PID2=$!

# ==========================================
# (3) Baseline UNet -> ERP depth (BerHu + Gradient, fair comparison) [GPU 2]
# ==========================================
echo "[GPU 2] Baseline UNet Audio -> ERP depth"
CUDA_VISIBLE_DEVICES=6 python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_baseline_v2_20260224 \
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
    mode.print_train_images=50 \
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
    model.generator=unet_256 \
    > "${LOG_DIR}/exp3_baseline_erp.log" 2>&1 &
PID3=$!

# ==========================================
# (4) Oracle: Pinhole RGB -> Pinhole depth (upper bound) [GPU 3]
# ==========================================
echo "[GPU 3] Oracle Pinhole RGB -> Pinhole depth"
CUDA_VISIBLE_DEVICES=7 python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_oracle_pinhole_v2_20260224 \
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
    mode.print_train_images=50 \
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
    model.generator=oracle_pinhole_256 \
    > "${LOG_DIR}/exp4_oracle_pinhole.log" 2>&1 &
PID4=$!

# ==========================================
# (5) Oracle: ERP RGB -> ERP depth (upper bound) [GPU 4]
# ==========================================
echo "[GPU 4] Oracle ERP RGB -> ERP depth"
CUDA_VISIBLE_DEVICES=8 python3 train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_oracle_erp_v2_20260224 \
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
    mode.print_train_images=50 \
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
    model.generator=oracle_erp_256 \
    > "${LOG_DIR}/exp5_oracle_erp.log" 2>&1 &
PID5=$!

echo ""
echo "=========================================="
echo "All 5 experiments launched:"
echo "  [GPU 0] PID ${PID1} - UNet+SH ERP        -> ${LOG_DIR}/exp1_unet_sh_erp.log"
echo "  [GPU 1] PID ${PID2} - Baseline Pinhole    -> ${LOG_DIR}/exp2_baseline_pinhole.log"
echo "  [GPU 2] PID ${PID3} - Baseline ERP        -> ${LOG_DIR}/exp3_baseline_erp.log"
echo "  [GPU 3] PID ${PID4} - Oracle Pinhole      -> ${LOG_DIR}/exp4_oracle_pinhole.log"
echo "  [GPU 4] PID ${PID5} - Oracle ERP          -> ${LOG_DIR}/exp5_oracle_erp.log"
echo "=========================================="
echo "Waiting for all experiments to finish..."

wait $PID1 $PID2 $PID3 $PID4 $PID5

echo "=========================================="
echo "All training completed!"
echo "=========================================="
