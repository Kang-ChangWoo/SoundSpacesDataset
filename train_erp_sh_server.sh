#!/bin/bash
# Server training script for UNet + Spherical Harmonics (SH) on ERP depth estimation
#
# Differences from train_erp_sh.sh:
# - Uses SOUNDSPACES_DATASET_DIR (default: /root/storage/matterport3d)
# - Keeps the original local workstation script unchanged

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Dataset root (scene directories live directly under this folder)
DATASET_DIR="${SOUNDSPACES_DATASET_DIR:-/root/storage/matterport3d}"

echo "Using dataset dir: ${DATASET_DIR}"

# GPU assignment: use GPUs 0..GPU_LAST (one experiment per GPU, round-robin)
# Override: GPU_LAST=7 bash train_erp_sh_server.sh
GPU_LAST=4
GPU_CURSOR=-1
next_gpu() {
  GPU_CURSOR=$(( (GPU_CURSOR + 1) % (GPU_LAST + 1) ))
}

# Parallel execution: all runs launch simultaneously in background
# Override: PARALLEL=0 bash train_erp_sh_server.sh  (for sequential)
PARALLEL=1
MAX_PARALLEL=$((GPU_LAST + 1))
RUN_LOG_DIR="${RUN_LOG_DIR:-server_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_LOG_DIR}"

RUNNING=0
FAIL=0

run_train() {
  local run_name="$1"
  local gpu_id="$2"
  shift 2

  local log_file="${RUN_LOG_DIR}/${run_name}_gpu${gpu_id}.log"
  echo "Assigning run (${run_name}) to physical GPU ${gpu_id} (CUDA_VISIBLE_DEVICES=${gpu_id})"
  echo "Log: ${log_file}"

  if [[ "${PARALLEL}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" >"${log_file}" 2>&1 &
    RUNNING=$((RUNNING + 1))

    # Concurrency gate
    while [[ "${RUNNING}" -ge "${MAX_PARALLEL}" ]]; do
      if ! wait -n; then
        FAIL=1
      fi
      RUNNING=$((RUNNING - 1))
    done
  else
    CUDA_VISIBLE_DEVICES="${gpu_id}" "$@"
  fi
}

wait_all() {
  if [[ "${PARALLEL}" != "1" ]]; then
    return 0
  fi

  while [[ "${RUNNING}" -gt 0 ]]; do
    if ! wait -n; then
      FAIL=1
    fi
    RUNNING=$((RUNNING - 1))
  done

  if [[ "${FAIL}" -ne 0 ]]; then
    echo "One or more runs failed. Check logs under: ${RUN_LOG_DIR}"
    exit 1
  fi
}

# NOTE: This script does not activate conda/venv.
# Make sure your current shell already has the right Python environment.

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

next_gpu
run_train "unet_sh_erp" "${GPU_CURSOR}" python3 -u train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_20260211_server \
    mode.batch_size=32 \
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
    dataset.dataset_dir="${DATASET_DIR}" \
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
    +mode.wandb_log_images=False

echo "UNet+SH training launched (GPU ${GPU_CURSOR})"

# ==========================================
# (2) Baseline UNet -> Pinhole depth (BerHu + Gradient, fair comparison)
# ==========================================
echo ""
echo "=========================================="
echo "Training: Baseline UNet Audio -> Pinhole depth"
echo "Loss: BerHu + GradientLoss (same as SH for fair comparison)"
echo "=========================================="

next_gpu
run_train "baseline_pinhole" "${GPU_CURSOR}" python3 -u train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_pinhole_baseline_v2_20260211_server \
    mode.batch_size=32 \
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
    dataset.dataset_dir="${DATASET_DIR}" \
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
    +mode.wandb_log_images=False

echo "Baseline Pinhole launched (GPU ${GPU_CURSOR})"

# ==========================================
# (3) Baseline UNet -> ERP depth (BerHu + Gradient, fair comparison)
# ==========================================
echo ""
echo "=========================================="
echo "Training: Baseline UNet Audio -> ERP depth"
echo "Loss: BerHu + GradientLoss (same as SH for fair comparison)"
echo "=========================================="

next_gpu
run_train "baseline_erp" "${GPU_CURSOR}" python3 -u train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_baseline_v2_20260211_server \
    mode.batch_size=32 \
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
    dataset.dataset_dir="${DATASET_DIR}" \
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
    +mode.wandb_log_images=False

echo "Baseline ERP launched (GPU ${GPU_CURSOR})"

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

next_gpu
run_train "oracle_pinhole_rgb" "${GPU_CURSOR}" python3 -u train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_oracle_pinhole_v2_20260211_server \
    mode.batch_size=32 \
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
    dataset.dataset_dir="${DATASET_DIR}" \
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
    +mode.wandb_log_images=False

echo "Oracle Pinhole launched (GPU ${GPU_CURSOR})"

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

next_gpu
run_train "oracle_erp_rgb" "${GPU_CURSOR}" python3 -u train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_oracle_erp_v2_20260211_server \
    mode.batch_size=32 \
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
    dataset.dataset_dir="${DATASET_DIR}" \
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
    +mode.wandb_log_images=False

echo "Oracle ERP launched (GPU ${GPU_CURSOR})"

echo ""
echo "=========================================="
echo "All 5 runs launched. Waiting for completion..."
echo "Logs: ${RUN_LOG_DIR}/"
echo "=========================================="

wait_all

echo "=========================================="
echo "All training completed!"
echo "=========================================="

