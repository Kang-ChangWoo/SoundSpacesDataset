#!/bin/bash
# Server training script for v260217 models (bidirectional SH cross-attention)
#
# Trains the new UNet+SH v2 model with bidirectional cross-attention:
#   Pass 1: Feature→SH, Pass 2: SH→Feature, Final readout
#
# Model: unet_256_sh_v2 (v260217_unet_sh_model.py)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Dataset root
DATASET_DIR="${SOUNDSPACES_DATASET_DIR:-/root/storage/matterport3d}"

echo "Using dataset dir: ${DATASET_DIR}"

# GPU assignment
GPU_LAST=${GPU_LAST:-4}
GPU_CURSOR=-1
next_gpu() {
  GPU_CURSOR=$(( (GPU_CURSOR + 1) % (GPU_LAST + 1) ))
}

# Parallel execution
PARALLEL=${PARALLEL:-1}
MAX_PARALLEL=$((GPU_LAST + 1))
RUN_LOG_DIR="${RUN_LOG_DIR:-server_logs/v260217_train_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_LOG_DIR}"

RUNNING=0
FAIL=0

run_train() {
  local run_name="$1"
  local gpu_id="$2"
  shift 2

  local log_file="${RUN_LOG_DIR}/${run_name}_gpu${gpu_id}.log"
  echo "Assigning run (${run_name}) to GPU ${gpu_id}"
  echo "Log: ${log_file}"

  if [[ "${PARALLEL}" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu_id}" "$@" >"${log_file}" 2>&1 &
    RUNNING=$((RUNNING + 1))

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

# ==========================================
# (1) UNet+SH v2 (bidirectional) -> ERP depth
# ==========================================
echo "=========================================="
echo "Training UNet+SH v2 (bidirectional cross-attention)"
echo "Input: Audio Spectrogram (binaural)"
echo "Output: ERP depth map (256x512)"
echo "Loss: BerHu + GradientLoss + SH Auxiliary"
echo "Model: unet_256_sh_v2"
echo "=========================================="

next_gpu
run_train "unet_sh_v2_erp" "${GPU_CURSOR}" python3 -u train.py \
    mode.mode=train \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_bidir_20260217_server \
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
    model.generator=unet_256_sh_v2 \
    model.sh_degree=10 \
    +mode.wandb_log_images=False

echo "UNet+SH v2 training launched (GPU ${GPU_CURSOR})"

echo ""
echo "=========================================="
echo "All runs launched. Waiting for completion..."
echo "Logs: ${RUN_LOG_DIR}/"
echo "=========================================="

wait_all

echo "=========================================="
echo "All training completed!"
echo "=========================================="
