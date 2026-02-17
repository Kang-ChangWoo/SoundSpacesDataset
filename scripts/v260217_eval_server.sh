#!/bin/bash
# Server evaluation script for v260217 models (bidirectional SH cross-attention)
#
# Evaluates the UNet+SH v2 model trained by v260217_train_server.sh
#
# Usage:
#   bash scripts/v260217_eval_server.sh
#   EVAL_EPOCH=100 EVAL_ON=val bash scripts/v260217_eval_server.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Dataset root
DATASET_DIR="${SOUNDSPACES_DATASET_DIR:-/root/storage/matterport3d}"

# Epoch to evaluate
EVAL_EPOCH=${EVAL_EPOCH:-best}
EVAL_ON=${EVAL_ON:-test}

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
RUN_LOG_DIR="${RUN_LOG_DIR:-server_logs/v260217_eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_LOG_DIR}"

RUNNING=0
FAIL=0

run_eval() {
  local run_name="$1"
  local gpu_id="$2"
  shift 2

  local log_file="${RUN_LOG_DIR}/${run_name}_gpu${gpu_id}.log"
  echo "Assigning eval (${run_name}) to GPU ${gpu_id}"
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
    echo "One or more evaluations failed. Check logs under: ${RUN_LOG_DIR}"
    exit 1
  fi
}

echo "=========================================="
echo "Evaluating v260217 models at epoch ${EVAL_EPOCH} on ${EVAL_ON} set"
echo "=========================================="

# ==========================================
# (1) UNet+SH v2 (bidirectional) -> ERP depth
# ==========================================
echo ""
echo "=========================================="
echo "Eval: UNet+SH v2 (bidirectional) Audio -> ERP depth"
echo "=========================================="

next_gpu
run_eval "unet_sh_v2_erp" "${GPU_CURSOR}" python3 -u test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_bidir_20260217_server \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=32 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
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
    model.generator=unet_256_sh_v2 \
    model.sh_degree=10 \
    mode.max_vis_samples=100

echo "UNet+SH v2 eval launched (GPU ${GPU_CURSOR})"

echo ""
echo "=========================================="
echo "All evaluations launched. Waiting for completion..."
echo "Logs: ${RUN_LOG_DIR}/"
echo "=========================================="

wait_all

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
