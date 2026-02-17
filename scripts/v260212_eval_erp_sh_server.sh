#!/bin/bash
# Server evaluation script for models trained with train_erp_sh_server.sh
#
# Differences from eval_erp_sh.sh:
# - Uses SOUNDSPACES_DATASET_DIR (default: /root/storage/matterport3d)
# - Evaluates all 5 models trained by train_erp_sh_server.sh
# - Supports parallel GPU execution

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Dataset root (scene directories live directly under this folder)
DATASET_DIR="${SOUNDSPACES_DATASET_DIR:-/root/storage/matterport3d}"

# Epoch to evaluate (override with: EVAL_EPOCH=150 bash eval_erp_sh_server.sh)
EVAL_EPOCH=${EVAL_EPOCH:-best}
EVAL_ON=${EVAL_ON:-test}  # 'test' or 'val'

echo "Using dataset dir: ${DATASET_DIR}"

# GPU assignment: use GPUs 0..GPU_LAST (one experiment per GPU, round-robin)
# Override: GPU_LAST=7 bash eval_erp_sh_server.sh
GPU_LAST=4
GPU_CURSOR=-1
next_gpu() {
  GPU_CURSOR=$(( (GPU_CURSOR + 1) % (GPU_LAST + 1) ))
}

# Parallel execution: all runs launch simultaneously in background
# Override: PARALLEL=0 bash eval_erp_sh_server.sh  (for sequential)
PARALLEL=1
MAX_PARALLEL=$((GPU_LAST + 1))
RUN_LOG_DIR="${RUN_LOG_DIR:-server_logs/eval_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_LOG_DIR}"

RUNNING=0
FAIL=0

run_eval() {
  local run_name="$1"
  local gpu_id="$2"
  shift 2

  local log_file="${RUN_LOG_DIR}/${run_name}_gpu${gpu_id}.log"
  echo "Assigning eval (${run_name}) to physical GPU ${gpu_id} (CUDA_VISIBLE_DEVICES=${gpu_id})"
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
    echo "One or more evaluations failed. Check logs under: ${RUN_LOG_DIR}"
    exit 1
  fi
}

# NOTE: This script does not activate conda/venv.
# Make sure your current shell already has the right Python environment.

echo "=========================================="
echo "Evaluating all 5 server models at epoch ${EVAL_EPOCH} on ${EVAL_ON} set"
echo "=========================================="

# ==========================================
# (1) UNet+SH -> ERP depth
# ==========================================
echo ""
echo "=========================================="
echo "Eval: UNet+SH Audio -> ERP depth"
echo "=========================================="

next_gpu
run_eval "unet_sh_erp" "${GPU_CURSOR}" python3 -u test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_20260214filter_server \
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
    model.generator=unet_256_sh \
    model.sh_degree=10 \
    mode.max_vis_samples=100

echo "UNet+SH eval launched (GPU ${GPU_CURSOR})"

# ==========================================
# (2) Baseline UNet -> Pinhole depth
# ==========================================
echo ""
echo "=========================================="
echo "Eval: Baseline UNet Audio -> Pinhole depth"
echo "=========================================="

next_gpu
run_eval "baseline_pinhole" "${GPU_CURSOR}" python3 -u test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_pinhole_baseline_v2_20260214filter_server \
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
    dataset.preprocess=crop \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    dataset.min_depth=0.01 \
    dataset.max_depth=10.0 \
    dataset.depth_type=pinhole \
    dataset.filter_wall_samples=True \
    model.generator=unet_256 \
    mode.max_vis_samples=100

echo "Baseline Pinhole eval launched (GPU ${GPU_CURSOR})"

# ==========================================
# (3) Baseline UNet -> ERP depth
# ==========================================
echo ""
echo "=========================================="
echo "Eval: Baseline UNet Audio -> ERP depth"
echo "=========================================="

next_gpu
run_eval "baseline_erp" "${GPU_CURSOR}" python3 -u test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_erp_baseline_v2_20260214filter_server \
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
    model.generator=unet_256 \
    mode.max_vis_samples=100

echo "Baseline ERP eval launched (GPU ${GPU_CURSOR})"

# ==========================================
# (4) Oracle: Pinhole RGB -> Pinhole depth
# ==========================================
echo ""
echo "=========================================="
echo "Eval: Oracle Pinhole RGB -> Pinhole depth"
echo "=========================================="

next_gpu
run_eval "oracle_pinhole_rgb" "${GPU_CURSOR}" python3 -u test.py \
    mode=test \
    mode.experiment_name=soundspaces_oracle_pinhole_v2_20260214filter_server \
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
    dataset.input_type=rgb \
    dataset.input_image_type=pinhole \
    dataset.preprocess=crop \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    dataset.min_depth=0.01 \
    dataset.max_depth=10.0 \
    dataset.depth_type=pinhole \
    dataset.filter_wall_samples=True \
    model.generator=oracle_pinhole_256 \
    mode.max_vis_samples=100

echo "Oracle Pinhole eval launched (GPU ${GPU_CURSOR})"

# ==========================================
# (5) Oracle: ERP RGB -> ERP depth
# ==========================================
echo ""
echo "=========================================="
echo "Eval: Oracle ERP RGB -> ERP depth"
echo "=========================================="

next_gpu
run_eval "oracle_erp_rgb" "${GPU_CURSOR}" python3 -u test.py \
    mode=test \
    mode.experiment_name=soundspaces_oracle_erp_v2_20260214filter_server \
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
    dataset.input_type=rgb \
    dataset.input_image_type=erp \
    dataset.preprocess=resize \
    dataset.depth_norm=True \
    'dataset.images_size=[256,512]' \
    dataset.min_depth=0.01 \
    dataset.max_depth=10.0 \
    dataset.depth_type=erp \
    model.generator=oracle_erp_256 \
    mode.max_vis_samples=100

echo "Oracle ERP eval launched (GPU ${GPU_CURSOR})"

echo ""
echo "=========================================="
echo "All 5 evaluations launched. Waiting for completion..."
echo "Logs: ${RUN_LOG_DIR}/"
echo "=========================================="

wait_all

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
