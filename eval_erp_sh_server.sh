#!/bin/bash
# Server evaluation script for models trained with train_erp_sh.sh (v2)
#
# Differences from eval_erp_sh.sh:
# - Uses SOUNDSPACES_DATASET_DIR (default: /root/storage/matterport3d)
# - Keeps the original local workstation script unchanged

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# NOTE: This script does not activate conda/venv.
# Make sure your current shell already has the right Python environment.

# Dataset root (scene directories live directly under this folder)
DATASET_DIR="${SOUNDSPACES_DATASET_DIR:-/root/storage/matterport3d}"

# Epoch to evaluate (override with: EVAL_EPOCH=150 bash eval_erp_sh_server.sh)
EVAL_EPOCH=${EVAL_EPOCH:-best}
EVAL_ON=${EVAL_ON:-test}  # 'test' or 'val'

echo "Using dataset dir: ${DATASET_DIR}"
echo "=========================================="
echo "Evaluating all v2 models at epoch ${EVAL_EPOCH} on ${EVAL_ON} set"
echo "=========================================="

# ==========================================
# (1) UNet+SH -> ERP depth (full experiment name testing)
# unet_256_sh_soundspaces_BS16_Lr0.0003_AdamW_soundspaces_audio_erp_sh_v2_20260211
# ==========================================
echo ""
echo "=========================================="
echo "Eval: unet_256_sh_soundspaces_BS16_Lr0.0003_AdamW_soundspaces_audio_erp_sh_v2_20260211"
echo "=========================================="

python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_20260211 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
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

# ==========================================
# (2) Oracle: Pinhole RGB -> Pinhole depth (upper bound)
# ==========================================
echo ""
echo "=========================================="
echo "Eval: Oracle Pinhole RGB -> Pinhole depth"
echo "=========================================="

python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_oracle_pinhole_v2_20260211 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
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
    model.generator=oracle_pinhole_256 \
    mode.max_vis_samples=100

# ==========================================
# (3) Oracle: ERP RGB -> ERP depth (upper bound)
# ==========================================
echo ""
echo "=========================================="
echo "Eval: Oracle ERP RGB -> ERP depth"
echo "=========================================="

python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_oracle_erp_v2_20260211 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
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

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="

