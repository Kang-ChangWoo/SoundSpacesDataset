#!/bin/bash
# Evaluation script for models trained with train_erp_sh.sh (v2)
# Evaluates all 5 models on the test set in parallel on separate GPUs
#
# Experiment names are constructed as:
#   {generator}_{dataset}_BS{batch_size}_Lr{learning_rate}_{optimizer}_{experiment_name}
#
# Usage:
#   bash eval_erp_sh.sh              # Evaluate using best model (default)
#   EVAL_EPOCH=150 bash eval_erp_sh.sh  # Evaluate at specific epoch
#   EVAL_EPOCH=best bash eval_erp_sh.sh # Explicitly use best model

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

# Epoch to evaluate (override with: EVAL_EPOCH=150 bash eval_erp_sh.sh)
# Default: 'best' loads the best model saved during training
EVAL_EPOCH=${EVAL_EPOCH:-best}
EVAL_ON='test'  # 'test' or 'val'

# Log directory
LOG_DIR="${SCRIPT_DIR}/server_logs"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Launching 5 evaluations in parallel (GPU 0-4)"
echo "Epoch: ${EVAL_EPOCH} | Split: ${EVAL_ON}"
echo "Logs: ${LOG_DIR}/"
echo "=========================================="

# ==========================================
# (1) UNet+SH -> ERP depth (cross-attention, degree=10) [GPU 0]
# ==========================================
echo "[GPU 0] Eval: UNet+SH Audio -> ERP depth"
CUDA_VISIBLE_DEVICES=4 python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_erp_sh_v2_20260224 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
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
    model.generator=unet_256_sh \
    model.sh_degree=10 \
    mode.max_vis_samples=100 \
    > "${LOG_DIR}/eval1_unet_sh_erp.log" 2>&1 &
PID1=$!

# ==========================================
# (2) Baseline UNet -> Pinhole depth [GPU 1]
# ==========================================
echo "[GPU 1] Eval: Baseline UNet Audio -> Pinhole depth"
CUDA_VISIBLE_DEVICES=5 python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_pinhole_baseline_v2_20260224 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
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
    model.generator=unet_256 \
    mode.max_vis_samples=100 \
    > "${LOG_DIR}/eval2_baseline_pinhole.log" 2>&1 &
PID2=$!

# ==========================================
# (3) Baseline UNet -> ERP depth [GPU 2]
# ==========================================
echo "[GPU 2] Eval: Baseline UNet Audio -> ERP depth"
CUDA_VISIBLE_DEVICES=6 python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_audio_erp_baseline_v2_20260224 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
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
    model.generator=unet_256 \
    mode.max_vis_samples=100 \
    > "${LOG_DIR}/eval3_baseline_erp.log" 2>&1 &
PID3=$!

# ==========================================
# (4) Oracle: Pinhole RGB -> Pinhole depth (upper bound) [GPU 3]
# ==========================================
echo "[GPU 3] Eval: Oracle Pinhole RGB -> Pinhole depth"
CUDA_VISIBLE_DEVICES=7 python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_oracle_pinhole_v2_20260224 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
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
    model.generator=oracle_pinhole_256 \
    mode.max_vis_samples=100 \
    > "${LOG_DIR}/eval4_oracle_pinhole.log" 2>&1 &
PID4=$!

# ==========================================
# (5) Oracle: ERP RGB -> ERP depth (upper bound) [GPU 4]
# ==========================================
echo "[GPU 4] Eval: Oracle ERP RGB -> ERP depth"
CUDA_VISIBLE_DEVICES=8 python3 test.py \
    mode=test \
    mode.experiment_name=soundspaces_oracle_erp_v2_20260224 \
    mode.checkpoints=${EVAL_EPOCH} \
    mode.eval_on=${EVAL_ON} \
    mode.criterion=BerHu \
    mode.batch_size=16 \
    mode.learning_rate=0.0003 \
    mode.optimizer=AdamW \
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
    model.generator=oracle_erp_256 \
    mode.max_vis_samples=100 \
    > "${LOG_DIR}/eval5_oracle_erp.log" 2>&1 &
PID5=$!

echo ""
echo "=========================================="
echo "All 5 evaluations launched:"
echo "  [GPU 0] PID ${PID1} - UNet+SH ERP        -> ${LOG_DIR}/eval1_unet_sh_erp.log"
echo "  [GPU 1] PID ${PID2} - Baseline Pinhole    -> ${LOG_DIR}/eval2_baseline_pinhole.log"
echo "  [GPU 2] PID ${PID3} - Baseline ERP        -> ${LOG_DIR}/eval3_baseline_erp.log"
echo "  [GPU 3] PID ${PID4} - Oracle Pinhole      -> ${LOG_DIR}/eval4_oracle_pinhole.log"
echo "  [GPU 4] PID ${PID5} - Oracle ERP          -> ${LOG_DIR}/eval5_oracle_erp.log"
echo "=========================================="
echo "Waiting for all evaluations to finish..."

wait $PID1 $PID2 $PID3 $PID4 $PID5

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
