#!/bin/bash
# Evaluation script for models trained with train_erp_sh.sh (v2)
# Evaluates all three models on the test set
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

# Epoch to evaluate (override with: EVAL_EPOCH=150 bash eval_erp_sh.sh)
# Default: 'best' loads the best model saved during training
EVAL_EPOCH=${EVAL_EPOCH:-best}
EVAL_ON='test'  # 'test' or 'val'

echo "=========================================="
echo "Evaluating all v2 models at epoch ${EVAL_EPOCH} on ${EVAL_ON} set"
echo "=========================================="

# # ==========================================
# # (1) UNet+SH -> ERP depth (cross-attention, degree=10)
# # ==========================================
# echo ""
# echo "=========================================="
# echo "Eval: UNet+SH Audio -> ERP depth (v2)"
# echo "=========================================="
#
# python3 test.py \
#     mode=test \
#     mode.experiment_name=soundspaces_audio_erp_sh_v2 \
#     mode.checkpoints=${EVAL_EPOCH} \
#     mode.eval_on=${EVAL_ON} \
#     mode.criterion=BerHu \
#     mode.batch_size=16 \
#     mode.learning_rate=0.0003 \
#     mode.optimizer=AdamW \
#     mode.num_threads=4 \
#     dataset.name=soundspaces \
#     dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
#     dataset.use_same_samples_all_splits=False \
#     dataset.single_scene_test_mode=False \
#     dataset.input_type=audio \
#     dataset.audio_format=spectrogram \
#     dataset.preprocess=resize \
#     dataset.depth_norm=True \
#     'dataset.images_size=[256,512]' \
#     dataset.min_depth=0.01 \
#     dataset.max_depth=10.0 \
#     dataset.depth_type=erp \
#     model.generator=unet_256_sh \
#     model.sh_degree=10 \
#     mode.max_vis_samples=100

# # ==========================================
# # (2) Baseline UNet -> Pinhole depth
# # ==========================================
# echo ""
# echo "=========================================="
# echo "Eval: Baseline UNet Audio -> Pinhole depth (v2)"
# echo "=========================================="
#
# python3 test.py \
#     mode=test \
#     mode.experiment_name=soundspaces_audio_pinhole_baseline_v2 \
#     mode.checkpoints=${EVAL_EPOCH} \
#     mode.eval_on=${EVAL_ON} \
#     mode.criterion=BerHu \
#     mode.batch_size=16 \
#     mode.learning_rate=0.0003 \
#     mode.optimizer=AdamW \
#     mode.num_threads=4 \
#     dataset.name=soundspaces \
#     dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
#     dataset.use_same_samples_all_splits=False \
#     dataset.single_scene_test_mode=False \
#     dataset.input_type=audio \
#     dataset.audio_format=spectrogram \
#     dataset.preprocess=resize \
#     dataset.depth_norm=True \
#     'dataset.images_size=[256,512]' \
#     dataset.min_depth=0.01 \
#     dataset.max_depth=10.0 \
#     dataset.depth_type=pinhole \
#     model.generator=unet_256 \
#     mode.max_vis_samples=100

# # ==========================================
# # (3) Baseline UNet -> ERP depth
# # ==========================================
# echo ""
# echo "=========================================="
# echo "Eval: Baseline UNet Audio -> ERP depth (v2)"
# echo "=========================================="
#
# python3 test.py \
#     mode=test \
#     mode.experiment_name=soundspaces_audio_erp_baseline_v2 \
#     mode.checkpoints=${EVAL_EPOCH} \
#     mode.eval_on=${EVAL_ON} \
#     mode.criterion=BerHu \
#     mode.batch_size=16 \
#     mode.learning_rate=0.0003 \
#     mode.optimizer=AdamW \
#     mode.num_threads=4 \
#     dataset.name=soundspaces \
#     dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
#     dataset.use_same_samples_all_splits=False \
#     dataset.single_scene_test_mode=False \
#     dataset.input_type=audio \
#     dataset.audio_format=spectrogram \
#     dataset.preprocess=resize \
#     dataset.depth_norm=True \
#     'dataset.images_size=[256,512]' \
#     dataset.min_depth=0.01 \
#     dataset.max_depth=10.0 \
#     dataset.depth_type=erp \
#     model.generator=unet_256 \
#     mode.max_vis_samples=100

# ==========================================
# (4) UNet+SH -> ERP depth (full experiment name testing)
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
    model.generator=unet_256_sh \
    model.sh_degree=10 \
    mode.max_vis_samples=100

# ==========================================
# (5) Oracle: Pinhole RGB -> Pinhole depth (upper bound)
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
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
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
# (6) Oracle: ERP RGB -> ERP depth (upper bound)
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
    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \
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
