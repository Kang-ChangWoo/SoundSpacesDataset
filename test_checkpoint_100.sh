#!/bin/bash
# Test script for checkpoint at epoch 100
# Experiment: unet_256_soundspaces_BS16_Lr0.001_AdamW_test_17DRP5sb8fy

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Testing Checkpoint: Epoch 100"
echo "Experiment: unet_256_soundspaces_BS16_Lr0.001_AdamW_test_17DRP5sb8fy"
echo "Working directory: $(pwd)"
echo "=========================================="

python3 test.py \
    mode.mode=test \
    mode.checkpoints=100 \
    +mode.experiment_name_full=unet_256_soundspaces_BS16_Lr0.001_AdamW_test_17DRP5sb8fy \
    mode.batch_size=16 \
    mode.learning_rate=0.001 \
    mode.optimizer=AdamW \
    mode.num_threads=4 \
    dataset.name=soundspaces \
    dataset.use_same_samples_all_splits=True \
    dataset.single_scene_name=17DRP5sb8fy

echo "=========================================="
echo "Test completed!"
echo "Results saved in: ./outputs/unet_256_soundspaces_BS16_Lr0.001_AdamW_test_17DRP5sb8fy/epoch_100/test/"
echo "=========================================="
