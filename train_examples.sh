#!/bin/bash
# Example training scripts for SoundSpacesDataset
# Copy and modify these commands as needed

cd "$(dirname "$0")"

# ============================================
# Example 1: Basic Training (Default Config)
# ============================================
echo "Example 1: Basic Training"
echo "python train.py"
echo ""

# ============================================
# Example 2: Training with Custom Experiment Name
# ============================================
echo "Example 2: Custom Experiment Name"
echo "python train.py mode.experiment_name=my_experiment"
echo ""

# ============================================
# Example 3: Training with Custom Batch Size and Learning Rate
# ============================================
echo "Example 3: Custom Batch Size and Learning Rate"
echo "python train.py \\"
echo "    mode.experiment_name=my_experiment \\"
echo "    mode.batch_size=32 \\"
echo "    mode.learning_rate=0.0001"
echo ""

# ============================================
# Example 4: Training with Custom Dataset Path
# ============================================
echo "Example 4: Custom Dataset Path"
echo "python train.py \\"
echo "    mode.experiment_name=my_experiment \\"
echo "    dataset.dataset_dir=/path/to/your/dataset"
echo ""

# ============================================
# Example 5: Training with Different Audio Format
# ============================================
echo "Example 5: Using Mel Spectrogram"
echo "python train.py \\"
echo "    mode.experiment_name=mel_spectrogram_exp \\"
echo "    dataset.audio_format=mel_spectrogram"
echo ""

# ============================================
# Example 6: Training with ERP Depth
# ============================================
echo "Example 6: Using ERP Depth"
echo "python train.py \\"
echo "    mode.experiment_name=erp_depth_exp \\"
echo "    dataset.depth_type=erp"
echo ""

# ============================================
# Example 7: Resume Training from Checkpoint
# ============================================
echo "Example 7: Resume from Checkpoint"
echo "python train.py \\"
echo "    mode.experiment_name=my_experiment \\"
echo "    mode.checkpoints=50"
echo ""

# ============================================
# Example 8: Full Custom Configuration
# ============================================
echo "Example 8: Full Custom Configuration"
echo "python train.py \\"
echo "    mode.experiment_name=full_custom_exp \\"
echo "    mode.batch_size=32 \\"
echo "    mode.epochs=200 \\"
echo "    mode.learning_rate=0.0005 \\"
echo "    mode.optimizer=Adam \\"
echo "    mode.validation=True \\"
echo "    mode.validation_iter=5 \\"
echo "    mode.saving_checkpoints=20 \\"
echo "    dataset.dataset_dir=/home/rvi-lab/workspace/sound-spaces/dataset \\"
echo "    dataset.audio_format=spectrogram \\"
echo "    dataset.depth_norm=True \\"
echo "    dataset.max_depth=30.0 \\"
echo "    dataset.depth_type=pinhole"
echo ""

# ============================================
# Example 9: Quick Test Run (Few Epochs)
# ============================================
echo "Example 9: Quick Test Run"
echo "python train.py \\"
echo "    mode.experiment_name=test_run \\"
echo "    mode.epochs=5 \\"
echo "    mode.batch_size=8 \\"
echo "    mode.validation_iter=1"
echo ""

# ============================================
# Example 10: Training with Different Optimizer
# ============================================
echo "Example 10: Using SGD Optimizer"
echo "python train.py \\"
echo "    mode.experiment_name=sgd_exp \\"
echo "    mode.optimizer=SGD \\"
echo "    mode.learning_rate=0.01"
echo ""

echo "=============================================="
echo "To run any of these examples, copy the command"
echo "and execute it in your terminal."
echo "=============================================="
