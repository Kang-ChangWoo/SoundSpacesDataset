# Installation Guide

## Option 1: Conda Environment (Recommended)

### Create conda environment from environment.yml

```bash
cd /home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset

# Create environment
conda env create -f environment.yml

# Activate environment
conda activate soundspaces_dataset
```

### Manual conda setup

```bash
# Create new conda environment
conda create -n soundspaces_dataset python=3.9 -y

# Activate environment
conda activate soundspaces_dataset

# Install PyTorch (adjust CUDA version as needed)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CPU only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other dependencies
conda install numpy pandas matplotlib scipy -y
pip install hydra-core omegaconf tensorboard Pillow
```

## Option 2: Pip with Virtual Environment

```bash
cd /home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Option 3: Using existing conda environment

If you already have a conda environment (e.g., from sound-spaces or habitat-sim):

```bash
# Activate your existing environment
conda activate your_env_name

# Install additional dependencies
pip install hydra-core omegaconf tensorboard
```

## Verify Installation

Test that everything is installed correctly:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torchaudio; print(f'TorchAudio: {torchaudio.__version__}')"
python -c "import hydra; print(f'Hydra: {hydra.__version__}')"
python -c "import tensorboard; print('TensorBoard: OK')"
```

## Troubleshooting

### CUDA Issues

If you have CUDA available, make sure to install the CUDA version of PyTorch:

```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch version
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Audio Backend Issues

If you encounter audio loading issues:

```bash
# Install soundfile (alternative audio backend)
pip install soundfile

# Or use conda
conda install -c conda-forge soundfile
```

### Hydra Version Issues

If you get Hydra compatibility errors:

```bash
pip install hydra-core==1.3.2 omegaconf==2.3.0
```

## Dependencies Summary

- **PyTorch**: Deep learning framework
- **TorchAudio**: Audio processing
- **Hydra**: Configuration management
- **TensorBoard**: Training visualization
- **NumPy/Pandas**: Data handling
- **Matplotlib**: Plotting
- **Pillow**: Image processing
- **SciPy**: Scientific computing (for some transforms)

## Next Steps

After installation, proceed to training:

```bash
# Activate environment
conda activate soundspaces_dataset

# Run training
python train.py mode.experiment_name=test_run dataset.dataset_dir=/path/to/dataset
```
