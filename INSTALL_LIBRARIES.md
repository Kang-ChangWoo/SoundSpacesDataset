# Library Installation List

## Conda Installation (Recommended)

```bash
conda create -n soundspaces_dataset python=3.9 -y
conda activate soundspaces_dataset

# PyTorch (choose based on your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# OR for CPU only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Core dependencies
conda install numpy pandas matplotlib scipy -y

# Additional packages via pip
pip install hydra-core omegaconf tensorboard Pillow
```

## Pip Installation (Alternative)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows

# Install all libraries
pip install torch torchvision torchaudio
pip install hydra-core omegaconf
pip install tensorboard
pip install numpy pandas matplotlib scipy
pip install Pillow
```

## Complete Library List

### Core Deep Learning
- **torch** - PyTorch deep learning framework
- **torchvision** - Computer vision utilities for PyTorch
- **torchaudio** - Audio processing for PyTorch

### Configuration Management
- **hydra-core** - Configuration management framework
- **omegaconf** - Configuration system (used by Hydra)

### Data Processing
- **numpy** - Numerical computing
- **pandas** - Data manipulation and analysis

### Visualization
- **matplotlib** - Plotting library
- **tensorboard** - Training visualization and logging

### Scientific Computing
- **scipy** - Scientific computing library

### Image Processing
- **Pillow** (PIL) - Image processing library

### Optional (if needed)
- **soundfile** - Alternative audio backend (if torchaudio has issues)

## One-Line Installation

### Conda
```bash
conda env create -f environment.yml && conda activate soundspaces_dataset
```

### Pip
```bash
pip install -r requirements.txt
```

## Minimum Version Requirements

- Python >= 3.8
- torch >= 2.0.0
- torchvision >= 0.15.0
- torchaudio >= 2.0.0
- hydra-core >= 1.3.0
- omegaconf >= 2.3.0
- tensorboard >= 2.10.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.5.0
- Pillow >= 9.0.0
- scipy >= 1.8.0

## Quick Copy-Paste Commands

### For Conda (with CUDA 11.8)
```bash
conda create -n soundspaces_dataset python=3.9 -y
conda activate soundspaces_dataset
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install numpy pandas matplotlib scipy -y
pip install hydra-core omegaconf tensorboard Pillow
```

### For Pip
```bash
pip install torch torchvision torchaudio hydra-core omegaconf tensorboard numpy pandas matplotlib scipy Pillow
```
