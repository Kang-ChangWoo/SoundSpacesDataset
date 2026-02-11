# How to Create Conda Environment

## Method 1: Using environment.yml File (Easiest)

```bash
# Navigate to project directory
cd /home/rvi-lab/workspace/Batvision-Dataset/SoundSpacesDataset

# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate soundspaces_dataset
```

## Method 2: Manual Creation Step-by-Step

### Step 1: Create the environment
```bash
conda create -n soundspaces_dataset python=3.9 -y
```

### Step 2: Activate the environment
```bash
conda activate soundspaces_dataset
```

### Step 3: Install PyTorch
```bash
# For GPU (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CPU only
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
```

### Step 4: Install other dependencies
```bash
# Using conda
conda install numpy pandas matplotlib scipy -y

# Using pip
pip install hydra-core omegaconf tensorboard Pillow
```

## Complete Command Sequence

```bash
# 1. Create environment
conda create -n soundspaces_dataset python=3.9 -y

# 2. Activate
conda activate soundspaces_dataset

# 3. Install PyTorch (GPU version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 4. Install dependencies
conda install numpy pandas matplotlib scipy -y
pip install hydra-core omegaconf tensorboard Pillow

# 5. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
```

## Useful Conda Commands

### Activate environment
```bash
conda activate soundspaces_dataset
```

### Deactivate environment
```bash
conda deactivate
```

### List all environments
```bash
conda env list
# or
conda info --envs
```

### Remove environment
```bash
conda env remove -n soundspaces_dataset
```

### Export environment (create environment.yml)
```bash
conda env export > environment.yml
```

### Clone existing environment
```bash
conda create -n new_env_name --clone soundspaces_dataset
```

## Troubleshooting

### If conda command not found
```bash
# Add conda to PATH (Linux/Mac)
export PATH="$HOME/anaconda3/bin:$PATH"
# or
export PATH="$HOME/miniconda3/bin:$PATH"
```

### Check if conda is installed
```bash
conda --version
```

### Update conda
```bash
conda update conda
```

## After Creating Environment

Once your environment is created and activated:

```bash
# Verify you're in the right environment
conda info

# Start training
python train.py mode.experiment_name=test_run
```
