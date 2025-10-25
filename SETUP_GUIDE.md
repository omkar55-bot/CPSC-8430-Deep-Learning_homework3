# Environment Setup Guide for DLHW3

## Option 1: Using Conda (Recommended)

### Method A: Create environment from environment.yml

```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate dlhw3

# Verify installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Method B: Create environment manually

```bash
# Create a new conda environment
conda create -n dlhw3 python=3.10 -y

# Activate the environment
conda activate dlhw3

# Install PyTorch (choose based on your CUDA version)
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# For CUDA 12.1:
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# For CPU only:
# conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other dependencies
pip install -r requirements.txt
```

## Option 2: Using pip with virtual environment

```bash
# Create virtual environment
python -m venv dlhw3_env

# Activate the environment
# On Linux/Mac:
source dlhw3_env/bin/activate
# On Windows:
# dlhw3_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch (visit https://pytorch.org for your specific CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt
```

## Verify Installation

After installation, run these commands to verify everything is working:

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Check Transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Check Accelerate
python -c "import accelerate; print(f'Accelerate: {accelerate.__version__}')"

# Check all imports
python -c "
import torch
import transformers
import datasets
import accelerate
import numpy as np
import pandas as pd
import sklearn
print('✅ All packages imported successfully!')
"
```

## Common Issues and Solutions

### Issue 1: CUDA Version Mismatch
**Problem**: PyTorch CUDA version doesn't match your GPU CUDA version

**Solution**: 
```bash
# Check your CUDA version
nvidia-smi

# Install matching PyTorch version from https://pytorch.org
```

### Issue 2: Import errors for torch
**Problem**: `ImportError: cannot import name 'torch'`

**Solution**:
```bash
# Uninstall and reinstall PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue 3: Accelerate not working with FP16
**Problem**: FP16 training fails

**Solution**:
```bash
# Make sure you have a compatible GPU (T4, V100, A100, etc.)
# Update accelerate
pip install --upgrade accelerate

# If still failing, disable FP16 in config:
# Set "use_fp16": false in your config file
```

### Issue 4: Out of Memory (OOM) errors
**Problem**: CUDA out of memory during training

**Solution**:
- Reduce batch_size in config file
- Increase gradient_accumulation_steps
- Use FP16 training (if not already enabled)
- Reduce max_length if possible

## Quick Test

Run this command to test if your setup is ready:

```bash
python -c "
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

print('Testing environment setup...')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Transformers version: {transformers.__version__}')

# Try loading a model (this will download ~400MB)
print('Loading test model...')
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
print('✅ Environment is ready!')
"
```

## Next Steps

After setting up your environment:

1. **Prepare your data**: Place train.json, dev.json, test.json in `data/` folder

2. **Start training**:
   ```bash
   python train.py --config configs/simple_config.json
   ```

3. **Monitor training**: 
   - Watch the progress bar
   - Check logs in `logs/` folder
   - If using wandb, check your dashboard

4. **Evaluate**:
   ```bash
   python evaluate.py --model_path output/best_model \
                      --test_data data/test.json \
                      --config configs/medium_config.json
   ```

## Environment Management

```bash
# List all conda environments
conda env list

# Activate environment
conda activate dlhw3

# Deactivate environment
conda deactivate

# Remove environment (if needed)
conda env remove -n dlhw3

# Export environment (to share with others)
conda env export > environment_exact.yml

# Update packages
conda update --all
pip install --upgrade -r requirements.txt
```

## GPU Memory Requirements

Approximate GPU memory needed for different configs:

| Config | Batch Size | FP32 Memory | FP16 Memory |
|--------|------------|-------------|-------------|
| Simple | 16         | ~12 GB      | ~6 GB       |
| Medium | 16         | ~12 GB      | ~6 GB       |
| Strong | 12         | ~14 GB      | ~7 GB       |
| Boss   | 8          | ~20 GB      | ~10 GB      |

If you have limited GPU memory (e.g., 8GB):
- Use FP16 training (enabled by default)
- Reduce batch size (e.g., 4 or 8)
- Increase gradient_accumulation_steps to compensate

## Troubleshooting

If you encounter any issues:

1. Check Python version: `python --version` (should be 3.8-3.11)
2. Check CUDA compatibility: `nvidia-smi`
3. Verify PyTorch installation: `python -c "import torch; print(torch.__version__)"`
4. Check GPU detection: `python -c "import torch; print(torch.cuda.is_available())"`
5. Update packages: `pip install --upgrade -r requirements.txt`

For more help, refer to:
- PyTorch installation guide: https://pytorch.org
- Hugging Face documentation: https://huggingface.co/docs
- Accelerate documentation: https://huggingface.co/docs/accelerate

Good luck! 🚀
