# Quick Installation Guide

## 🚀 One-Line Installation (Recommended)

```bash
bash setup_conda.sh
```

This script will:
- Check if conda is installed
- Detect your CUDA version
- Create the conda environment
- Install all dependencies
- Verify the installation

---

## 📦 Manual Installation Options

### Option 1: Using environment.yml

```bash
conda env create -f environment.yml
conda activate dlhw3
```

### Option 2: Using requirements.txt

```bash
# Create environment
conda create -n dlhw3 python=3.10 -y
conda activate dlhw3

# Install PyTorch (CUDA 11.8)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Install other packages
pip install -r requirements.txt
```

### Option 3: CPU-only installation

```bash
conda create -n dlhw3 python=3.10 -y
conda activate dlhw3
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
pip install -r requirements.txt
```

---

## ✅ Verify Installation

```bash
conda activate dlhw3
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

---

## 🎯 Quick Start

```bash
# Activate environment
conda activate dlhw3

# Train
python train.py --config configs/simple_config.json

# Evaluate
python evaluate.py --model_path output/best_model \
                   --test_data data/test.json \
                   --config configs/simple_config.json
```

---

## 📚 Files Created

- **requirements.txt** - Updated with compatible package versions
- **environment.yml** - Conda environment specification
- **setup_conda.sh** - Automated setup script
- **SETUP_GUIDE.md** - Detailed installation guide with troubleshooting

---

## 🔧 CUDA Version Guide

| Your GPU | Recommended CUDA | PyTorch Command |
|----------|------------------|-----------------|
| RTX 4090/4080 | 12.1 | `pytorch-cuda=12.1` |
| RTX 3090/3080 | 11.8 | `pytorch-cuda=11.8` |
| Tesla V100 | 11.8 | `pytorch-cuda=11.8` |
| Tesla T4 | 11.8 | `pytorch-cuda=11.8` |
| No GPU | CPU | `cpuonly` |

Check your CUDA version:
```bash
nvidia-smi
```

---

## ⚠️ Common Issues

**Problem**: CUDA out of memory
- Reduce `batch_size` in config
- Enable `use_fp16: true`
- Increase `gradient_accumulation_steps`

**Problem**: Import errors
- Make sure environment is activated: `conda activate dlhw3`
- Reinstall packages: `pip install -r requirements.txt`

**Problem**: FP16 not working
- Check if your GPU supports FP16 (T4, V100, A100)
- Update accelerate: `pip install --upgrade accelerate`
- Set `use_fp16: false` in config if needed

---

For detailed troubleshooting, see **SETUP_GUIDE.md**
