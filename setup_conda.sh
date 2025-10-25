#!/bin/bash
# Setup script for DLHW3 conda environment

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║         DLHW3 - Spoken SQuAD QA Environment Setup               ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Anaconda or Miniconda first:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda found: $(conda --version)"
echo ""

# Detect CUDA version if available
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 Detecting CUDA version..."
    nvidia-smi --query-gpu=driver_version --format=csv,noheader
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "   CUDA Version: $CUDA_VERSION"
    echo ""
else
    echo "⚠️  Warning: nvidia-smi not found. Will install CPU version."
    echo ""
fi

# Ask user for installation method
echo "Choose installation method:"
echo "1) Use environment.yml (recommended)"
echo "2) Manual installation with requirements.txt"
echo "3) CPU-only installation"
read -p "Enter choice [1-3]: " choice
echo ""

case $choice in
    1)
        echo "📦 Creating conda environment from environment.yml..."
        conda env create -f environment.yml
        ENV_NAME="dlhw3"
        ;;
    2)
        echo "📦 Creating conda environment manually..."
        ENV_NAME="dlhw3"
        
        # Create base environment
        conda create -n $ENV_NAME python=3.10 -y
        
        # Activate environment
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
        
        # Ask for CUDA version
        echo ""
        echo "Select PyTorch CUDA version:"
        echo "1) CUDA 11.8"
        echo "2) CUDA 12.1"
        echo "3) CPU only"
        read -p "Enter choice [1-3]: " cuda_choice
        
        case $cuda_choice in
            1)
                echo "Installing PyTorch with CUDA 11.8..."
                conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
                ;;
            2)
                echo "Installing PyTorch with CUDA 12.1..."
                conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
                ;;
            3)
                echo "Installing PyTorch (CPU only)..."
                conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
                ;;
            *)
                echo "Invalid choice. Defaulting to CUDA 11.8..."
                conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
                ;;
        esac
        
        # Install other requirements
        echo ""
        echo "📦 Installing other dependencies..."
        pip install -r requirements.txt
        ;;
    3)
        echo "📦 Creating CPU-only conda environment..."
        ENV_NAME="dlhw3"
        
        conda create -n $ENV_NAME python=3.10 -y
        eval "$(conda shell.bash hook)"
        conda activate $ENV_NAME
        
        conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
        pip install -r requirements.txt
        ;;
    *)
        echo "❌ Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                   Verifying Installation                        ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Activate environment for testing
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Verify installation
echo "🔍 Checking Python version..."
python --version

echo ""
echo "🔍 Checking PyTorch installation..."
python -c "import torch; print(f'   PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    python -c "import torch; print(f'   CUDA version: {torch.version.cuda}')"
    python -c "import torch; print(f'   GPU count: {torch.cuda.device_count()}')"
fi

echo ""
echo "🔍 Checking other packages..."
python -c "import transformers; print(f'   Transformers: {transformers.__version__}')"
python -c "import accelerate; print(f'   Accelerate: {accelerate.__version__}')"
python -c "import numpy; print(f'   NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'   Pandas: {pandas.__version__}')"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ Installation Complete! ✅                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "To activate the environment, run:"
echo "   conda activate $ENV_NAME"
echo ""
echo "To start training, run:"
echo "   python train.py --config configs/simple_config.json"
echo ""
echo "For more information, see SETUP_GUIDE.md"
echo ""
