#!/bin/bash
# Quick commands for DLHW3 - Copy and paste these!

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║              DLHW3 Quick Command Cheatsheet                     ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

cat << 'EOF'
# ============================================================
# 1. FIRST TIME SETUP
# ============================================================

# Setup conda environment
conda env create -f environment.yml
conda activate dlhw3

# Verify installation
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# ============================================================
# 2. EVERY TIME YOU START
# ============================================================

# Navigate to project
cd /home/ailab-students/DLHW3

# Activate environment
conda activate dlhw3

# ============================================================
# 3. TRAINING COMMANDS
# ============================================================

# Simple (fastest, for testing)
python train.py --config configs/simple_config.json

# Medium (recommended for homework)
python train.py --config configs/medium_config.json

# Strong (better performance)
python train.py --config configs/strong_config.json

# Boss (best performance, takes longer)
python train.py --config configs/boss_config.json

# ============================================================
# 4. EVALUATION COMMANDS
# ============================================================

# Evaluate trained model
python evaluate.py \
    --model_path output/best_model_epoch_3 \
    --test_data data/test.json \
    --config configs/simple_config.json \
    --output_file predictions.json

# Interactive prediction mode
python evaluate.py \
    --model_path output/best_model_epoch_3 \
    --test_data data/test.json \
    --config configs/simple_config.json \
    --single_prediction

# ============================================================
# 5. MONITORING COMMANDS
# ============================================================

# Check GPU status
nvidia-smi

# Monitor GPU in real-time
watch -n 1 nvidia-smi

# Check CUDA version
nvcc --version

# Check Python packages
pip list | grep -E 'torch|transformers|accelerate'

# ============================================================
# 6. USEFUL COMMANDS
# ============================================================

# Create data directory
mkdir -p data

# Check data files
ls -lh data/

# View first few lines of data
head data/train.json

# Count samples in data
cat data/train.json | jq '. | length'

# Check output directory
ls -lh output/

# View predictions
head -20 predictions.json

# ============================================================
# 7. TROUBLESHOOTING COMMANDS
# ============================================================

# If out of memory, use smaller batch size
# Edit config and change: "batch_size": 4

# Check if environment is activated
conda info --envs

# Reinstall packages
pip install -r requirements.txt --upgrade

# Clear GPU cache (in Python)
python -c "import torch; torch.cuda.empty_cache()"

# ============================================================
# 8. DATA PREPARATION
# ============================================================

# If you need to download data (example)
# wget https://example.com/spoken_squad_data.zip
# unzip spoken_squad_data.zip -d data/

# Verify JSON format
python -c "import json; data = json.load(open('data/train.json')); print(f'Loaded {len(data)} samples')"

# ============================================================
# 9. COMPLETE WORKFLOW (COPY-PASTE)
# ============================================================

# First time setup
conda env create -f environment.yml
conda activate dlhw3
mkdir -p data

# Training
python train.py --config configs/simple_config.json

# Evaluation
python evaluate.py \
    --model_path output/best_model_epoch_3 \
    --test_data data/test.json \
    --config configs/simple_config.json \
    --output_file predictions.json

# ============================================================
# 10. ADVANCED OPTIONS
# ============================================================

# Train with specific GPU
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/simple_config.json

# Train on CPU (slow!)
CUDA_VISIBLE_DEVICES="" python train.py --config configs/simple_config.json

# Multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/boss_config.json

# Background training (saves output to file)
nohup python train.py --config configs/medium_config.json > training.log 2>&1 &

# Check background process
tail -f training.log

EOF

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  TIP: Copy the commands you need and paste in your terminal!    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
