# 🚀 How to Run DLHW3 - Complete Guide

## Step 1: Setup Environment (First Time Only)

### Option A: Automated Setup (Recommended)
```bash
# Navigate to project directory
cd /home/ailab-students/DLHW3

# Run the setup script
bash setup_conda.sh
```

### Option B: Manual Setup
```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate dlhw3
```

### Verify Installation
```bash
python -c "import torch; print('✅ PyTorch:', torch.__version__)"
python -c "import transformers; print('✅ Transformers:', transformers.__version__)"
```

---

## Step 2: Prepare Your Data

Create a `data/` directory and place your dataset files:

```bash
mkdir -p data
```

Your data files should be in JSON format:
- `data/train.json` - Training data
- `data/dev.json` - Validation/development data
- `data/test.json` - Test data

Expected JSON format:
```json
[
  {
    "id": "example_1",
    "question": "What causes precipitation to fall?",
    "context": "In meteorology, precipitation is any product...",
    "answer_text": "gravity",
    "answer_start": 45
  }
]
```

---

## Step 3: Choose Your Configuration Level

The project has 4 difficulty levels:

| Level | Config File | Description | Training Time (FP16) |
|-------|-------------|-------------|----------------------|
| **Simple** | `configs/simple_config.json` | Basic configuration, good for testing | ~8 minutes |
| **Medium** | `configs/medium_config.json` | + LR decay + doc_stride optimization | ~8 minutes |
| **Strong** | `configs/strong_config.json` | + Better preprocessing + ELECTRA model | ~25 minutes |
| **Boss** | `configs/boss_config.json` | + Ensemble + advanced postprocessing | ~2.5 hours |

---

## Step 4: Start Training

### Simple Level (Recommended for First Run)
```bash
# Activate environment
conda activate dlhw3

# Train with simple config
python train.py --config configs/simple_config.json
```

### Medium Level
```bash
python train.py --config configs/medium_config.json
```

### Strong Level
```bash
python train.py --config configs/strong_config.json
```

### Boss Level (Requires more GPU memory)
```bash
python train.py --config configs/boss_config.json
```

### Using the Shell Script
```bash
# Make script executable (first time only)
chmod +x run_training.sh

# Run training
./run_training.sh simple   # or medium, strong, boss
```

---

## Step 5: Monitor Training

While training, you'll see:

```
Training Epoch 1: 100%|████████████| 2319/2319 [07:45<00:00, 4.98it/s, loss=1.2345, avg_loss=1.3456]
Evaluating Epoch 1: 100%|████████████| 168/168 [00:32<00:00, 5.19it/s]
Epoch 1 Metrics: {'f1': 78.45, 'exact_match': 65.32, 'total_questions': 5351}
```

**What to watch for:**
- Loss should decrease over time
- F1 score should increase
- Training completes without errors

**Optional: Use Weights & Biases (wandb)**
1. Set `"use_wandb": true` in your config
2. Login: `wandb login`
3. View training at: https://wandb.ai

---

## Step 6: Evaluate Your Model

After training, evaluate on test data:

```bash
python evaluate.py \
    --model_path output/best_model_epoch_3 \
    --test_data data/test.json \
    --config configs/simple_config.json \
    --output_file predictions.json
```

### Using the Shell Script
```bash
# Make script executable (first time only)
chmod +x run_evaluation.sh

# Run evaluation
./run_evaluation.sh
```

### Output
```
Running inference...
100%|████████████| 168/168 [00:45<00:00, 3.71it/s]
Postprocessing predictions...
Predictions saved to predictions.json
Evaluation Results:
F1 Score: 78.45%
Exact Match: 65.32%
Total Questions: 5351
```

---

## Step 7: Make Single Predictions (Interactive Mode)

Test your model interactively:

```bash
python evaluate.py \
    --model_path output/best_model_epoch_3 \
    --test_data data/test.json \
    --config configs/simple_config.json \
    --single_prediction
```

Then enter your question and context:
```
Enter question: What is the capital of France?
Enter context: Paris is the capital and most populous city of France...

Predicted Answer: Paris
Confidence: 15.4523
```

---

## Complete Workflow Example

Here's a complete example from start to finish:

```bash
# 1. Setup environment (first time only)
cd /home/ailab-students/DLHW3
conda env create -f environment.yml
conda activate dlhw3

# 2. Verify installation
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 3. Prepare data (assuming you have the files)
mkdir -p data
# Copy your train.json, dev.json, test.json to data/

# 4. Quick test with simple config
python train.py --config configs/simple_config.json

# 5. Wait for training to complete (~8 minutes with FP16 on T4 GPU)
# Watch the progress bars and metrics

# 6. Evaluate the best model
python evaluate.py \
    --model_path output/best_model_epoch_3 \
    --test_data data/test.json \
    --config configs/simple_config.json \
    --output_file predictions.json

# 7. Check predictions
head -20 predictions.json
```

---

## Advanced Usage

### Modify Training Parameters

Edit the config file before training:

```bash
nano configs/simple_config.json
```

Key parameters to adjust:
- `batch_size`: Reduce if GPU out of memory (e.g., 8, 4)
- `num_epochs`: Increase for better training (e.g., 5, 10)
- `learning_rate`: Tune for better convergence (e.g., 1e-5, 5e-5)
- `gradient_accumulation_steps`: Increase if reducing batch_size
- `use_fp16`: Set to `false` if GPU doesn't support FP16

### Continue Training from Checkpoint

```bash
# Training automatically saves checkpoints in output/
# To resume, just point to the checkpoint directory
python train.py --config configs/medium_config.json --resume_from output/checkpoint_epoch_2
```

### Train with Multiple GPUs

```bash
# The Accelerator handles multi-GPU automatically
# Just make sure both GPUs are visible
CUDA_VISIBLE_DEVICES=0,1 python train.py --config configs/boss_config.json
```

### Train on CPU (Slow, Not Recommended)

```bash
# Force CPU usage
CUDA_VISIBLE_DEVICES="" python train.py --config configs/simple_config.json
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Option 1: Reduce batch size
# Edit config: "batch_size": 8  (or 4)

# Option 2: Increase gradient accumulation
# Edit config: "gradient_accumulation_steps": 4

# Option 3: Enable FP16 if not already
# Edit config: "use_fp16": true

# Option 4: Reduce max_length
# Edit config: "max_length": 384
```

### Issue 2: Import Errors

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution:**
```bash
# Make sure environment is activated
conda activate dlhw3

# Reinstall packages
pip install -r requirements.txt
```

### Issue 3: Data Not Found

**Error**: `FileNotFoundError: [Errno 2] No such file or directory: 'data/train.json'`

**Solution:**
```bash
# Create data directory
mkdir -p data

# Copy your data files
cp /path/to/your/train.json data/
cp /path/to/your/dev.json data/
cp /path/to/your/test.json data/
```

### Issue 4: Slow Training

**Problem**: Training is very slow

**Solutions:**
1. Enable FP16: Set `"use_fp16": true` in config
2. Check GPU usage: `nvidia-smi` (should show GPU utilization)
3. Increase batch size if you have memory
4. Check data loading: Set `"num_workers": 4` or higher

### Issue 5: Poor Performance

**Problem**: Low F1 score or accuracy

**Solutions:**
1. Train for more epochs
2. Try different model: Use `configs/strong_config.json`
3. Adjust learning rate: Try 2e-5 or 5e-5
4. Enable early stopping: `"early_stopping": true`
5. Use ensemble: `configs/boss_config.json`

---

## Directory Structure After Running

```
DLHW3/
├── data/
│   ├── train.json
│   ├── dev.json
│   └── test.json
├── output/                          # Created during training
│   ├── best_model_epoch_3/
│   │   ├── pytorch_model.bin
│   │   ├── config.json
│   │   └── tokenizer files...
│   ├── checkpoint_epoch_1/
│   └── final_model/
├── logs/                            # Training logs (if configured)
├── predictions.json                 # Evaluation output
└── wandb/                           # Weights & Biases logs (if enabled)
```

---

## Performance Benchmarks

Expected performance on Spoken-SQuAD test set:

| Config | F1 Score | EM Score | Training Time (T4 FP16) |
|--------|----------|----------|-------------------------|
| Simple | ~70-75% | ~60-65% | 8 minutes |
| Medium | ~75-80% | ~65-70% | 8 minutes |
| Strong | ~80-85% | ~70-75% | 25 minutes |
| Boss | ~85-90% | ~75-80% | 2.5 hours |

*Note: Actual performance depends on data quality and hyperparameters*

---

## Quick Reference Commands

```bash
# Setup
conda env create -f environment.yml
conda activate dlhw3

# Train
python train.py --config configs/simple_config.json

# Evaluate
python evaluate.py --model_path output/best_model_epoch_3 \
                   --test_data data/test.json \
                   --config configs/simple_config.json

# Interactive prediction
python evaluate.py --model_path output/best_model_epoch_3 \
                   --test_data data/test.json \
                   --config configs/simple_config.json \
                   --single_prediction

# Check GPU
nvidia-smi

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check logs
tail -f logs/training.log  # if logging to file
```

---

## Next Steps

1. ✅ Setup environment
2. ✅ Prepare data
3. ✅ Run simple config training
4. ✅ Evaluate model
5. 🎯 Try medium config for better results
6. 🎯 Experiment with strong config
7. 🎯 Fine-tune hyperparameters
8. 🎯 Try boss config for best performance

---

## Need Help?

- Check `IMPROVEMENTS_GUIDE.md` for implementation details
- Check `SLIDE_REQUIREMENTS.md` for slide-to-code mapping
- Check `SETUP_GUIDE.md` for installation troubleshooting
- Check error messages carefully - they usually indicate the issue

Good luck with your homework! 🎓🚀
