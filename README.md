# CPSC-8430 Deep Learning - Homework 3
## Spoken-SQuAD Question Answering System

A transformer-based extractive Question Answering system for the Spoken-SQuAD dataset (Chinese).

## 📊 Results

| Model  | F1 Score | Exact Match | Training Time |
|--------|----------|-------------|---------------|
| Simple | 46.04%   | 32.20%      | ~45 min       |
| Medium | 46.02%   | 32.03%      | ~1.5 hours    |
| **Strong** | **53.80%** ⭐ | **38.70%** ⭐ | **~3 hours** |
| Boss   | 50.96%   | 35.08%      | ~12 hours     |

**Winner**: Strong configuration (ELECTRA-base with FP16 mixed precision)

## 🎯 Project Overview

This project implements an extractive Question Answering system using:
- **Models**: BERT-base-chinese and ELECTRA-base-chinese
- **Dataset**: Spoken-SQuAD (33,677 train, 3,434 dev, 5,351 test questions)
- **Framework**: PyTorch + Transformers + Accelerate
- **Techniques**: FP16 training, gradient accumulation, ensemble learning

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda env create -f environment.yml
conda activate dlhw3

# Or use pip
pip install -r requirements.txt
```

### 2. Download Dataset

Download Spoken-SQuAD dataset and place in `data/` directory:
```
data/
├── train.json
├── dev.json
└── test.json
```

### 3. Train Models

```bash
# Simple configuration (3 epochs, BERT)
python train.py --config configs/simple_config.json

# Strong configuration (8 epochs, ELECTRA, FP16) - BEST
python train.py --config configs/strong_config.json

# Boss configuration (15 epochs, ensemble)
python train.py --config configs/boss_config.json
```

### 4. Evaluate Models

```bash
python evaluate.py \
    --model_path output/strong/final_model \
    --test_data data/test.json \
    --config configs/strong_config.json \
    --output_file predictions.json
```

## 📁 Project Structure

```
DLHW3/
├── train.py                    # Training script
├── evaluate.py                 # Evaluation and inference
├── model.py                    # Model definitions
├── dataset.py                  # Dataset loading and preprocessing
├── utils.py                    # Metrics and postprocessing
├── configs/                    # Configuration files
│   ├── simple_config.json
│   ├── medium_config.json
│   ├── strong_config.json
│   └── boss_config.json
├── data/                       # Dataset (not included)
├── model_weights/              # Trained models (not included - too large)
├── PROJECT_REPORT.txt          # Detailed project report
├── RESULTS_SUMMARY.txt         # Results analysis
└── QUICK_REFERENCE.txt         # Quick reference guide
```

## 🔧 Key Features

### Preprocessing Improvements
- ✅ Sliding window approach with `doc_stride=150`
- ✅ Answer centering in training windows
- ✅ Proper handling of long contexts (>512 tokens)
- ✅ SQuAD format data processing

### Training Optimizations
- ✅ Mixed precision training (FP16) with Accelerate
- ✅ Gradient accumulation for larger effective batch sizes
- ✅ Linear learning rate scheduling with warmup
- ✅ Early stopping based on validation F1
- ✅ Weights & Biases experiment tracking

### Postprocessing Enhancements
- ✅ Answer span validation (end_idx >= start_idx)
- ✅ Tuned `max_answer_length=30` tokens
- ✅ Multi-window answer selection
- ✅ Top-K candidate evaluation

## 📝 Configurations

### Simple (Baseline)
- Model: BERT-base-chinese
- Epochs: 3
- Batch size: 16
- Learning rate: 3e-5
- FP16: No

### Medium (+ LR Scheduling)
- Model: BERT-base-chinese
- Epochs: 5
- Gradient accumulation: 2
- LR scheduler: Linear warmup + decay

### Strong (Best Performance) ⭐
- Model: ELECTRA-base-chinese
- Epochs: 8
- FP16: Yes
- Gradient accumulation: 4
- Dropout: 0.2

### Boss (Ensemble)
- Model: 3× ELECTRA-base-chinese
- Epochs: 15 per model
- Ensemble: Average predictions
- Very high regularization

## 📊 Detailed Results

See `RESULTS_SUMMARY.txt` for comprehensive analysis including:
- Performance breakdown by configuration
- Hyperparameter comparison
- Why Strong beat Boss ensemble
- Preprocessing/postprocessing improvements
- Challenges and solutions

## 🔍 Key Insights

1. **Model Architecture Matters Most**: ELECTRA outperformed BERT by +7.76% F1
2. **Strong Single Model Beat Ensemble**: Proper tuning > complexity
3. **Preprocessing is Critical**: Sliding windows and answer centering improved results
4. **Postprocessing Validation Essential**: Fixed invalid span predictions
5. **FP16 Training**: Faster training without accuracy loss

## 📚 Documentation

- `PROJECT_REPORT.txt` - Complete project documentation
- `RESULTS_SUMMARY.txt` - Results and performance analysis
- `QUICK_REFERENCE.txt` - Commands and quick reference
- `HOW_TO_RUN.md` - Detailed running instructions
- `IMPROVEMENTS_GUIDE.md` - All improvements explained

## 🐛 Issues Fixed

1. ✅ Gradient clipping bug with FP16 training
2. ✅ Model loading with safetensors format
3. ✅ Ensemble model loading (3 separate models)
4. ✅ max_answer_length tuning (100 → 30 tokens)
5. ✅ Invalid answer span validation

## 💡 Requirements

```
Python 3.9+
PyTorch 2.0+
Transformers 4.30+
Accelerate 0.20+
CUDA 11.8+ (for GPU training)
```

See `requirements.txt` for complete list.

## 🎓 Course Information

- **Course**: CPSC-8430 Deep Learning
- **Assignment**: Homework 3 - Question Answering
- **Dataset**: Spoken-SQuAD (Chinese)
- **Task**: Extractive Question Answering

## 📖 References

- [SQuAD Paper](https://arxiv.org/abs/1606.05250)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [ELECTRA Paper](https://arxiv.org/abs/2003.10555)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## ⚠️ Model Weights Not Included

Due to GitHub's file size limitations (100MB max), trained model weights (~2.4GB total) are **not** included in this repository. 

To use the models:
1. Train from scratch using provided configs (recommended)
2. Download pre-trained weights if available separately

See `model_weights/README.md` for details.

## 📄 License

This is a student project for educational purposes.

## 🙏 Acknowledgments

- Hugging Face for Transformers library
- Spoken-SQuAD dataset creators
- Course instructors and TAs

---

**Best Performance**: Strong configuration achieved **53.80% F1** and **38.70% EM** on 5,351 test questions.
