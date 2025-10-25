# Spoken-SQuAD Question Answering - HW3

This project implements a BERT-based extractive question answering system for the Spoken-SQuAD dataset, which contains question-answer pairs derived from spoken audio with various noise levels.

## 🎯 Task Overview

- **Task**: Extractive Question Answering (EQA) similar to SQuAD
- **Dataset**: Spoken-SQuAD with 37,111 training and 5,351 testing question-answer pairs
- **Challenge**: Handle long paragraphs with BERT's 512 token limit using sliding windows
- **Goal**: Predict start and end positions of answer spans in the context

## 📊 Dataset Details

- **Training Set**: 37,111 QA pairs (WER: 22.77%)
- **Testing Set**: 5,351 QA pairs with noise variations:
  - No noise: 22.73% WER
  - Noise V1: 44.22% WER  
  - Noise V2: 54.82% WER

## 🏗️ Project Structure

```
DLHW3/
├── configs/                 # Configuration files for different levels
│   ├── simple_config.json   # Basic configuration
│   ├── medium_config.json   # With learning rate decay + doc_stride optimization
│   ├── strong_config.json   # Better preprocessing + other models
│   └── boss_config.json     # Ensemble + advanced postprocessing
├── data/                    # Dataset directory
│   ├── train.json          # Training data
│   ├── dev.json            # Development data  
│   └── test.json           # Test data
├── output/                  # Model outputs and checkpoints
├── logs/                    # Training logs
├── evaluation_results/      # Evaluation outputs
├── dataset.py              # Dataset loading and preprocessing
├── model.py                # BERT-based QA model definitions
├── train.py                # Training script
├── evaluate.py             # Evaluation script
├── utils.py                # Utility functions
├── run_training.sh         # Training execution script
├── run_evaluation.sh       # Evaluation execution script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x run_training.sh run_evaluation.sh
```

### 2. Prepare Data

Place your Spoken-SQuAD data files in the `data/` directory:
- `train.json` - Training data
- `dev.json` - Development/validation data  
- `test.json` - Test data

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

### 3. Training

Choose your difficulty level and run training:

```bash
# Simple level (Sample code)
./run_training.sh simple

# Medium level (Linear LR decay + doc_stride optimization) 
./run_training.sh medium

# Strong level (Better preprocessing + other pretrained models)
./run_training.sh strong

# Boss level (Ensemble + advanced postprocessing)
./run_training.sh boss
```

### 4. Evaluation

```bash
# Evaluate trained model
./run_evaluation.sh output/best_model_epoch_3 data/test.json configs/simple_config.json

# Interactive prediction mode
python evaluate.py --model_path output/best_model_epoch_3 --config configs/simple_config.json --single_prediction
```

## 📈 Performance Levels

### 🟢 Simple Level
- **Implementation**: Basic BERT fine-tuning with sliding windows
- **Training Time**: ~7-40 minutes
- **Features**:
  - Standard BERT-base-chinese model
  - Basic sliding window approach for long contexts
  - Default hyperparameters

### 🟡 Medium Level  
- **Improvements**: Linear learning rate decay + doc_stride optimization
- **Training Time**: ~7-40 minutes
- **Features**:
  - Linear learning rate scheduling with warmup
  - Optimized `doc_stride` parameter for better window overlap
  - Gradient accumulation

### 🟠 Strong Level
- **Improvements**: Better preprocessing + alternative pretrained models
- **Training Time**: ~20 minutes - 2 hours
- **Features**:
  - Advanced Chinese models (ELECTRA, RoBERTa-wwm)  
  - Improved text preprocessing
  - Enhanced sliding window strategy
  - Mixed precision training (FP16)

### 🔴 Boss Level
- **Improvements**: Ensemble methods + advanced postprocessing
- **Training Time**: ~2-12.5 hours
- **Features**:
  - Multi-model ensemble (ELECTRA + RoBERTa + BERT)
  - Sophisticated answer postprocessing
  - Advanced hyperparameter optimization
  - Cross-validation techniques

## 🛠️ Key Technical Features

### Sliding Window Strategy
- **Training**: Center windows around known answer positions
- **Testing**: Generate overlapping windows with configurable stride
- **Window Size**: Dynamically calculated based on question length

### Model Architecture
- **Base Models**: Any HuggingFace transformer (BERT, ELECTRA, RoBERTa)
- **Task Head**: Linear layers for start/end position prediction
- **Ensemble**: Weighted combination of multiple models (Boss level)

### Advanced Preprocessing  
- **Tokenization**: Proper handling of Chinese text
- **Answer Alignment**: Robust character-to-token mapping
- **Context Truncation**: Smart windowing preserving answer spans

### Training Optimizations
- **Mixed Precision**: FP16 training for faster convergence
- **Gradient Accumulation**: Handle larger effective batch sizes
- **Learning Rate Scheduling**: Warmup + linear decay
- **Early Stopping**: Prevent overfitting

## 📊 Evaluation Metrics

- **F1 Score**: Token-level overlap between prediction and ground truth
- **Exact Match**: Exact string match after normalization
- **Word Error Rate (WER)**: For speech recognition quality assessment

## 🔧 Configuration

Each performance level has its own configuration file with optimized hyperparameters:

- **Learning Rate**: 5e-6 to 3e-5 depending on model and level
- **Batch Size**: 8-16 with gradient accumulation  
- **Doc Stride**: 16-128 tokens for window overlap
- **Max Length**: 512 tokens (BERT limit)
- **Epochs**: 3-15 depending on complexity

## 📝 Usage Tips

### For Better Performance:
1. **Data Quality**: Ensure proper text preprocessing and answer alignment
2. **Model Selection**: Try different Chinese models from HuggingFace  
3. **Hypertuning**: Adjust `doc_stride` and learning rate for your dataset
4. **Ensemble**: Combine multiple models for Boss-level performance
5. **Postprocessing**: Implement null answer detection and confidence scoring

### Common Issues:
- **OOM Errors**: Reduce batch size or enable gradient accumulation
- **Poor F1**: Check answer span alignment and preprocessing
- **Slow Training**: Enable FP16 mixed precision training
- **Long Contexts**: Optimize doc_stride parameter

## 🎖️ Estimated Training Times

| Level | K80 | T4 | T4 (FP16) | P100 | V100 |
|-------|-----|----|---------| -----|------|
| Simple| 40m | 20m| 8m      | 10m  | 7m   |
| Medium| 40m | 20m| 8m      | 10m  | 7m   |
| Strong| 2h  | 1h | 25m     | 35m  | 20m  |
| Boss  |12.5h| 6h | 2.5h    | 4.5h | 2h   |

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [Spoken-SQuAD Dataset](https://github.com/chiahsuan156/Spoken-SQuAD)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [SQuAD Dataset](https://rajpurkar.github.io/SQuAD-explorer/)

---

**Good luck with your implementation! 🚀**

For questions or issues, please check the training logs in the `logs/` directory or review the configuration files.
