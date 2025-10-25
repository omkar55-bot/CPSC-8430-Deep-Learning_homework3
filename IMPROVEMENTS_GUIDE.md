# DLHW3 Code Improvements Guide

This document outlines all the improvements made to the Spoken-SQuAD Question Answering project based on the course slides.

## 📋 Summary of Improvements

### 1. **Doc Stride Optimization** (Medium Level) ✅
**Location**: All config files, `dataset.py`

**What was changed**:
- Changed `doc_stride` from 128/64/32/16 to **150** in all configurations
- This is the distance between the start positions of two consecutive windows

**Why it matters**:
- Better handling of overlapping windows for long documents
- Reduces the chance of answers being split across windows
- According to the slide: `self.doc_stride = 150` is recommended

**Files modified**:
- `configs/simple_config.json`: doc_stride = 150
- `configs/medium_config.json`: doc_stride = 150
- `configs/strong_config.json`: doc_stride = 150
- `configs/boss_config.json`: doc_stride = 150
- `dataset.py`: Default parameter changed to 150

---

### 2. **Linear Learning Rate Decay** (Medium Level) ✅
**Location**: `train.py`

**What was implemented**:
- Added comprehensive documentation for two methods of learning rate decay
- Method 1: Manual adjustment by decrementing `optimizer.param_groups[0]["lr"]`
- Method 2: Automatic using `get_linear_schedule_with_warmup` (Already implemented)

**Code example added**:
```python
# Method 1: Manual decay
for i in range(total_step):
    optimizer.param_groups[0]["lr"] -= learning_rate / total_step

# Method 2: Scheduler (Recommended - already implemented)
scheduler = get_linear_schedule_with_warmup(optimizer, ...)
for i in range(total_step):
    optimizer.step()
    scheduler.step()
```

**Why it matters**:
- Prevents overfitting by gradually reducing learning rate
- The graph in the slide shows learning rate decreasing linearly to zero
- Helps model converge to better minima

---

### 3. **Preprocessing Improvements** (Strong Level) ✅
**Location**: `dataset.py` - `_create_training_windows()` method

**What was changed**:
- Improved window creation to **center answers in the middle of windows**
- Prevents model from learning that answers are always near boundaries

**Key improvement**:
```python
# OLD: Simple window creation around answer
window_start = max(0, answer_token_start - available_context_len // 2)

# NEW: Properly center answer in window
mid = (answer_start_token + answer_end_token) // 2
paragraph_start = max(0, min(mid - self.max_length // 2, 
                             len(tokenized_paragraph['input_ids']) - self.max_length))
```

**Why it matters**:
- The slide asks: "How to prevent model from learning something it should not learn during training?"
- Answer: Answers are not always near the middle of window in real scenarios
- By centering answers during training, we avoid creating artificial patterns

---

### 4. **Postprocessing Improvements** (Boss Level) ✅
**Location**: `utils.py` - `get_best_answer_from_window()` method

**What was changed**:
- Added explicit handling for cases where `end_index < start_index`
- Added documentation highlighting this bug

**Key improvement**:
```python
# POSTPROCESSING IMPROVEMENT: Handle cases where end < start
# Check if this is a valid span - if end < start, skip it
if end_idx < start_idx:
    continue  # Skip invalid spans instead of using them
```

**Why it matters**:
- The slide explicitly warns: "what if predicted end_index < predicted start_index?"
- Without this check, you might extract invalid/reversed answer spans
- This is a common bug that can significantly hurt performance

---

### 5. **Automatic Mixed Precision (FP16)** Training ✅
**Location**: `train.py`

**What was documented**:
- Added comprehensive explanation of FP16 training
- Explained benefits: 1.5-3.0x speedup while maintaining accuracy
- Showed how to use with Accelerate library

**Implementation**:
```python
# Change fp16_training to True in config
self.accelerator = Accelerator(
    mixed_precision='fp16' if config.get('use_fp16', False) else 'no',
    gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
)
```

**Why it matters**:
- Slide shows training time reduction: Simple (20m→8m), Medium (20m→8m), Strong (1h→25m), Boss (6h→2.5h)
- Warning: Only works on some GPUs (e.g., T4, V100)
- Must install: `pip install accelerate==0.2.0`

---

### 6. **Gradient Accumulation** ✅
**Location**: `train.py`

**What was documented**:
- Added explanation of gradient accumulation technique
- Shows how it enables larger effective batch sizes

**Implementation**:
```python
# Use when GPU memory is not enough but you want larger batch size
for batch_idx, batch in enumerate(progress_bar):
    with self.accelerator.accumulate(self.model):
        # Forward, backward, but don't update yet
        loss.backward()
        # Update only after accumulation_steps batches
```

**Why it matters**:
- Allows training with larger effective batch sizes
- Splits global batch into smaller mini-batches
- Accumulates gradients without updating parameters until all mini-batches processed

---

## 🎯 Configuration Guide

### Simple Config (Sample Code)
- Basic settings
- doc_stride: 150
- 3 epochs
- FP16: enabled
- Good for testing

### Medium Config (Linear LR Decay + Doc Stride)
- doc_stride: 150 (optimized)
- Linear learning rate scheduler: enabled
- 5 epochs
- Gradient accumulation: 2 steps
- Early stopping: enabled

### Strong Config (Better Preprocessing + Other Models)
- Different pretrained model: `hfl/chinese-electra-180g-base-discriminator`
- doc_stride: 150
- Improved preprocessing (centered windows)
- 8 epochs
- Higher dropout (0.2) for regularization

### Boss Config (Ensemble + Advanced Postprocessing)
- Ensemble of 3 models
- doc_stride: 150
- All postprocessing improvements
- 15 epochs
- Gradient accumulation: 8 steps
- Maximum answer length: 100

---

## 📝 TODO Checklist

Based on the slides, here's what you need to verify/complete:

- [x] **Doc Stride**: Changed to 150 in all configs
- [x] **Linear LR Decay**: Documented and implemented
- [x] **Preprocessing**: Center answers in windows during training
- [x] **Postprocessing**: Handle end_index < start_index
- [x] **FP16 Training**: Documented and enabled in configs
- [x] **Gradient Accumulation**: Documented and configured

---

## 🚀 How to Use

1. **Install dependencies** (including accelerate for FP16):
```bash
pip install -r requirements.txt
pip install accelerate==0.2.0
```

2. **Train with different levels**:
```bash
# Simple
./run_training.sh simple

# Medium (with LR decay + doc_stride=150)
./run_training.sh medium

# Strong (better preprocessing + different models)
./run_training.sh strong

# Boss (ensemble + all improvements)
./run_training.sh boss
```

3. **Evaluate**:
```bash
./run_evaluation.sh
```

---

## 🔍 Key Insights from Slides

1. **Doc Stride = 150**: Optimal value for window overlap
2. **Center Answers**: Prevents learning artificial boundary patterns
3. **Check Start/End Order**: Critical postprocessing bug fix
4. **FP16 Training**: Massive speedup with minimal accuracy loss
5. **Gradient Accumulation**: Enables larger batch sizes on limited GPU

---

## 📚 References

- [Accelerate Documentation](https://huggingface.co/docs/accelerate/)
- [Intro to Native PyTorch Automatic Mixed Precision](https://pytorch.org/docs/stable/amp.html)
- [Gradient Accumulation in PyTorch](https://kozodoi.me/blog/20210219/gradient-accumulation)
- [Hugging Face Learning Rate Schedulers](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)

---

## ⚠️ Important Notes

1. **GPU Compatibility**: FP16 training only works on certain GPUs (T4, V100, etc.)
2. **Memory Management**: Use gradient accumulation if running out of GPU memory
3. **Answer Centering**: Only applied during training, not evaluation
4. **Postprocessing**: Always validate start < end in predictions
5. **Doc Stride**: 150 is recommended, but you can experiment with other values

---

Good luck with your homework! 🎓
