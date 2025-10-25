# Model Weights

Due to GitHub's file size limitations (100MB max), the trained model weights are **NOT** included in this repository.

## Model Information

This project trained 4 different configurations:

1. **Simple** - BERT-base-chinese, 3 epochs
   - F1 Score: 46.04%, Exact Match: 32.20%
   - Model size: ~400MB

2. **Medium** - BERT-base-chinese, 5 epochs  
   - F1 Score: 46.02%, Exact Match: 32.03%
   - Model size: ~400MB

3. **Strong** - ELECTRA-base-chinese, 8 epochs (BEST)
   - F1 Score: 53.80%, Exact Match: 38.70%
   - Model size: ~388MB

4. **Boss** - Ensemble of 3 ELECTRA models, 15 epochs each
   - F1 Score: 50.96%, Exact Match: 35.08%
   - Total size: ~1.2GB (3 models)

## How to Get the Models

### Option 1: Train from Scratch (Recommended)

Follow the instructions in `HOW_TO_RUN.md` to train the models:

```bash
# Setup environment
conda env create -f environment.yml
conda activate dlhw3

# Download Spoken-SQuAD dataset
# Place in data/ directory

# Train models
python train.py --config configs/simple_config.json
python train.py --config configs/medium_config.json  
python train.py --config configs/strong_config.json
python train.py --config configs/boss_config.json
```

### Option 2: Download Pre-trained Weights

Pre-trained model weights can be downloaded from:
- [Google Drive Link] (if available)
- [Hugging Face Hub] (if uploaded)

Place the downloaded models in:
```
model_weights/
├── output/
│   ├── simple/
│   │   └── final_model/
│   ├── medium/
│   │   └── final_model/
│   ├── strong/
│   │   └── final_model/
│   └── boss/
│       └── final_model/
```

## Expected Directory Structure

Each `final_model/` directory should contain:
- `config.json` - Model configuration
- `model.safetensors` - Model weights (~388-400MB)
- `tokenizer.json` - Tokenizer configuration
- `tokenizer_config.json` - Tokenizer settings
- `vocab.txt` - Vocabulary file
- `special_tokens_map.json` - Special tokens

For Boss ensemble, there are 3 subdirectories: `model_0/`, `model_1/`, `model_2/`

## Storage Requirements

- Simple: ~400MB
- Medium: ~400MB
- Strong: ~388MB
- Boss: ~1.2GB (3 models)
- **Total**: ~2.4GB

## Results Summary

All models were evaluated on 5,351 test questions from Spoken-SQuAD dataset.

See `RESULTS_SUMMARY.txt` for detailed performance analysis.
