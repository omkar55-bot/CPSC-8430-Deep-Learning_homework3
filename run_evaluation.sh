#!/bin/bash

# Spoken-SQuAD Question Answering Evaluation Script
# Usage: ./run_evaluation.sh [model_path] [test_data] [config]

MODEL_PATH=${1:-"output/best_model_epoch_3"}
TEST_DATA=${2:-"data/test.json"}
CONFIG=${3:-"configs/simple_config.json"}

echo "Evaluating model: $MODEL_PATH"
echo "Test data: $TEST_DATA"
echo "Config: $CONFIG"

# Create output directory
mkdir -p evaluation_results

# Run evaluation
python evaluate.py \
    --model_path $MODEL_PATH \
    --test_data $TEST_DATA \
    --config $CONFIG \
    --output_file evaluation_results/predictions_$(date +%Y%m%d_%H%M%S).json

echo "Evaluation completed!"
echo "Results saved in evaluation_results/"
