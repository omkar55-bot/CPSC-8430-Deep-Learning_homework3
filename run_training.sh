#!/bin/bash

# Spoken-SQuAD Question Answering Training Script
# Usage: ./run_training.sh [simple|medium|strong|boss]

LEVEL=${1:-simple}
echo "Running training for level: $LEVEL"

# Create necessary directories
mkdir -p output
mkdir -p data
mkdir -p logs

# Set config file based on level
CONFIG_FILE="configs/${LEVEL}_config.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Config file $CONFIG_FILE not found!"
    exit 1
fi

echo "Using config: $CONFIG_FILE"

# Run training with error handling
python train.py --config $CONFIG_FILE 2>&1 | tee logs/train_${LEVEL}_$(date +%Y%m%d_%H%M%S).log

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
    echo "Model saved in output/ directory"
else
    echo "Training failed! Check logs for details."
    exit 1
fi
