#!/usr/bin/env python3
"""
Generate training report from saved models and outputs
Use this to create plots and metrics for your homework report
"""

import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def collect_training_metrics(output_dir='output'):
    """Collect metrics from all checkpoints"""
    metrics = []
    
    # Look for checkpoint directories
    for checkpoint_dir in sorted(Path(output_dir).glob('*_epoch_*')):
        epoch_num = checkpoint_dir.name.split('_')[-1]
        config_path = checkpoint_dir / 'config.json'
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Extract any metrics saved in config
                if 'metrics' in config:
                    metrics.append({
                        'epoch': int(epoch_num),
                        **config['metrics']
                    })
    
    return pd.DataFrame(metrics) if metrics else None

def plot_training_curves(df, save_path='training_report.png'):
    """Create training curve plots"""
    if df is None or len(df) == 0:
        print("No metrics found to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot F1 Score
    if 'f1' in df.columns:
        axes[0, 0].plot(df['epoch'], df['f1'], marker='o', label='F1 Score')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('F1 Score (%)')
        axes[0, 0].set_title('F1 Score over Epochs')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
    
    # Plot Exact Match
    if 'exact_match' in df.columns:
        axes[0, 1].plot(df['epoch'], df['exact_match'], marker='s', color='orange', label='Exact Match')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Exact Match (%)')
        axes[0, 1].set_title('Exact Match over Epochs')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
    
    # Plot Loss (if available)
    if 'loss' in df.columns:
        axes[1, 0].plot(df['epoch'], df['loss'], marker='^', color='red', label='Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Training Loss over Epochs')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
    
    # Summary table
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    ================
    Total Epochs: {df['epoch'].max()}
    Best F1: {df['f1'].max():.2f}% (Epoch {df.loc[df['f1'].idxmax(), 'epoch']})
    Best EM: {df['exact_match'].max():.2f}% (Epoch {df.loc[df['exact_match'].idxmax(), 'epoch']})
    Final F1: {df.iloc[-1]['f1']:.2f}%
    Final EM: {df.iloc[-1]['exact_match']:.2f}%
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, family='monospace')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Training curves saved to {save_path}")
    return fig

def generate_report(output_dir='output', report_file='training_report.md'):
    """Generate a markdown report"""
    
    # Collect metrics
    df = collect_training_metrics(output_dir)
    
    if df is None:
        print("⚠️ No metrics found. Make sure training has completed at least 1 epoch.")
        return
    
    # Create plots
    plot_training_curves(df)
    
    # Generate markdown report
    report = f"""# Training Report - Spoken SQuAD QA

## Training Configuration
- Model: BERT-base-chinese
- Total Epochs: {df['epoch'].max()}
- Dataset: Spoken-SQuAD

## Results Summary

### Best Performance
- **Best F1 Score**: {df['f1'].max():.2f}% (Epoch {df.loc[df['f1'].idxmax(), 'epoch']})
- **Best Exact Match**: {df['exact_match'].max():.2f}% (Epoch {df.loc[df['exact_match'].idxmax(), 'epoch']})

### Final Performance (Epoch {df.iloc[-1]['epoch']})
- **F1 Score**: {df.iloc[-1]['f1']:.2f}%
- **Exact Match**: {df.iloc[-1]['exact_match']:.2f}%

## Training Curves

![Training Curves](training_report.png)

## Metrics by Epoch

{df.to_markdown(index=False)}

## Files Generated
- Model checkpoints: `output/checkpoint_epoch_X/`
- Best model: `output/best_model_epoch_X/`
- Training curves: `training_report.png`
- This report: `training_report.md`

## How to Use the Model

```bash
# Evaluate on test set
python evaluate.py \\
    --model_path output/best_model_epoch_{df.loc[df['f1'].idxmax(), 'epoch']} \\
    --test_data data/test.json \\
    --config configs/simple_config.json \\
    --output_file predictions.json
```

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report saved to {report_file}")
    print(f"\nSummary:")
    print(f"  Best F1: {df['f1'].max():.2f}%")
    print(f"  Best EM: {df['exact_match'].max():.2f}%")

if __name__ == '__main__':
    generate_report()
