import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from dataset import SpokenSquadDataset, collate_fn
from model import SpokenSquadModel, EnsembleModel
from utils import compute_metrics, postprocess_predictions
import json
import os
from tqdm import tqdm
import wandb
from accelerate import Accelerator
import argparse
from typing import Dict, List

class Trainer:
    """Training class for Spoken-SQuAD QA"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        
        # Initialize accelerator for mixed precision and multi-GPU
        # ##### Training Tip: Automatic mixed precision #####
        # PyTorch trains with 32-bit floating point (FP32) arithmetic by default
        # Automatic Mixed Precision (AMP) enables automatic conversion of
        # certain GPU operations from FP32 precision to half-precision (FP16)
        # Offer about 1.5-3.0x speed up while maintaining accuracy
        # Change "fp16_training" to True to support mixed precision training (fp16)
        # fp16_training = False (current setting)
        # if fp16_training:
        #     pip install accelerate==0.2.0
        #     from accelerate import Accelerator
        #     accelerator = Accelerator(fp16=True)
        # Documentation for the toolkit: https://huggingface.co/docs/accelerate/
        # if fp16_training:
        #     model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
        # if fp16_training:
        #     accelerator.backward(output.loss)
        # else:
        #     output.loss.backward()
        # Reference:
        #     accelerate documentation
        #     intro to native pytorch automatic mixed precision
        # Warning: only work on some gpu (e.g. T4, V100)
        
        self.accelerator = Accelerator(
            mixed_precision='fp16' if config.get('use_fp16', False) else 'no',
            gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1)
        )
        
        # Initialize wandb for tracking
        if config.get('use_wandb', False):
            wandb.init(project="spoken-squad", config=config)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        
        # Initialize model
        if config.get('use_ensemble', False):
            self.model = EnsembleModel(config['ensemble_models'])
        else:
            self.model = SpokenSquadModel(
                model_name=config['model_name'],
                dropout=config.get('dropout', 0.1)
            )
        
        # Setup datasets
        self.train_dataset = SpokenSquadDataset(
            config['train_data_path'],
            self.tokenizer,
            max_length=config['max_length'],
            doc_stride=config.get('doc_stride', 128),
            mode='train'
        )
        
        self.val_dataset = SpokenSquadDataset(
            config['val_data_path'],
            self.tokenizer,
            max_length=config['max_length'],
            doc_stride=config.get('doc_stride', 128),
            mode='test'
        )
        
        # Setup data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4)
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['eval_batch_size'],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=config.get('num_workers', 4)
        )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Setup scheduler (Medium level improvement)
        # ##### TODO: Apply linear learning rate decay #####
        # Method 1: Adjust learning rate manually
        # Decrement optimizer.param_groups[0]["lr"] by learning_rate / total_training_step per step
        # This block is only an example! You only need to add 1 or 2 lines
        # learning_rate = 1e-4
        # optimizer = AdamW(model.parameters(), lr=learning_rate)
        # total_step = 1000
        # for i in range(total_step):
        #     optimizer.param_groups[0]["lr"] -= learning_rate / total_step
        # optimizer.param_groups[0]["lr"]  # Should be very close to zero
        # -1.5863074217500927e-18
        
        # Method 2: Adjust learning rate automatically using scheduler (Recommended)
        # huggingface (Recommended): link
        # pytorch: link
        # scheduler = ...
        # for i in range(total_step):
        #     ...
        #     optimizer.step()
        #     scheduler.step()
        #     ...
        # You only need to add 3 or 4 lines
        
        if config.get('use_scheduler', True):
            total_steps = len(self.train_loader) * config['num_epochs']
            warmup_steps = int(total_steps * config.get('warmup_ratio', 0.1))
            
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = None
        
        # Prepare for accelerated training
        self.model, self.optimizer, self.train_loader, self.val_loader = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        
        if self.scheduler:
            self.scheduler = self.accelerator.prepare(self.scheduler)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Training Epoch {epoch + 1}",
            disable=not self.accelerator.is_local_main_process
        )
        
        # ##### Training Tip: Gradient accumulation #####
        # Use it when gpu memory is not enough but you want to use larger batch size
        # Split global batch into smaller mini-batches
        # For each mini-batch: Accumulate gradient without updating model parameters
        # Update model parameters after processing all mini-batches in the global batch
        # Reference: Gradient Accumulation in PyTorch
        
        for batch_idx, batch in enumerate(progress_bar):
            with self.accelerator.accumulate(self.model):
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions']
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping (must be done before optimizer.step)
                if self.config.get('max_grad_norm', None):
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(), 
                            self.config['max_grad_norm']
                        )
                
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}"
            })
            
            # Log to wandb
            if self.config.get('use_wandb', False) and self.accelerator.is_local_main_process:
                wandb.log({
                    'train_loss': loss.item(),
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'epoch': epoch
                })
        
        return total_loss / len(self.train_loader)
    
    def evaluate(self, epoch: int):
        """Evaluate the model"""
        self.model.eval()
        all_predictions = []
        all_start_logits = []
        all_end_logits = []
        
        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader, 
                desc=f"Evaluating Epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in progress_bar:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids']
                )
                
                start_logits = self.accelerator.gather(outputs.start_logits)
                end_logits = self.accelerator.gather(outputs.end_logits)
                
                all_start_logits.extend(start_logits.cpu().numpy())
                all_end_logits.extend(end_logits.cpu().numpy())
        
        # Postprocess predictions (Boss level improvement)
        predictions = postprocess_predictions(
            all_start_logits,
            all_end_logits,
            self.val_dataset,
            self.tokenizer,
            max_answer_length=self.config.get('max_answer_length', 30)
        )
        
        # Compute metrics
        metrics = compute_metrics(predictions, self.config['val_data_path'])
        
        if self.accelerator.is_local_main_process:
            print(f"Epoch {epoch + 1} Metrics: {metrics}")
            
            if self.config.get('use_wandb', False):
                wandb.log({**metrics, 'epoch': epoch})
        
        return metrics
    
    def train(self):
        """Main training loop"""
        best_f1 = 0.0
        
        for epoch in range(self.config['num_epochs']):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Evaluation
            if (epoch + 1) % self.config.get('eval_every', 1) == 0:
                metrics = self.evaluate(epoch)
                
                # Save best model
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    self.save_model(f"best_model_epoch_{epoch + 1}")
                
                # Early stopping
                if self.config.get('early_stopping', False):
                    # Implement early stopping logic here
                    pass
            
            # Save checkpoint
            if (epoch + 1) % self.config.get('save_every', 5) == 0:
                self.save_model(f"checkpoint_epoch_{epoch + 1}")
        
        # Save final model
        self.save_model("final_model")
    
    def save_model(self, name: str):
        """Save model and tokenizer"""
        if self.accelerator.is_local_main_process:
            save_dir = os.path.join(self.config['output_dir'], name)
            os.makedirs(save_dir, exist_ok=True)
            
            # Unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            if isinstance(unwrapped_model, EnsembleModel):
                # Save ensemble components separately
                for i, model in enumerate(unwrapped_model.models):
                    model_save_dir = os.path.join(save_dir, f"model_{i}")
                    model.bert.save_pretrained(model_save_dir)
            else:
                unwrapped_model.bert.save_pretrained(save_dir)
            
            # Save tokenizer
            self.tokenizer.save_pretrained(save_dir)
            
            # Save config
            with open(os.path.join(save_dir, 'config.json'), 'w') as f:
                json.dump(self.config, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
