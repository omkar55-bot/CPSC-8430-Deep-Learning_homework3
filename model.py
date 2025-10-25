import torch
import torch.nn as nn
from transformers import AutoModelForQuestionAnswering
from typing import Dict, Tuple

class SpokenSquadModel(nn.Module):
    """Model for Spoken-SQuAD Question Answering"""
    
    def __init__(self, model_name: str = "bert-base-chinese", dropout: float = 0.1):
        """
        Args:
            model_name: HuggingFace model name (can be any from the hub)
            dropout: Dropout rate for additional layers
        """
        super().__init__()
        
        # Load pretrained model
        self.bert = AutoModelForQuestionAnswering.from_pretrained(model_name)
        
        # Get hidden size from the model config
        self.hidden_size = self.bert.config.hidden_size
        
        # Additional dropout for regularization (Strong level improvement)
        self.dropout = nn.Dropout(dropout)
        
        # Optional: Add additional layers for Boss level
        # Uncomment for Boss level improvements
        # self.start_classifier = nn.Linear(self.hidden_size, 1)
        # self.end_classifier = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids, 
                start_positions=None, end_positions=None):
        """
        Forward pass
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs (0 for question, 1 for context)
            start_positions: Ground truth start positions (training only)
            end_positions: Ground truth end positions (training only)
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions,
            return_dict=True
        )
        
        return outputs


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for Boss level performance"""
    
    def __init__(self, model_names: list, weights: list = None):
        """
        Args:
            model_names: List of HuggingFace model names
            weights: Weights for ensemble (if None, use equal weights)
        """
        super().__init__()
        
        self.models = nn.ModuleList([
            SpokenSquadModel(model_name) for model_name in model_names
        ])
        
        self.weights = weights if weights else [1.0 / len(model_names)] * len(model_names)
    
    def forward(self, input_ids, attention_mask, token_type_ids, 
                start_positions=None, end_positions=None):
        """Ensemble forward pass"""
        start_logits_list = []
        end_logits_list = []
        total_loss = 0.0
        
        for i, model in enumerate(self.models):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            start_logits_list.append(outputs.start_logits * self.weights[i])
            end_logits_list.append(outputs.end_logits * self.weights[i])
            
            if outputs.loss is not None:
                total_loss += outputs.loss * self.weights[i]
        
        # Combine logits
        start_logits = torch.stack(start_logits_list).sum(dim=0)
        end_logits = torch.stack(end_logits_list).sum(dim=0)
        
        # Create output similar to HuggingFace format
        class EnsembleOutput:
            def __init__(self, start_logits, end_logits, loss=None):
                self.start_logits = start_logits
                self.end_logits = end_logits
                self.loss = loss
        
        return EnsembleOutput(start_logits, end_logits, total_loss if start_positions is not None else None)
