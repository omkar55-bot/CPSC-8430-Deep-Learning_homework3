import torch
import json
import argparse
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from model import SpokenSquadModel, EnsembleModel
from dataset import SpokenSquadDataset, collate_fn
from utils import postprocess_predictions, compute_metrics
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

class Evaluator:
    """Evaluation class for Spoken-SQuAD QA"""
    
    def __init__(self, model_path: str, config: dict):
        """
        Args:
            model_path: Path to the trained model
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        if config.get('use_ensemble', False):
            self.model = self.load_ensemble_model(model_path)
        else:
            # Try loading with AutoModelForQuestionAnswering first
            try:
                self.model = AutoModelForQuestionAnswering.from_pretrained(model_path)
                print(f"✅ Loaded model from {model_path}")
            except:
                # Fallback: the saved config is training config, not model config
                # Load the base model and then load weights
                print(f"⚠️ Loading model using fallback method...")
                self.model = AutoModelForQuestionAnswering.from_pretrained(config.get('model_name', 'bert-base-chinese'))
                
                # Load safetensors or pytorch_model.bin
                model_file = os.path.join(model_path, 'model.safetensors')
                if os.path.exists(model_file):
                    from safetensors.torch import load_file
                    state_dict = load_file(model_file)
                    self.model.load_state_dict(state_dict)
                    print(f"✅ Loaded weights from {model_file}")
                else:
                    model_file = os.path.join(model_path, 'pytorch_model.bin')
                    state_dict = torch.load(model_file, map_location=self.device)
                    self.model.load_state_dict(state_dict)
                    print(f"✅ Loaded weights from {model_file}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def load_ensemble_model(self, model_path: str):
        """Load ensemble model"""
        ensemble_models = []
        
        # Look for model_0, model_1, model_2, etc.
        model_idx = 0
        while True:
            model_dir = os.path.join(model_path, f'model_{model_idx}')
            if not os.path.exists(model_dir):
                break
            
            print(f"Loading ensemble model {model_idx} from {model_dir}...")
            model = AutoModelForQuestionAnswering.from_pretrained(model_dir)
            ensemble_models.append(model)
            model_idx += 1
        
        if len(ensemble_models) == 0:
            raise ValueError(f"No ensemble models found in {model_path}")
        
        print(f"✅ Loaded {len(ensemble_models)} ensemble models")
        
        # Create a simple ensemble wrapper
        class SimpleEnsemble:
            def __init__(self, models):
                self.models = models
            
            def to(self, device):
                for model in self.models:
                    model.to(device)
                return self
            
            def eval(self):
                for model in self.models:
                    model.eval()
                return self
            
            def __call__(self, **kwargs):
                # Average predictions from all models
                all_start_logits = []
                all_end_logits = []
                
                for model in self.models:
                    outputs = model(**kwargs)
                    all_start_logits.append(outputs.start_logits)
                    all_end_logits.append(outputs.end_logits)
                
                # Average the logits
                avg_start_logits = torch.stack(all_start_logits).mean(dim=0)
                avg_end_logits = torch.stack(all_end_logits).mean(dim=0)
                
                # Return in same format as single model
                class EnsembleOutput:
                    def __init__(self, start_logits, end_logits):
                        self.start_logits = start_logits
                        self.end_logits = end_logits
                
                return EnsembleOutput(avg_start_logits, avg_end_logits)
        
        return SimpleEnsemble(ensemble_models)
    
    def evaluate_dataset(self, test_data_path: str, output_file: str = None):
        """
        Evaluate model on test dataset
        
        Args:
            test_data_path: Path to test data
            output_file: Optional path to save predictions
        """
        # Create dataset
        test_dataset = SpokenSquadDataset(
            test_data_path,
            self.tokenizer,
            max_length=self.config['max_length'],
            doc_stride=self.config.get('doc_stride', 128),
            mode='test'
        )
        
        # Create data loader
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.get('eval_batch_size', 16),
            shuffle=False,
            collate_fn=collate_fn
        )
        
        # Run inference
        all_start_logits = []
        all_end_logits = []
        
        print("Running inference...")
        with torch.no_grad():
            for batch in tqdm(test_loader):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids']
                )
                
                all_start_logits.extend(outputs.start_logits.cpu().numpy())
                all_end_logits.extend(outputs.end_logits.cpu().numpy())
        
        # Postprocess predictions
        print("Postprocessing predictions...")
        predictions = postprocess_predictions(
            all_start_logits,
            all_end_logits,
            test_dataset,
            self.tokenizer,
            max_answer_length=self.config.get('max_answer_length', 30)
        )
        
        # Save predictions if requested
        if output_file:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            print(f"Predictions saved to {output_file}")
        
        # Compute metrics if ground truth is available
        try:
            metrics = compute_metrics(predictions, test_data_path)
            print(f"Evaluation Results:")
            print(f"F1 Score: {metrics['f1']:.2f}%")
            print(f"Exact Match: {metrics['exact_match']:.2f}%")
            print(f"Total Questions: {metrics['total_questions']}")
            return metrics
        except Exception as e:
            print(f"Could not compute metrics: {e}")
            return predictions
    
    def predict_single(self, question: str, context: str) -> dict:
        """
        Predict answer for a single question-context pair
        
        Args:
            question: Question text
            context: Context text
            
        Returns:
            Dictionary with prediction details
        """
        # Tokenize input
        encoded = self.tokenizer(
            question,
            context,
            max_length=self.config['max_length'],
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Move to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded['input_ids'],
                attention_mask=encoded['attention_mask'],
                token_type_ids=encoded['token_type_ids']
            )
        
        # Get best answer span
        start_logits = outputs.start_logits.cpu().numpy()[0]
        end_logits = outputs.end_logits.cpu().numpy()[0]
        
        # Find best span
        best_start = np.argmax(start_logits)
        best_end = np.argmax(end_logits)
        
        if best_end < best_start:
            best_end = best_start
        
        # Extract answer
        input_ids = encoded['input_ids'][0].cpu().numpy()
        answer_tokens = input_ids[best_start:best_end + 1]
        answer_text = self.tokenizer.decode(
            answer_tokens, 
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        return {
            'question': question,
            'context': context,
            'answer': answer_text.strip(),
            'start_position': best_start,
            'end_position': best_end,
            'start_score': float(start_logits[best_start]),
            'end_score': float(end_logits[best_end]),
            'confidence': float(start_logits[best_start] + end_logits[best_end])
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model')
    parser.add_argument('--test_data', type=str, required=True,
                       help='Path to test data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--output_file', type=str, default='predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--single_prediction', action='store_true',
                       help='Interactive single prediction mode')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create evaluator
    evaluator = Evaluator(args.model_path, config)
    
    if args.single_prediction:
        # Interactive mode
        print("Interactive prediction mode. Type 'quit' to exit.")
        while True:
            question = input("\nEnter question: ")
            if question.lower() == 'quit':
                break
            
            context = input("Enter context: ")
            if context.lower() == 'quit':
                break
            
            result = evaluator.predict_single(question, context)
            print(f"\nPredicted Answer: {result['answer']}")
            print(f"Confidence: {result['confidence']:.4f}")
    else:
        # Evaluate on test dataset
        evaluator.evaluate_dataset(args.test_data, args.output_file)


if __name__ == "__main__":
    main()
