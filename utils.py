import torch
import numpy as np
import json
from collections import Counter
import string
import re
from typing import Dict, List, Tuple, Any

def normalize_answer(s: str) -> str:
    """Normalize answer string for evaluation"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth"""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0
    
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Check if prediction exactly matches ground truth"""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def compute_metrics(predictions: Dict, ground_truth_path: str) -> Dict[str, float]:
    """Compute evaluation metrics"""
    # Load ground truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth_data = json.load(f)
    
    # Create ground truth mapping
    ground_truth = {}
    
    # Handle both SQuAD format and flat list format
    if isinstance(ground_truth_data, dict) and 'data' in ground_truth_data:
        # SQuAD format - need to flatten
        for article in ground_truth_data['data']:
            for paragraph in article['paragraphs']:
                for qa in paragraph['qas']:
                    if 'answers' in qa and len(qa['answers']) > 0:
                        ground_truth[qa['id']] = qa['answers'][0]['text']
    else:
        # Flat list format
        for item in ground_truth_data:
            ground_truth[item['id']] = item['answer_text']
    
    f1_scores = []
    exact_matches = []
    
    for qid, prediction in predictions.items():
        if qid in ground_truth:
            gt_answer = ground_truth[qid]
            
            f1_scores.append(f1_score(prediction, gt_answer))
            exact_matches.append(exact_match_score(prediction, gt_answer))
    
    return {
        'f1': np.mean(f1_scores) * 100,
        'exact_match': np.mean(exact_matches) * 100,
        'total_questions': len(f1_scores)
    }


def postprocess_predictions(start_logits: List[np.ndarray], 
                          end_logits: List[np.ndarray],
                          dataset: Any,
                          tokenizer: Any,
                          max_answer_length: int = 30,
                          null_score_diff_threshold: float = 0.0) -> Dict[str, str]:
    """
    Postprocess model predictions to extract final answers
    This is a Boss-level improvement for better answer extraction
    """
    predictions = {}
    
    # Group predictions by example_id (for sliding windows)
    example_predictions = {}
    
    for idx, example in enumerate(dataset.examples):
        example_id = example['example_id']
        
        if example_id not in example_predictions:
            example_predictions[example_id] = []
        
        example_predictions[example_id].append({
            'start_logits': start_logits[idx],
            'end_logits': end_logits[idx],
            'input_ids': example['input_ids'],
            'attention_mask': example['attention_mask'],
            'context_offset': example['context_offset']
        })
    
    # Process each example
    for example_id, windows in example_predictions.items():
        best_answer = ""
        best_score = float('-inf')
        
        for window in windows:
            # Get the best answer span for this window
            answer, score = get_best_answer_from_window(
                window, tokenizer, max_answer_length
            )
            
            if score > best_score:
                best_score = score
                best_answer = answer
        
        # Apply null answer threshold (Boss level improvement)
        if best_score < null_score_diff_threshold:
            best_answer = ""
        
        predictions[example_id] = best_answer
    
    return predictions


def get_best_answer_from_window(window: Dict, 
                              tokenizer: Any, 
                              max_answer_length: int) -> Tuple[str, float]:
    """
    Extract best answer from a single window
    ##### TODO: Postprocessing #####
    # There is a bug and room for improvement in postprocessing
    # Hint: Open your prediction file to see what is wrong
    # (e.g. what if predicted end_index < predicted start_index?)
    """
    start_logits = window['start_logits']
    end_logits = window['end_logits']
    input_ids = window['input_ids']
    attention_mask = window['attention_mask']
    
    # Convert to numpy if needed
    if torch.is_tensor(start_logits):
        start_logits = start_logits.cpu().numpy()
    if torch.is_tensor(end_logits):
        end_logits = end_logits.cpu().numpy()
    if torch.is_tensor(input_ids):
        input_ids = input_ids.cpu().numpy()
    if torch.is_tensor(attention_mask):
        attention_mask = attention_mask.cpu().numpy()
    
    # Find valid start and end positions
    valid_positions = np.where(attention_mask == 1)[0]
    
    # Get top k start and end positions
    k = 20
    start_indices = np.argsort(start_logits)[-k:][::-1]
    end_indices = np.argsort(end_logits)[-k:][::-1]
    
    best_answer = ""
    best_score = float('-inf')
    
    for start_idx in start_indices:
        if start_idx not in valid_positions:
            continue
            
        for end_idx in end_indices:
            if end_idx not in valid_positions:
                continue
            
            # POSTPROCESSING IMPROVEMENT: Handle cases where end < start
            # Check if this is a valid span - if end < start, skip it
            if end_idx < start_idx:
                continue
            
            if end_idx - start_idx + 1 > max_answer_length:
                continue
            
            # Calculate score
            score = start_logits[start_idx] + end_logits[end_idx]
            
            if score > best_score:
                best_score = score
                
                # Extract answer text
                answer_tokens = input_ids[start_idx:end_idx + 1]
                answer_text = tokenizer.decode(
                    answer_tokens, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
                best_answer = answer_text.strip()
    
    return best_answer, best_score


def create_sliding_windows(text: str, 
                         tokenizer: Any, 
                         max_length: int, 
                         doc_stride: int,
                         question: str = "") -> List[Dict]:
    """
    Create sliding windows for long documents (Strong level improvement)
    This is used during inference for better handling of long contexts
    """
    # Tokenize question to know how much space we have for context
    question_tokens = tokenizer(question, add_special_tokens=False)
    question_length = len(question_tokens['input_ids'])
    
    # Available space for context (accounting for special tokens)
    max_context_length = max_length - question_length - 3  # [CLS], [SEP], [SEP]
    
    # Tokenize the full context
    context_tokens = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    
    windows = []
    start_pos = 0
    
    while start_pos < len(context_tokens['input_ids']):
        # Define window end
        end_pos = min(start_pos + max_context_length, len(context_tokens['input_ids']))
        
        # Get character positions for this window
        if 'offset_mapping' in context_tokens:
            char_start = context_tokens['offset_mapping'][start_pos][0]
            char_end = context_tokens['offset_mapping'][end_pos - 1][1]
            window_text = text[char_start:char_end]
        else:
            # Fallback if offset_mapping not available
            window_tokens = context_tokens['input_ids'][start_pos:end_pos]
            window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
        
        windows.append({
            'text': window_text,
            'start_char': char_start if 'offset_mapping' in context_tokens else 0,
            'end_char': char_end if 'offset_mapping' in context_tokens else len(window_text)
        })
        
        # Move to next window
        if end_pos >= len(context_tokens['input_ids']):
            break
        
        start_pos += doc_stride
    
    return windows


def load_spoken_squad_data(file_path: str) -> List[Dict]:
    """Load Spoken-SQuAD data from JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Ensure consistent format
    formatted_data = []
    for item in data:
        formatted_item = {
            'id': item.get('id', str(len(formatted_data))),
            'question': item['question'],
            'context': item['context'],
        }
        
        # Add answer info if available (for training data)
        if 'answer_text' in item:
            formatted_item['answer_text'] = item['answer_text']
        if 'answer_start' in item:
            formatted_item['answer_start'] = item['answer_start']
        
        formatted_data.append(formatted_item)
    
    return formatted_data


class EarlyStopping:
    """Early stopping utility class"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = score
            self.counter = 0
        
        return False


def calculate_word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) for speech recognition evaluation"""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    
    # Simple WER calculation (for more accurate results, use edit distance)
    if len(ref_words) == 0:
        return float('inf') if len(hyp_words) > 0 else 0.0
    
    # This is a simplified version - for production use proper edit distance
    correct = sum(1 for r, h in zip(ref_words, hyp_words) if r == h)
    wer = (len(ref_words) - correct) / len(ref_words)
    
    return wer * 100
