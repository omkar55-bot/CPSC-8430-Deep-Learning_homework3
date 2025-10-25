import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from typing import Dict, List, Tuple, Optional

class SpokenSquadDataset(Dataset):
    """Dataset class for Spoken-SQuAD data"""
    
    def __init__(self, data_path: str, tokenizer, max_length: int = 512, 
                 doc_stride: int = 150, mode: str = 'train'):
        """
        Args:
            data_path: Path to the JSON data file
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            doc_stride: Stride for sliding window (TODO: Change to 150 for better performance)
            mode: 'train' or 'test'
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.mode = mode
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # Handle both SQuAD format and flat list format
        if isinstance(raw_data, dict) and 'data' in raw_data:
            # SQuAD format - need to flatten
            self.data = self._flatten_squad_data(raw_data['data'])
        else:
            # Already flat list
            self.data = raw_data
        
        # Preprocess data
        self.examples = self._preprocess_data()
    
    def _flatten_squad_data(self, squad_data: List[Dict]) -> List[Dict]:
        """Flatten SQuAD format data to simple list format"""
        flattened = []
        
        for article in squad_data:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                
                for qa in paragraph['qas']:
                    item = {
                        'id': qa['id'],
                        'question': qa['question'],
                        'context': context,
                    }
                    
                    # Add answer info if available
                    if 'answers' in qa and len(qa['answers']) > 0:
                        item['answer_text'] = qa['answers'][0]['text']
                        item['answer_start'] = qa['answers'][0]['answer_start']
                    elif self.mode == 'test':
                        # Test data might not have answers
                        item['answer_text'] = ''
                        item['answer_start'] = -1
                    
                    flattened.append(item)
        
        return flattened
    
    def _preprocess_data(self) -> List[Dict]:
        """Preprocess the raw data into model-ready format"""
        examples = []
        
        for item in self.data:
            question = item['question']
            context = item['context']
            
            if self.mode == 'train':
                # For training, we know the answer position
                answer_text = item.get('answer_text', '')
                answer_start = item.get('answer_start', -1)
                
                if answer_start == -1:
                    continue  # Skip items without answers in training
                
                # Create sliding windows around the answer for long contexts
                examples.extend(self._create_training_windows(
                    question, context, answer_text, answer_start, item.get('id', len(examples))
                ))
            else:
                # For testing, create all possible windows
                examples.extend(self._create_test_windows(
                    question, context, item.get('id', len(examples))
                ))
        
        return examples
    
    def _create_training_windows(self, question: str, context: str, 
                               answer_text: str, answer_start: int, 
                               example_id: str) -> List[Dict]:
        """
        Create training windows centered around the answer
        ##### TODO: Preprocessing #####
        # Hint: How to prevent model from learning something it should not learn
        # during training? (i.e. answers are not always near the middle of window)
        # Solution: Center windows around the answer to avoid boundary effects
        """
        # Tokenize question and context separately first
        question_tokens = self.tokenizer(question, add_special_tokens=False)
        context_tokens = self.tokenizer(context, add_special_tokens=False)
        
        # Calculate available space for context
        question_len = len(question_tokens['input_ids'])
        available_context_len = self.max_length - question_len - 3  # [CLS], [SEP], [SEP]
        
        if len(context_tokens['input_ids']) <= available_context_len:
            # Context fits in one window
            return [self._create_single_example(question, context, answer_text, 
                                              answer_start, example_id, 0)]
        else:
            # Need sliding windows - center around answer to prevent learning boundary artifacts
            # Convert answer start position to token position
            tokenized_paragraph = self.tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)
            answer_end = answer_start + len(answer_text)
            
            # Find answer token positions
            answer_start_token = None
            answer_end_token = None
            
            for idx, (start_char, end_char) in enumerate(tokenized_paragraph['offset_mapping']):
                if start_char <= answer_start < end_char:
                    answer_start_token = idx
                if start_char < answer_end <= end_char:
                    answer_end_token = idx
                    break
            
            if answer_start_token is None or answer_end_token is None:
                # Fallback - use first window
                char_end = self._token_to_char_position(context, context_tokens, available_context_len)
                windowed_context = context[:char_end]
                return [self._create_single_example(question, windowed_context, 
                                                  answer_text, answer_start, 
                                                  example_id, 0)]
            
            # A single window is obtained by slicing the portion of paragraph containing the answer
            # Center the answer in the middle of the window
            mid = (answer_start_token + answer_end_token) // 2
            paragraph_start = max(0, min(mid - self.max_length // 2, len(tokenized_paragraph['input_ids']) - self.max_length))
            paragraph_end = paragraph_start + self.max_length
            
            # Make sure paragraph_end doesn't exceed the tokenized paragraph length
            paragraph_end = min(paragraph_end, len(tokenized_paragraph['input_ids']))
            
            # Get character positions
            if paragraph_start < len(tokenized_paragraph['offset_mapping']):
                char_start = tokenized_paragraph['offset_mapping'][paragraph_start][0]
            else:
                char_start = 0
                
            if paragraph_end > 0 and paragraph_end <= len(tokenized_paragraph['offset_mapping']):
                char_end = tokenized_paragraph['offset_mapping'][paragraph_end - 1][1]
            else:
                char_end = len(context)
            
            windowed_context = context[char_start:char_end]
            adjusted_answer_start = answer_start - char_start
            
            # Slice question/paragraph and add special tokens ([101]: CLS, 102: SEP)
            # input_ids_question = [101] + tokenized_question.ids[:self.max_question_len] + [102]
            # input_ids_paragraph = tokenized_paragraph.ids[paragraph_start : paragraph_end] + [102]
            
            return [self._create_single_example(question, windowed_context, 
                                              answer_text, adjusted_answer_start, 
                                              example_id, char_start)]
    
    def _create_test_windows(self, question: str, context: str, 
                           example_id: str) -> List[Dict]:
        """Create all possible sliding windows for testing"""
        question_tokens = self.tokenizer(question, add_special_tokens=False)
        context_tokens = self.tokenizer(context, add_special_tokens=False)
        
        question_len = len(question_tokens['input_ids'])
        available_context_len = self.max_length - question_len - 3
        
        if len(context_tokens['input_ids']) <= available_context_len:
            return [self._create_single_example(question, context, "", -1, example_id, 0)]
        
        # Create sliding windows
        windows = []
        start_token = 0
        
        while start_token < len(context_tokens['input_ids']):
            end_token = min(start_token + available_context_len, 
                          len(context_tokens['input_ids']))
            
            char_start = self._token_to_char_position(context, context_tokens, start_token)
            char_end = self._token_to_char_position(context, context_tokens, end_token)
            
            windowed_context = context[char_start:char_end]
            windows.append(self._create_single_example(question, windowed_context, 
                                                     "", -1, example_id, char_start))
            
            if end_token >= len(context_tokens['input_ids']):
                break
            
            start_token += self.doc_stride
        
        return windows
    
    def _create_single_example(self, question: str, context: str, 
                             answer_text: str, answer_start: int, 
                             example_id: str, context_offset: int) -> Dict:
        """Create a single training example"""
        # Tokenize the question-context pair
        encoded = self.tokenizer(
            question,
            context,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_tensors='pt'
        )
        
        example = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'token_type_ids': encoded['token_type_ids'].squeeze(),
            'example_id': example_id,
            'context_offset': context_offset
        }
        
        if self.mode == 'train' and answer_start != -1:
            # Find answer token positions
            start_positions, end_positions = self._find_answer_positions(
                encoded, answer_text, answer_start, context
            )
            example['start_positions'] = torch.tensor(start_positions, dtype=torch.long)
            example['end_positions'] = torch.tensor(end_positions, dtype=torch.long)
        
        return example
    
    def _find_answer_positions(self, encoded, answer_text: str, 
                             answer_start: int, context: str) -> Tuple[int, int]:
        """Find token positions for answer span"""
        offset_mapping = encoded['offset_mapping'].squeeze().numpy()
        
        answer_end = answer_start + len(answer_text)
        start_position = 0
        end_position = 0
        
        for idx, (start_char, end_char) in enumerate(offset_mapping):
            if start_char <= answer_start < end_char:
                start_position = idx
            if start_char < answer_end <= end_char:
                end_position = idx
                break
        
        return start_position, end_position
    
    def _create_char_to_token_map(self, text: str, tokens) -> List[int]:
        """Create mapping from character positions to token positions"""
        char_to_token = []
        token_idx = 0
        
        for char_idx in range(len(text)):
            while (token_idx < len(tokens['input_ids']) and 
                   self.tokenizer.decode([tokens['input_ids'][token_idx]]) and
                   char_idx >= len(self.tokenizer.decode(tokens['input_ids'][:token_idx+1]))):
                token_idx += 1
            char_to_token.append(min(token_idx, len(tokens['input_ids']) - 1))
        
        return char_to_token
    
    def _token_to_char_position(self, text: str, tokens, token_pos: int) -> int:
        """Convert token position to character position"""
        if token_pos == 0:
            return 0
        if token_pos >= len(tokens['input_ids']):
            return len(text)
        
        decoded_text = self.tokenizer.decode(tokens['input_ids'][:token_pos])
        return len(decoded_text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    """Custom collate function for DataLoader"""
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        if key in ['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions']:
            batched[key] = torch.stack([item[key] for item in batch])
        else:
            batched[key] = [item[key] for item in batch]
    
    return batched
