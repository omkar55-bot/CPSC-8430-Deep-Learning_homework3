import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np

def analyze_dataset(data_path: str, output_dir: str = "analysis"):
    """
    Analyze Spoken-SQuAD dataset and generate insights
    
    Args:
        data_path: Path to the JSON data file
        output_dir: Directory to save analysis plots and reports
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(data)
    
    # Basic statistics
    print("=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    
    if 'question' in df.columns:
        df['question_length'] = df['question'].str.len()
        print(f"Average question length: {df['question_length'].mean():.2f} characters")
        print(f"Question length std: {df['question_length'].std():.2f}")
    
    if 'context' in df.columns:
        df['context_length'] = df['context'].str.len()
        print(f"Average context length: {df['context_length'].mean():.2f} characters")
        print(f"Context length std: {df['context_length'].std():.2f}")
    
    if 'answer_text' in df.columns:
        df['answer_length'] = df['answer_text'].str.len()
        print(f"Average answer length: {df['answer_length'].mean():.2f} characters")
        print(f"Answer length std: {df['answer_length'].std():.2f}")
    
    # Plot distributions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Question length distribution
    if 'question_length' in df.columns:
        axes[0, 0].hist(df['question_length'], bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_title('Question Length Distribution')
        axes[0, 0].set_xlabel('Characters')
        axes[0, 0].set_ylabel('Frequency')
    
    # Context length distribution  
    if 'context_length' in df.columns:
        axes[0, 1].hist(df['context_length'], bins=50, alpha=0.7, color='green')
        axes[0, 1].set_title('Context Length Distribution')
        axes[0, 1].set_xlabel('Characters')
        axes[0, 1].set_ylabel('Frequency')
    
    # Answer length distribution
    if 'answer_length' in df.columns:
        axes[1, 0].hist(df['answer_length'], bins=50, alpha=0.7, color='red')
        axes[1, 0].set_title('Answer Length Distribution')
        axes[1, 0].set_xlabel('Characters')
        axes[1, 0].set_ylabel('Frequency')
    
    # Answer position distribution
    if 'answer_start' in df.columns:
        axes[1, 1].hist(df['answer_start'], bins=50, alpha=0.7, color='orange')
        axes[1, 1].set_title('Answer Start Position Distribution')
        axes[1, 1].set_xlabel('Character Position')
        axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/dataset_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save statistics to CSV
    stats_df = df.describe()
    stats_df.to_csv(f"{output_dir}/dataset_statistics.csv")
    
    print(f"\nAnalysis saved to {output_dir}/")
    return df


def create_sample_data(output_path: str = "data/sample.json", num_samples: int = 100):
    """
    Create sample Spoken-SQuAD data for testing
    
    Args:
        output_path: Path to save sample data
        num_samples: Number of sample QA pairs to create
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sample contexts (simplified Chinese examples)
    sample_contexts = [
        "深度学习是机器学习的一个分支，它基于人工神经网络进行学习。深度学习在计算机视觉、自然语言处理等领域取得了重大突破。BERT是一个基于Transformer的双向编码器，它在多个自然语言处理任务上都取得了最先进的结果。",
        "中国是世界上人口最多的国家，拥有超过14亿人口。北京是中国的首都，也是政治和文化中心。上海是中国最大的城市，是重要的经济和金融中心。",
        "人工智能（AI）是计算机科学的一个领域，致力于创造能够执行通常需要人类智能的任务的系统。机器学习是人工智能的一个子集，它使计算机能够从数据中学习而无需明确编程。",
        "气象学中，降水是指从云中落下的任何形式的水。重力是降水落下的主要原因。降水的主要形式包括细雨、雨、雨夹雪、雪、冰雹等。",
        "自然语言处理（NLP）是计算机科学和人工智能的一个分支，专注于计算机与人类语言之间的交互。问答系统是NLP的一个重要应用，它能够自动回答用户提出的问题。"
    ]
    
    # Sample question templates and answers
    qa_templates = [
        ("什么是深度学习？", "机器学习的一个分支", 0),
        ("BERT是什么？", "基于Transformer的双向编码器", 2),
        ("中国的首都是哪里？", "北京", 1), 
        ("什么导致降水落下？", "重力", 3),
        ("NLP是什么的缩写？", "自然语言处理", 4)
    ]
    
    sample_data = []
    
    for i in range(num_samples):
        # Select context and QA pair
        qa_idx = i % len(qa_templates)
        context_idx = qa_templates[qa_idx][2]
        
        question, answer, _ = qa_templates[qa_idx]
        context = sample_contexts[context_idx]
        
        # Find answer position in context
        answer_start = context.find(answer)
        if answer_start == -1:
            answer_start = 0  # Fallback
        
        sample_data.append({
            "id": f"sample_{i}",
            "question": question,
            "context": context,
            "answer_text": answer,
            "answer_start": answer_start
        })
    
    # Save sample data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {num_samples} sample QA pairs in {output_path}")


def validate_data_format(data_path: str) -> bool:
    """
    Validate that data follows the expected format
    
    Args:
        data_path: Path to the JSON data file
        
    Returns:
        True if format is valid, False otherwise
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print("Error: Data should be a list of examples")
            return False
        
        required_fields = ['question', 'context']
        optional_fields = ['answer_text', 'answer_start', 'id']
        
        for i, item in enumerate(data[:5]):  # Check first 5 items
            if not isinstance(item, dict):
                print(f"Error: Item {i} is not a dictionary")
                return False
            
            for field in required_fields:
                if field not in item:
                    print(f"Error: Item {i} missing required field '{field}'")
                    return False
            
            # Check answer alignment if both answer fields present
            if 'answer_text' in item and 'answer_start' in item:
                answer_text = item['answer_text']
                answer_start = item['answer_start']
                context = item['context']
                
                if answer_start >= 0:
                    extracted_answer = context[answer_start:answer_start + len(answer_text)]
                    if extracted_answer != answer_text:
                        print(f"Warning: Answer alignment issue in item {i}")
                        print(f"Expected: '{answer_text}'")
                        print(f"Found: '{extracted_answer}'")
        
        print(f"✓ Data format validation passed for {data_path}")
        print(f"✓ Found {len(data)} examples")
        return True
        
    except Exception as e:
        print(f"Error validating data: {e}")
        return False


if __name__ == "__main__":
    # Create sample data for testing
    create_sample_data("data/sample_train.json", 1000)
    create_sample_data("data/sample_dev.json", 200) 
    create_sample_data("data/sample_test.json", 200)
    
    # Validate sample data
    validate_data_format("data/sample_train.json")
    
    # Analyze sample data
    analyze_dataset("data/sample_train.json")
