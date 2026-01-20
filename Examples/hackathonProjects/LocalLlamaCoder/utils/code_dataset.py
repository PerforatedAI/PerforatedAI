"""
Code Dataset Utilities
Handles loading and preprocessing of code datasets for fine-tuning
"""

import torch
from datasets import Dataset
import random


def create_synthetic_code_dataset(num_samples=100, max_length=512):
    """
    Create synthetic code dataset for demo purposes
    
    Args:
        num_samples: Number of samples to generate
        max_length: Maximum sequence length
        
    Returns:
        List of code examples
    """
    # Common code patterns for demonstration
    python_patterns = [
        "def {func}({args}):\n    {body}\n    return {ret}",
        "class {cls}:\n    def __init__(self{args}):\n        {body}",
        "for {var} in {iter}:\n    {body}",
        "if {cond}:\n    {body}\nelse:\n    {alt}",
        "try:\n    {body}\nexcept {exc}:\n    {handler}",
        "async def {func}({args}):\n    {body}\n    return await {ret}",
        "with {ctx} as {var}:\n    {body}",
        "@decorator\ndef {func}({args}):\n    {body}",
    ]
    
    # Generate synthetic samples
    samples = []
    for i in range(num_samples):
        pattern = random.choice(python_patterns)
        
        # Fill in placeholders
        code = pattern.format(
            func=random.choice(['process', 'calculate', 'fetch', 'transform']),
            cls=random.choice(['DataProcessor', 'APIClient', 'Cache', 'Handler']),
            args=random.choice(['', 'x', 'data', 'x, y', 'self, value']),
            body=random.choice(['pass', 'result = process()', 'print(data)', 'x += 1']),
            ret=random.choice(['None', 'result', 'True', 'data']),
            var=random.choice(['i', 'item', 'x', 'element']),
            iter=random.choice(['range(10)', 'items', 'data', 'collection']),
            cond=random.choice(['x > 0', 'data is not None', 'flag', 'len(items) > 0']),
            alt=random.choice(['pass', 'return None', 'raise Error()']),
            exc=random.choice(['Exception', 'ValueError', 'KeyError']),
            handler=random.choice(['pass', 'print(e)', 'log.error(e)']),
            ctx=random.choice(['open(file)', 'lock', 'connection']),
            decorator=random.choice(['@staticmethod', '@property', '@cached'])
        )
        
        samples.append({'text': code})
    
    return samples


def load_code_dataset(tokenizer, config):
    """
    Load and preprocess code dataset
    
    Args:
        tokenizer: HuggingFace tokenizer
        config: Dataset configuration
        
    Returns:
        Tuple of (train_dataset, eval_dataset)
    """
    print(f"📚 Loading dataset: {config['name']}")
    
    # Check if demo mode
    if config.get('demo_mode', False):
        print("  Using synthetic data for quick demo...")
        data = create_synthetic_code_dataset(
            num_samples=config.get('num_samples', 100),
            max_length=config['max_length']
        )
    else:
        # Load real dataset if available
        if config['name'] == 'synthetic':
            print("  Generating synthetic dataset...")
            data = create_synthetic_code_dataset(
                num_samples=config.get('num_samples_demo', 1000),
                max_length=config['max_length']
            )
        else:
            # For real datasets like codeparrot, stack-v2
            try:
                from datasets import load_dataset
                print(f"  Loading from HuggingFace: {config['name']}")
                dataset = load_dataset(config['name'], split='train[:1000]')
                data = [{'text': item['content']} for item in dataset]
            except Exception as e:
                print(f"  ⚠️  Could not load {config['name']}: {e}")
                print("  Falling back to synthetic data...")
                data = create_synthetic_code_dataset(1000, config['max_length'])
    
    # Create Dataset object
    dataset = Dataset.from_list(data)
    
    # Tokenize
    def tokenize_function(examples):
        result = tokenizer(
            examples['text'],
            truncation=True,
            max_length=config['max_length'],
            padding='max_length',
            return_tensors=None
        )
        result['labels'] = result['input_ids'].copy()
        return result
    
    print("  Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    # Split into train/eval
    split_ratio = config.get('split_ratio', {'train': 0.9, 'validation': 0.1})
    split_dataset = tokenized_dataset.train_test_split(
        test_size=split_ratio['validation'],
        seed=42
    )
    
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"✅ Dataset loaded:")
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Eval: {len(eval_dataset)} samples")
    
    return train_dataset, eval_dataset


def prepare_code_prompt(prompt, language='python'):
    """
    Prepare code prompt with language-specific formatting
    
    Args:
        prompt: Raw code prompt
        language: Programming language
        
    Returns:
        Formatted prompt
    """
    if language == 'python':
        # Ensure proper indentation
        lines = prompt.split('\n')
        formatted = '\n'.join(line.rstrip() for line in lines)
        return formatted
    
    return prompt


def extract_function_signature(code):
    """
    Extract function signature from code
    
    Args:
        code: Code string
        
    Returns:
        Function signature if found, else None
    """
    lines = code.split('\n')
    for line in lines:
        if line.strip().startswith('def ') or line.strip().startswith('async def '):
            return line.strip()
    return None
