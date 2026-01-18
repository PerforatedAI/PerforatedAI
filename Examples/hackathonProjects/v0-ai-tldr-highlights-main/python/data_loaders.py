"""
Data loaders for various summarization and forum datasets.

Supports:
- Reddit TLDR datasets (Webis-TLDR-17, TLDR9+, TLDRHQ)
- SAMSum (chat summarization)
- ForumSum (forum conversation summarization)
- Generic summarization datasets (CNN/DailyMail, XSum)
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import pandas as pd
from datasets import load_dataset, Dataset
from dataclasses import dataclass
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    source: str  # 'huggingface', 'local', 'download'
    split: str = 'train'
    max_samples: Optional[int] = None
    cache_dir: Optional[str] = None


class ForumDataLoader:
    """Unified data loader for forum and summarization datasets."""
    
    def __init__(self, cache_dir: str = "./data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_reddit_tldr(self, subset: str = "webis-tldr-17") -> Dataset:
        """
        Load Reddit TLDR datasets.
        
        Args:
            subset: 'webis-tldr-17', 'tldr9+', or 'tldrhq'
        """
        logger.info(f"Loading Reddit TLDR dataset: {subset}")
        
        if subset == "webis-tldr-17":
            # Load from Hugging Face
            dataset = load_dataset(
                "webis/tldr-17",
                cache_dir=str(self.cache_dir)
            )
            return self._process_reddit_tldr(dataset['train'])
        
        elif subset == "tldr9+":
            logger.info("TLDR9+ requires manual download from authors")
            raise NotImplementedError("Please download TLDR9+ from https://github.com/salesforce/ctrl-sum")
        
        elif subset == "tldrhq":
            dataset = load_dataset(
                "reddit_tifu",
                "long",
                cache_dir=str(self.cache_dir)
            )
            return self._process_reddit_tifu(dataset['train'])
        
        else:
            raise ValueError(f"Unknown Reddit TLDR subset: {subset}")
    
    def _process_reddit_tldr(self, dataset: Dataset) -> Dataset:
        """Process Webis TLDR-17 format."""
        
        def format_example(example):
            return {
                'thread_id': example.get('id', ''),
                'content': example.get('content', ''),
                'summary': example.get('summary', ''),
                'source': 'reddit',
                'metadata': {
                    'subreddit': example.get('subreddit', ''),
                    'author': example.get('author', ''),
                    'score': example.get('score', 0)
                }
            }
        
        return dataset.map(format_example)
    
    def _process_reddit_tifu(self, dataset: Dataset) -> Dataset:
        """Process Reddit TIFU format."""
        
        def format_example(example):
            return {
                'thread_id': example.get('ups', ''),
                'content': example.get('documents', ''),
                'summary': example.get('tldr', ''),
                'source': 'reddit_tifu',
                'metadata': {}
            }
        
        return dataset.map(format_example)
    
    def load_samsum(self) -> Dataset:
        """Load SAMSum dialogue summarization dataset."""
        logger.info("Loading SAMSum dataset")
        
        dataset = load_dataset(
            "samsum",
            cache_dir=str(self.cache_dir)
        )
        
        def format_example(example):
            return {
                'thread_id': example.get('id', ''),
                'content': example.get('dialogue', ''),
                'summary': example.get('summary', ''),
                'source': 'samsum',
                'metadata': {'type': 'chat'}
            }
        
        return dataset['train'].map(format_example)
    
    def load_cnn_dailymail(self) -> Dataset:
        """Load CNN/DailyMail news summarization dataset."""
        logger.info("Loading CNN/DailyMail dataset")
        
        dataset = load_dataset(
            "cnn_dailymail",
            "3.0.0",
            cache_dir=str(self.cache_dir)
        )
        
        def format_example(example):
            return {
                'thread_id': example.get('id', ''),
                'content': example.get('article', ''),
                'summary': example.get('highlights', ''),
                'source': 'cnn_dailymail',
                'metadata': {'type': 'news'}
            }
        
        return dataset['train'].map(format_example)
    
    def load_xsum(self) -> Dataset:
        """Load XSum extreme summarization dataset."""
        logger.info("Loading XSum dataset")
        
        dataset = load_dataset(
            "xsum",
            cache_dir=str(self.cache_dir)
        )
        
        def format_example(example):
            return {
                'thread_id': example.get('id', ''),
                'content': example.get('document', ''),
                'summary': example.get('summary', ''),
                'source': 'xsum',
                'metadata': {'type': 'news_extreme'}
            }
        
        return dataset['train'].map(format_example)
    
    def create_sentence_classification_dataset(
        self,
        threads: List[Dict[str, Any]],
        output_path: str
    ):
        """
        Create sentence classification dataset from threads.
        
        Extracts sentences and labels them for highlight classification.
        """
        logger.info("Creating sentence classification dataset")
        
        sentences = []
        labels = []
        
        for thread in threads:
            content = thread['content']
            summary = thread['summary']
            
            # Split into sentences
            thread_sentences = self._split_sentences(content)
            summary_sentences = set(self._split_sentences(summary))
            
            for sentence in thread_sentences:
                # Heuristic labeling
                category = self._categorize_sentence(sentence, summary_sentences)
                
                sentences.append(sentence)
                labels.append(category)
        
        # Create DataFrame
        df = pd.DataFrame({
            'sentence': sentences,
            'category': labels
        })
        
        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logger.info(f"Saved {len(df)} sentences to {output_path}")
        
        return df
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _categorize_sentence(self, sentence: str, summary_sentences: set) -> str:
        """
        Heuristically categorize a sentence.
        
        This is a simple baseline - real labels should come from annotation.
        """
        sentence_lower = sentence.lower()
        
        # Check if in summary (likely a fact or solution)
        if any(s.lower() in sentence_lower or sentence_lower in s.lower() 
               for s in summary_sentences):
            if '?' in sentence:
                return 'question'
            elif any(word in sentence_lower for word in ['should', 'try', 'use', 'recommend']):
                return 'solution'
            else:
                return 'fact'
        
        # Check for question
        if sentence.endswith('?') or sentence_lower.startswith(('how', 'what', 'why', 'when', 'where', 'who')):
            return 'question'
        
        # Check for citation
        if 'http' in sentence or 'www.' in sentence or 'source:' in sentence_lower:
            return 'citation'
        
        # Check for opinion
        if any(word in sentence_lower for word in ['i think', 'i believe', 'in my opinion', 'imho', 'imo']):
            return 'opinion'
        
        # Check for solution
        if any(word in sentence_lower for word in ['solution', 'fix', 'try', 'should', 'could', 'recommend']):
            return 'solution'
        
        # Check for greeting/irrelevant
        if any(word in sentence_lower for word in ['thanks', 'thank you', 'hello', 'hi ', 'bye']):
            return 'irrelevant'
        
        # Default to fact
        return 'fact'
    
    def load_custom_forum_data(self, file_path: str, format: str = 'json') -> List[Dict[str, Any]]:
        """
        Load custom forum data from local file.
        
        Args:
            file_path: Path to data file
            format: 'json', 'csv', or 'jsonl'
        """
        logger.info(f"Loading custom forum data from {file_path}")
        
        file_path = Path(file_path)
        
        if format == 'json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        
        elif format == 'jsonl':
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
        
        elif format == 'csv':
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
        
        else:
            raise ValueError(f"Unknown format: {format}")
        
        return data
    
    def merge_datasets(
        self,
        datasets: List[Dataset],
        sample_weights: Optional[List[float]] = None
    ) -> Dataset:
        """
        Merge multiple datasets with optional sampling weights.
        
        Args:
            datasets: List of datasets to merge
            sample_weights: Optional weights for sampling from each dataset
        """
        logger.info(f"Merging {len(datasets)} datasets")
        
        from datasets import concatenate_datasets
        
        if sample_weights:
            # Sample from each dataset according to weights
            sampled_datasets = []
            for dataset, weight in zip(datasets, sample_weights):
                n_samples = int(len(dataset) * weight)
                sampled = dataset.shuffle(seed=42).select(range(min(n_samples, len(dataset))))
                sampled_datasets.append(sampled)
            
            return concatenate_datasets(sampled_datasets)
        else:
            return concatenate_datasets(datasets)


def prepare_training_data(
    output_dir: str = "./data",
    include_datasets: Optional[List[str]] = None
):
    """
    Prepare training data from multiple sources.
    
    Args:
        output_dir: Directory to save processed data
        include_datasets: List of datasets to include (default: all available)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loader = ForumDataLoader(cache_dir=str(output_dir / "cache"))
    
    if include_datasets is None:
        include_datasets = ['reddit_tldr', 'samsum', 'cnn_dailymail']
    
    all_threads = []
    
    # Load datasets
    if 'reddit_tldr' in include_datasets:
        try:
            reddit_data = loader.load_reddit_tldr('tldrhq')
            all_threads.extend([dict(item) for item in reddit_data])
            logger.info(f"Loaded {len(reddit_data)} Reddit TLDR examples")
        except Exception as e:
            logger.warning(f"Failed to load Reddit TLDR: {e}")
    
    if 'samsum' in include_datasets:
        try:
            samsum_data = loader.load_samsum()
            all_threads.extend([dict(item) for item in samsum_data])
            logger.info(f"Loaded {len(samsum_data)} SAMSum examples")
        except Exception as e:
            logger.warning(f"Failed to load SAMSum: {e}")
    
    if 'cnn_dailymail' in include_datasets:
        try:
            cnn_data = loader.load_cnn_dailymail()
            # Sample a subset (CNN/DM is very large)
            cnn_sample = cnn_data.shuffle(seed=42).select(range(min(10000, len(cnn_data))))
            all_threads.extend([dict(item) for item in cnn_sample])
            logger.info(f"Loaded {len(cnn_sample)} CNN/DailyMail examples")
        except Exception as e:
            logger.warning(f"Failed to load CNN/DailyMail: {e}")
    
    # Create sentence classification dataset
    sentence_dataset = loader.create_sentence_classification_dataset(
        all_threads,
        output_path=str(output_dir / "sentence_classification.csv")
    )
    
    logger.info(f"Created sentence classification dataset with {len(sentence_dataset)} examples")
    
    # Save thread data
    threads_path = output_dir / "threads.jsonl"
    with open(threads_path, 'w') as f:
        for thread in all_threads:
            f.write(json.dumps(thread) + '\n')
    
    logger.info(f"Saved {len(all_threads)} threads to {threads_path}")
    
    # Create train/val/test splits
    from sklearn.model_selection import train_test_split
    
    train_data, temp_data = train_test_split(sentence_dataset, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    train_data.to_csv(output_dir / "train.csv", index=False)
    val_data.to_csv(output_dir / "val.csv", index=False)
    test_data.to_csv(output_dir / "test.csv", index=False)
    
    logger.info(f"Split data: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    return {
        'train': len(train_data),
        'val': len(val_data),
        'test': len(test_data),
        'total_threads': len(all_threads)
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare training data')
    parser.add_argument('--output-dir', type=str, default='./data', help='Output directory')
    parser.add_argument('--datasets', nargs='+', 
                       choices=['reddit_tldr', 'samsum', 'cnn_dailymail', 'xsum'],
                       default=['reddit_tldr', 'samsum'],
                       help='Datasets to include')
    
    args = parser.parse_args()
    
    stats = prepare_training_data(
        output_dir=args.output_dir,
        include_datasets=args.datasets
    )
    
    print("\nData preparation complete!")
    print(json.dumps(stats, indent=2))
