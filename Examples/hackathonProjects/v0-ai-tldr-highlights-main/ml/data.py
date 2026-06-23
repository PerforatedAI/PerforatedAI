"""
Dataset Loading and Preprocessing for PAI Experiments

Provides consistent data loading for all experiment types with:
- Standard summarization datasets (SAMSum, Reddit TL;DR)
- Custom Foru.ms export format
- Sentence-level preprocessing with offset tracking

Usage:
    from ml.data import load_dataset, preprocess_for_summarization
    
    dataset = load_dataset("samsum", split="train")
    examples = preprocess_for_summarization(dataset)
"""

import re
import logging
from typing import Dict, List, Any, Optional, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)


@dataclass
class ProcessedExample:
    """A preprocessed training example with metadata."""
    id: str
    source_text: str
    target_text: str
    sentences: List[str]
    sentence_offsets: List[Tuple[int, int]]  # (start, end) character offsets
    metadata: Dict[str, Any]


def load_dataset(
    name: str,
    split: str = "train",
    path: Optional[str] = None,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load a dataset by name.
    
    Supported datasets:
    - samsum: SAMSum dialogue summarization
    - reddit_tldr: Reddit TL;DR dataset
    - forums_export: Local Foru.ms JSONL export
    - cnn_dailymail: CNN/DailyMail news summarization
    
    Args:
        name: Dataset name
        split: Data split (train, validation, test)
        path: Path to local file (for forums_export)
        max_samples: Maximum samples to load
        
    Returns:
        List of example dictionaries
    """
    logger.info(f"Loading dataset: {name} ({split})")
    
    if name == "samsum":
        return _load_samsum(split, max_samples)
    elif name == "reddit_tldr":
        return _load_reddit_tldr(split, max_samples)
    elif name == "forums_export":
        return _load_forums_export(path, max_samples)
    elif name == "cnn_dailymail":
        return _load_cnn_dailymail(split, max_samples)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def _load_samsum(split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Load SAMSum dialogue summarization dataset."""
    try:
        from datasets import load_dataset as hf_load
        
        dataset = hf_load("samsum", split=split)
        
        examples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            examples.append({
                "id": item.get("id", f"samsum_{i}"),
                "source": item["dialogue"],
                "target": item["summary"],
                "metadata": {"source": "samsum"}
            })
        
        logger.info(f"Loaded {len(examples)} SAMSum examples")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to load SAMSum: {e}")
        return _get_samsum_fallback(max_samples)


def _get_samsum_fallback(max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Fallback SAMSum examples for testing without internet."""
    examples = [
        {
            "id": "samsum_demo_1",
            "source": "Hannah: Hey, do you have Betty's number?\nAmanda: Lemme check\nHannah: <file_gif>\nAmanda: Sorry, can't find it.\nAmanda: Ask Larry\nAmanda: He called her last week\nHannah: I don't want to bother him.\nHannah: I'll text him\nAmanda: OK\nAmanda: Good luck!",
            "target": "Hannah needs Betty's number. Amanda doesn't have it but suggests asking Larry, who called Betty last week.",
            "metadata": {"source": "samsum_fallback"}
        },
        {
            "id": "samsum_demo_2",
            "source": "Eric: TORTURE!!! And target tells us only 40min delivery. Impossible!\nRob: I know! Mine was supposed to be ready 12:15. Still waiting.\nEric: Target is a joke...\nRob: What did you get?\nEric: A new chair\nRob: Nice. I got a monitor stand",
            "target": "Eric and Rob are both waiting for delayed Target deliveries. Eric ordered a chair and Rob ordered a monitor stand.",
            "metadata": {"source": "samsum_fallback"}
        },
    ]
    if max_samples:
        examples = examples[:max_samples]
    return examples


def _load_reddit_tldr(split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Load Reddit TL;DR dataset."""
    try:
        from datasets import load_dataset as hf_load
        
        # Use webis/tldr-17 or similar
        dataset = hf_load("webis/tldr-17", split=split)
        
        examples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            examples.append({
                "id": f"reddit_{i}",
                "source": item.get("content", item.get("body", "")),
                "target": item.get("summary", item.get("tldr", "")),
                "metadata": {
                    "source": "reddit_tldr",
                    "subreddit": item.get("subreddit", "unknown")
                }
            })
        
        logger.info(f"Loaded {len(examples)} Reddit TL;DR examples")
        return examples
        
    except Exception as e:
        logger.warning(f"Failed to load Reddit TL;DR: {e}, using fallback")
        return _get_reddit_fallback(max_samples)


def _get_reddit_fallback(max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Fallback Reddit examples."""
    examples = [
        {
            "id": "reddit_demo_1",
            "source": "I've been working as a software engineer for 5 years and just got a job offer from a startup. The pay is 30% less but they're offering equity. The product seems interesting and the team is small. My current job is stable but boring. I'm 28 and have some savings. Should I take the risk?",
            "target": "Software engineer with 5 years experience considering startup offer with 30% pay cut but equity. Current job is stable but boring.",
            "metadata": {"source": "reddit_fallback", "subreddit": "cscareerquestions"}
        },
    ]
    if max_samples:
        examples = examples[:max_samples]
    return examples


def _load_forums_export(path: Optional[str], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Load local Foru.ms JSONL export."""
    if not path:
        logger.warning("No path provided for forums_export, using demo data")
        return _get_forums_demo_data(max_samples)
    
    examples = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            item = json.loads(line)
            
            # Combine posts into source text
            posts_text = "\n\n".join([
                f"[{p.get('author', 'Anonymous')}]: {p.get('content', '')}"
                for p in item.get("posts", [])
            ])
            
            examples.append({
                "id": item.get("thread_id", f"forums_{i}"),
                "source": posts_text,
                "target": item.get("summary", ""),
                "metadata": {
                    "source": "forums_export",
                    "thread_id": item.get("thread_id"),
                    "forum_id": item.get("forum_id"),
                    "post_count": len(item.get("posts", []))
                }
            })
    
    logger.info(f"Loaded {len(examples)} Foru.ms examples")
    return examples


def _get_forums_demo_data(max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Demo Foru.ms data for testing."""
    examples = [
        {
            "id": "forums_demo_1",
            "source": "[TechEnthusiast]: Has anyone tried the new M4 MacBook Pro? Thinking of upgrading from my M1.\n\n[DevOps_Sarah]: I upgraded last month. The performance jump is noticeable for Docker workloads. Build times dropped by about 40%.\n\n[TechEnthusiast]: That's impressive! How's the battery life?\n\n[DevOps_Sarah]: About the same as M1 honestly. Maybe slightly better under heavy load.",
            "target": "Discussion about M4 MacBook Pro upgrade from M1. User reports 40% faster Docker builds with similar battery life.",
            "metadata": {"source": "forums_demo"}
        },
    ]
    if max_samples:
        examples = examples[:max_samples]
    return examples


def _load_cnn_dailymail(split: str, max_samples: Optional[int]) -> List[Dict[str, Any]]:
    """Load CNN/DailyMail dataset."""
    try:
        from datasets import load_dataset as hf_load
        
        dataset = hf_load("cnn_dailymail", "3.0.0", split=split)
        
        examples = []
        for i, item in enumerate(dataset):
            if max_samples and i >= max_samples:
                break
            examples.append({
                "id": item.get("id", f"cnn_{i}"),
                "source": item["article"],
                "target": item["highlights"],
                "metadata": {"source": "cnn_dailymail"}
            })
        
        logger.info(f"Loaded {len(examples)} CNN/DailyMail examples")
        return examples
        
    except Exception as e:
        logger.error(f"Failed to load CNN/DailyMail: {e}")
        return []


def split_into_sentences(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with character offsets.
    
    Uses the same splitting rule as production ingestion for consistency.
    
    Args:
        text: Input text
        
    Returns:
        List of (sentence, start_offset, end_offset) tuples
    """
    # Standard sentence splitting regex
    pattern = r'(?<=[.!?])\s+'
    
    sentences = []
    last_end = 0
    
    for match in re.finditer(pattern, text):
        sentence = text[last_end:match.start()].strip()
        if sentence and len(sentence) > 10:  # Filter very short fragments
            sentences.append((sentence, last_end, match.start()))
        last_end = match.end()
    
    # Handle last sentence
    final = text[last_end:].strip()
    if final and len(final) > 10:
        sentences.append((final, last_end, len(text)))
    
    return sentences


def preprocess_for_summarization(
    examples: List[Dict[str, Any]],
    max_source_length: int = 512,
    max_target_length: int = 128,
) -> List[ProcessedExample]:
    """
    Preprocess examples for summarization training.
    
    Args:
        examples: Raw examples from load_dataset
        max_source_length: Maximum source tokens (for truncation guidance)
        max_target_length: Maximum target tokens
        
    Returns:
        List of ProcessedExample with sentences and offsets
    """
    processed = []
    
    for ex in examples:
        # Normalize text
        source = _normalize_text(ex["source"])
        target = _normalize_text(ex["target"])
        
        # Extract sentences with offsets
        sentence_data = split_into_sentences(source)
        sentences = [s[0] for s in sentence_data]
        offsets = [(s[1], s[2]) for s in sentence_data]
        
        processed.append(ProcessedExample(
            id=ex["id"],
            source_text=source,
            target_text=target,
            sentences=sentences,
            sentence_offsets=offsets,
            metadata=ex.get("metadata", {})
        ))
    
    logger.info(f"Preprocessed {len(processed)} examples")
    return processed


def _normalize_text(text: str) -> str:
    """Normalize text for training."""
    # Unicode normalization
    import unicodedata
    text = unicodedata.normalize("NFKC", text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip
    text = text.strip()
    
    return text


def yield_training_examples(
    config: Any,
    split: str = "train",
) -> Iterator[ProcessedExample]:
    """
    Yield preprocessed training examples.
    
    This is the main entry point for training data. It handles:
    - Dataset loading based on config
    - Preprocessing with sentence extraction
    - Optional caching
    
    Args:
        config: ExperimentConfig
        split: Data split
        
    Yields:
        ProcessedExample instances
    """
    # Load raw data
    max_samples = config.max_train_samples if split == "train" else config.max_eval_samples
    examples = load_dataset(
        name=config.dataset,
        split=split,
        path=config.dataset_path,
        max_samples=max_samples,
    )
    
    # Preprocess
    processed = preprocess_for_summarization(examples)
    
    for ex in processed:
        yield ex
