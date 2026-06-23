"""
Dataset Loading, Preprocessing, and W&B Artifact Versioning

Provides:
- Consistent sentence splitting (same as production for highlight anchoring)
- W&B Artifact registration for reproducibility
- Dataset versioning with metadata

Usage:
    from ml.datasets import (
        split_into_sentences,
        preprocess_for_training,
        save_and_register_dataset,
        download_dataset_artifact,
    )
    
    # Register a dataset
    artifact = save_and_register_dataset(data, "forums-v1", "lowlucy", "v0-ai-tldr-highlights")
    
    # Download in training
    path = download_dataset_artifact("lowlucy/v0-ai-tldr-highlights/forums-v1:v0")
"""

import os
import re
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Sentence splitting regex - MUST match production ingestion
SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Uses the SAME splitting rule as production ingestion so highlights anchor reliably.
    
    Args:
        text: Input text
        
    Returns:
        List of sentence strings
    """
    return [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]


def split_into_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    """
    Split text into sentences with character offsets.
    
    Args:
        text: Input text
        
    Returns:
        List of (sentence, start_offset, end_offset) tuples
    """
    sentences = []
    last_end = 0
    
    for match in SENT_SPLIT_RE.finditer(text):
        sentence = text[last_end:match.start()].strip()
        if sentence and len(sentence) > 5:
            sentences.append((sentence, last_end, match.start()))
        last_end = match.end()
    
    # Handle final sentence
    final = text[last_end:].strip()
    if final and len(final) > 5:
        sentences.append((final, last_end, len(text)))
    
    return sentences


@dataclass
class SentenceRecord:
    """A sentence with full provenance for training/eval."""
    thread_id: str
    post_id: str
    sentence_idx: int
    sentence: str
    char_start: int
    char_end: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def preprocess_thread_for_training(thread: Dict[str, Any]) -> List[SentenceRecord]:
    """
    Convert a thread dict into sentence records with provenance.
    
    thread format:
    {
        'thread_id': 't1',
        'posts': [
            {'post_id': 'p1', 'text': '...', 'author': '...'},
            ...
        ]
    }
    
    Returns:
        List of SentenceRecord objects with full provenance
    """
    records = []
    
    for post in thread.get('posts', []):
        post_id = post.get('post_id', post.get('id', ''))
        text = post.get('text', post.get('content', ''))
        
        sentences_with_offsets = split_into_sentences_with_offsets(text)
        
        for idx, (sentence, start, end) in enumerate(sentences_with_offsets):
            records.append(SentenceRecord(
                thread_id=thread.get('thread_id', ''),
                post_id=post_id,
                sentence_idx=idx,
                sentence=sentence,
                char_start=start,
                char_end=end,
            ))
    
    return records


def save_and_register_dataset(
    preprocessed_list: List[Dict[str, Any]],
    artifact_name: str,
    entity: str,
    project: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Save preprocessed data and register as W&B Artifact.
    
    Args:
        preprocessed_list: List of records to save
        artifact_name: Artifact name (e.g., 'dataset-samsum')
        entity: W&B entity (username or team)
        project: W&B project name
        metadata: Optional additional metadata
        
    Returns:
        wandb.Artifact object or None
    """
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed, skipping artifact registration")
        return None
    
    # Create local directory
    tmp_dir = Path('ml/artifacts/datasets') / artifact_name
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as JSONL
    jsonl_path = tmp_dir / 'data.jsonl'
    with jsonl_path.open('w', encoding='utf8') as f:
        for record in preprocessed_list:
            if hasattr(record, 'to_dict'):
                record = record.to_dict()
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    # Build metadata
    artifact_metadata = {
        'num_samples': len(preprocessed_list),
        'format': 'jsonl',
        'preprocessing_version': '1.0',
    }
    if metadata:
        artifact_metadata.update(metadata)
    
    # Register artifact
    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type='dataset',
            metadata=artifact_metadata,
        )
        artifact.add_file(str(jsonl_path))
        
        # Create a short-lived run for registration
        run = wandb.init(
            project=project,
            entity=entity,
            job_type='dataset_register',
            reinit=True,
        )
        run.log_artifact(artifact)
        run.finish()
        
        logger.info(f"Registered dataset artifact: {artifact_name}")
        return artifact
        
    except Exception as e:
        logger.error(f"Failed to register artifact: {e}")
        return None


def download_dataset_artifact(
    artifact_ref: str,
    out_dir: str = 'ml/data',
    project: Optional[str] = None,
) -> str:
    """
    Download a dataset artifact from W&B.
    
    Args:
        artifact_ref: Artifact reference (e.g., 'lowlucy/v0-ai-tldr-highlights/dataset-samsum:v0')
        out_dir: Local directory to download to
        project: Optional project name (extracted from artifact_ref if not provided)
        
    Returns:
        Path to downloaded artifact directory
    """
    try:
        import wandb
    except ImportError:
        raise ImportError("wandb required for artifact download")
    
    project = project or os.getenv("WANDB_PROJECT", "v0-ai-tldr-highlights")
    
    try:
        run = wandb.init(project=project, reinit=True, job_type='artifact_download')
        artifact = run.use_artifact(artifact_ref)
        artifact_dir = artifact.download(root=out_dir)
        run.finish()
        
        logger.info(f"Downloaded artifact to: {artifact_dir}")
        return artifact_dir
        
    except Exception as e:
        logger.error(f"Failed to download artifact: {e}")
        raise


def load_from_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load records from a JSONL file."""
    records = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records
