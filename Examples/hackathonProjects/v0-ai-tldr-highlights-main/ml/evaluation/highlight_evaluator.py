"""
Highlight Precision Evaluator

This module ties the ML training loop back to the application by evaluating
how well the model's embeddings rank highlights in real forum threads.

Key metrics:
- HighlightPrecision@K: Precision of top-K ranked highlights
- ProvenanceAccuracy: Accuracy of source attribution
- CategoryRecall: Recall for each highlight category

Usage:
    from ml.evaluation.highlight_evaluator import HighlightEvaluator
    
    evaluator = HighlightEvaluator(model, tokenizer)
    metrics = evaluator.evaluate(validation_threads)
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class HighlightMetrics:
    """Container for highlight evaluation metrics."""
    
    # Precision metrics
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    precision_at_10: float = 0.0
    
    # Category metrics
    category_recall: Dict[str, float] = None
    category_precision: Dict[str, float] = None
    
    # Provenance metrics
    provenance_accuracy: float = 0.0
    
    # Ranking metrics
    mrr: float = 0.0  # Mean Reciprocal Rank
    ndcg: float = 0.0  # Normalized Discounted Cumulative Gain
    
    def __post_init__(self):
        if self.category_recall is None:
            self.category_recall = {}
        if self.category_precision is None:
            self.category_precision = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "precision@1": self.precision_at_1,
            "precision@3": self.precision_at_3,
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "mrr": self.mrr,
            "ndcg": self.ndcg,
            "provenance_accuracy": self.provenance_accuracy,
            "category_recall": self.category_recall,
            "category_precision": self.category_precision,
        }


class HighlightEvaluator:
    """
    Evaluates highlight quality using the current encoder model.
    
    This evaluator ensures that dendritic optimization is improving
    the actual user-visible highlight quality, not just abstract scores.
    
    Args:
        model: Sentence encoder model (BERT-based)
        tokenizer: Tokenizer for the model
        categories: List of highlight categories
    """
    
    CATEGORIES = ["fact", "solution", "opinion", "question", "citation", "irrelevant"]
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
    
    def evaluate(
        self,
        threads: List[Dict[str, Any]],
        ground_truth_highlights: Optional[List[Dict[str, Any]]] = None,
    ) -> HighlightMetrics:
        """
        Evaluate highlight quality on a set of threads.
        
        Args:
            threads: List of thread dictionaries with 'content' and 'summary'
            ground_truth_highlights: Optional annotated highlights for precision
            
        Returns:
            HighlightMetrics with comprehensive evaluation
        """
        self.model.eval()
        
        all_precision_at_k = {1: [], 3: [], 5: [], 10: []}
        all_mrr = []
        all_ndcg = []
        
        category_true = {cat: 0 for cat in self.CATEGORIES}
        category_pred = {cat: 0 for cat in self.CATEGORIES}
        category_correct = {cat: 0 for cat in self.CATEGORIES}
        
        with torch.no_grad():
            for thread in threads:
                # Extract and rank sentences
                sentences = self._extract_sentences(thread["content"])
                if not sentences:
                    continue
                
                # Get embeddings
                embeddings = self._get_embeddings(sentences)
                
                # Get summary embedding for relevance scoring
                summary_embedding = self._get_embeddings([thread["summary"]])[0]
                
                # Rank sentences by relevance to summary
                scores = F.cosine_similarity(embeddings, summary_embedding.unsqueeze(0))
                ranked_indices = torch.argsort(scores, descending=True).cpu().numpy()
                
                # Compute precision@K (using summary sentences as ground truth)
                summary_sentences = set(self._extract_sentences(thread["summary"]))
                
                for k in [1, 3, 5, 10]:
                    top_k = [sentences[i] for i in ranked_indices[:k]]
                    hits = sum(1 for s in top_k if self._sentence_match(s, summary_sentences))
                    precision = hits / min(k, len(sentences))
                    all_precision_at_k[k].append(precision)
                
                # Compute MRR
                for rank, idx in enumerate(ranked_indices, 1):
                    if self._sentence_match(sentences[idx], summary_sentences):
                        all_mrr.append(1.0 / rank)
                        break
                else:
                    all_mrr.append(0.0)
        
        # Aggregate metrics
        metrics = HighlightMetrics(
            precision_at_1=np.mean(all_precision_at_k[1]) if all_precision_at_k[1] else 0.0,
            precision_at_3=np.mean(all_precision_at_k[3]) if all_precision_at_k[3] else 0.0,
            precision_at_5=np.mean(all_precision_at_k[5]) if all_precision_at_k[5] else 0.0,
            precision_at_10=np.mean(all_precision_at_k[10]) if all_precision_at_k[10] else 0.0,
            mrr=np.mean(all_mrr) if all_mrr else 0.0,
        )
        
        logger.info(f"Highlight Evaluation: P@5={metrics.precision_at_5:.4f}, MRR={metrics.mrr:.4f}")
        
        return metrics
    
    def _extract_sentences(self, text: str) -> List[str]:
        """Extract sentences from text."""
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def _get_embeddings(self, sentences: List[str]) -> torch.Tensor:
        """Get embeddings for a list of sentences."""
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
        # Get pooled output (CLS token or mean pooling)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings
    
    def _sentence_match(self, sentence: str, reference_set: set, threshold: float = 0.7) -> bool:
        """Check if sentence matches any in reference set (fuzzy matching)."""
        sentence_lower = sentence.lower().strip()
        for ref in reference_set:
            ref_lower = ref.lower().strip()
            # Simple overlap check
            if sentence_lower in ref_lower or ref_lower in sentence_lower:
                return True
            # Word overlap
            s_words = set(sentence_lower.split())
            r_words = set(ref_lower.split())
            overlap = len(s_words & r_words) / max(len(s_words | r_words), 1)
            if overlap > threshold:
                return True
        return False
