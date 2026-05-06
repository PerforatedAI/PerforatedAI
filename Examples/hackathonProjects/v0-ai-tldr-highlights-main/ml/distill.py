"""
Knowledge Distillation for Compressed Models

Provides teacher-student distillation to help compressed models recover performance.

Usage:
    from ml.distill import DistillationTrainer, distillation_step
    
    trainer = DistillationTrainer(teacher, student, temperature=2.0, alpha=0.5)
    loss = trainer.step(inputs, targets)
"""

import logging
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """
    Teacher-student distillation trainer.
    
    Trains student to match both:
    1. Ground truth labels (cross-entropy)
    2. Teacher soft predictions (KL divergence)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        temperature: float = 2.0,
        alpha: float = 0.5,
        device: str = 'cuda'
    ):
        """
        Initialize distillation trainer.
        
        Args:
            teacher: Pretrained teacher model (frozen)
            student: Student model to train
            temperature: Softmax temperature for soft targets
            alpha: Weight for distillation loss (1-alpha for CE loss)
            device: Target device
        """
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.temperature = temperature
        self.alpha = alpha
        self.device = device
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        logger.info(f"Distillation trainer initialized:")
        logger.info(f"  temperature={temperature}, alpha={alpha}")
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        targets: torch.Tensor,
        pad_token_id: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Student output logits [B, T, V]
            teacher_logits: Teacher output logits [B, T, V]
            targets: Ground truth labels [B, T]
            pad_token_id: Padding token ID to ignore
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Cross-entropy with ground truth
        ce_loss = F.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            targets.view(-1),
            ignore_index=pad_token_id
        )
        
        # KL divergence with teacher (soft targets)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)
        
        # Create mask for non-padding positions
        mask = (targets != pad_token_id).unsqueeze(-1).float()
        
        kl_loss = F.kl_div(
            student_soft * mask,
            teacher_soft * mask,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Combined loss
        total_loss = (1 - self.alpha) * ce_loss + self.alpha * kl_loss
        
        return total_loss, {
            'ce_loss': ce_loss.item(),
            'kl_loss': kl_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def step(
        self,
        batch: Dict[str, torch.Tensor],
        pad_token_id: int = 0
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform one distillation step.
        
        Args:
            batch: Dictionary with input_ids, attention_mask, labels
            pad_token_id: Padding token ID
            
        Returns:
            Tuple of (loss, metrics_dict)
        """
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get teacher predictions (no grad)
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            teacher_logits = teacher_outputs.logits
        
        # Get student predictions
        student_outputs = self.student(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        student_logits = student_outputs.logits
        
        # Compute distillation loss
        loss, metrics = self.distillation_loss(
            student_logits,
            teacher_logits,
            batch['labels'],
            pad_token_id
        )
        
        return loss, metrics


def distillation_step(
    student: nn.Module,
    teacher: nn.Module,
    inputs: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 2.0,
    alpha: float = 0.5,
    pad_token_id: int = 0
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Standalone distillation step function.
    
    Args:
        student: Student model
        teacher: Teacher model (frozen)
        inputs: Input token IDs
        attention_mask: Attention mask
        targets: Target token IDs
        temperature: Softmax temperature
        alpha: Distillation loss weight
        pad_token_id: Padding token ID
        
    Returns:
        Tuple of (loss, metrics_dict)
    """
    teacher.eval()
    
    # Teacher forward (no grad)
    with torch.no_grad():
        teacher_outputs = teacher(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=targets
        )
        teacher_logits = teacher_outputs.logits
    
    # Student forward
    student_outputs = student(
        input_ids=inputs,
        attention_mask=attention_mask,
        labels=targets
    )
    student_logits = student_outputs.logits
    
    # Cross-entropy
    ce_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1),
        ignore_index=pad_token_id
    )
    
    # KL divergence
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    
    kl_loss = F.kl_div(
        student_soft,
        teacher_soft,
        reduction='batchmean'
    ) * (temperature ** 2)
    
    total_loss = (1 - alpha) * ce_loss + alpha * kl_loss
    
    return total_loss, {
        'ce_loss': ce_loss.item(),
        'kl_loss': kl_loss.item(),
        'total_loss': total_loss.item()
    }


def create_teacher_targets(
    teacher: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: Any,
    device: str = 'cuda',
    max_length: int = 128
) -> list:
    """
    Generate teacher predictions for a dataset.
    
    Args:
        teacher: Pretrained teacher model
        dataloader: Data loader
        tokenizer: Tokenizer for decoding
        device: Target device
        max_length: Max generation length
        
    Returns:
        List of generated summaries
    """
    teacher.eval()
    teacher.to(device)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = teacher.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
            
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            all_predictions.extend(predictions)
    
    return all_predictions
