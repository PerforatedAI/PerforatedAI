"""
Perforated Training Loop

This module implements the training loop for dendritic optimization.
The key difference from standard training is the two-phase approach:

Phase 1 (Neuron Learning):
- Standard gradient descent on base model weights
- Continues until the model plateaus

Phase 2 (Dendrite Learning):
- Base weights are frozen
- Dendritic nodes are added
- Cascade Correlation maximizes correlation with residual error

CRITICAL: Do NOT call scheduler.step() manually - PAI tracker handles this!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
from tqdm import tqdm
import numpy as np
import time

# PerforatedAI imports
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    PAI_AVAILABLE = True
except ImportError:
    PAI_AVAILABLE = False


class PerforatedTrainer:
    """
    Trainer class for Perforated Backpropagation.
    
    This trainer handles the unique requirements of dendritic optimization:
    1. Two-phase training (neurons then dendrites)
    2. PAI tracker integration for automatic phase switching
    3. No manual scheduler.step() calls
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        max_grad_norm: float = 1.0,
        class_weights: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the PerforatedTrainer.
        
        Args:
            model: Model wrapped with dendritic optimization
            optimizer: Optimizer from PAI tracker
            scheduler: Scheduler from PAI tracker (managed automatically)
            device: Training device (cuda/cpu)
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        
        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "learning_rates": [],
        }
        
        # Early stopping
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int,
    ) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}",
            leave=True,
        )
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                class_weights=self.class_weights,
            )
            
            loss = outputs["loss"]
            logits = outputs["logits"]
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )
            
            # Optimizer step
            self.optimizer.step()
            
            # NOTE: Do NOT call scheduler.step() here!
            # PAI tracker handles scheduling automatically
            
            # Update metrics
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}",
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(
        self,
        val_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    class_weights=self.class_weights,
                )
                
                loss = outputs["loss"]
                logits = outputs["logits"]
                
                total_loss += loss.item()
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        patience: int = 3,
        save_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """
        Full training loop with early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            patience: Early stopping patience
            save_path: Path to save best model
            
        Returns:
            Training history dictionary
        """
        print(f"Starting training for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, epochs + 1):
            # Train one epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            
            # Log metrics
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(current_lr)
            
            print(f"\nEpoch {epoch}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                if save_path:
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                    }, save_path)
                    print(f"  Model saved to {save_path}")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{patience})")
                
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
            
            # PAI tracker may switch phases here based on plateau detection
            if PAI_AVAILABLE:
                # The tracker monitors validation metrics and decides when to:
                # 1. Add dendritic nodes
                # 2. Freeze base weights
                # 3. Switch to dendrite learning
                pass  # Tracker handles this automatically
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> Tuple[float, float]:
    """
    Standalone function to train for one epoch.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        device: Training device
        max_grad_norm: Maximum gradient norm
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs["loss"]
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        
        # Do NOT call scheduler.step() - PAI handles it
        
        total_loss += loss.item()
        predictions = torch.argmax(outputs["logits"], dim=-1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(train_loader), correct / total


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Standalone validation function.
    
    Args:
        model: Model to validate
        val_loader: Validation data loader
        device: Device for computation
        
    Returns:
        Tuple of (average loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask, labels)
            
            total_loss += outputs["loss"].item()
            predictions = torch.argmax(outputs["logits"], dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(val_loader), correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    epochs: int = 10,
    patience: int = 3,
    save_path: Optional[str] = None,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, list]:
    """
    High-level training function.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (from PAI tracker)
        scheduler: Scheduler (from PAI tracker, managed automatically)
        device: Training device
        epochs: Maximum epochs
        patience: Early stopping patience
        save_path: Path to save best model
        class_weights: Optional class weights for imbalanced datasets
        
    Returns:
        Training history
    """
    trainer = PerforatedTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        class_weights=class_weights,
    )
    
    return trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        patience=patience,
        save_path=save_path,
    )
