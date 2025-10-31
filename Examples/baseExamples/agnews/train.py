"""
AG News BERT Classification

Usage:
    1. Edit config.json to set your parameters
    2. Run: python train_agnews.py
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt


class AGNewsTrainer(ABC):
    """Abstract base class for AG News training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_seed(config['seed'])
        
        # Metrics tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'test_loss': [], 'test_acc': []
        }
    
    def set_seed(self, seed: int):
        """Set random seeds for reproducibility"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def load_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Load and prepare AG News dataset"""
        print("Loading AG News dataset...")
        dataset = load_dataset("ag_news")
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_name'])
        
        # Create validation split
        train_val = dataset["train"].train_test_split(
            test_size=0.1, seed=self.config['seed']
        )
        
        def tokenize(examples):
            return tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length", 
                max_length=self.config['max_len']
            )
        
        # Tokenize and prepare
        train_ds = train_val["train"].map(tokenize, batched=True)
        val_ds = train_val["test"].map(tokenize, batched=True)
        test_ds = dataset["test"].map(tokenize, batched=True)
        
        # Rename label column
        train_ds = train_ds.rename_column("label", "labels")
        val_ds = val_ds.rename_column("label", "labels")
        test_ds = test_ds.rename_column("label", "labels")

        # Set format for PyTorch
        train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        val_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        
        # Create dataloaders
        train_loader = DataLoader(
            train_ds, 
            batch_size=self.config['batch_size'], 
            shuffle=True, 
            num_workers=self.config['num_workers']
        )
        val_loader = DataLoader(
            val_ds, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config['num_workers']
        )
        test_loader = DataLoader(
            test_ds, 
            batch_size=self.config['batch_size'], 
            shuffle=False, 
            num_workers=self.config['num_workers']
        )
        
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
        return train_loader, val_loader, test_loader
    
    def create_model(self) -> nn.Module:
        """Create BERT model for 4-class classification"""
        config = AutoConfig.from_pretrained(self.config['model_name'], num_labels=4)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model_name'], config=config
        )
        return model
    
    def train_epoch(
        self, 
        model: nn.Module, 
        loader: DataLoader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float]:
        """Train for one epoch"""
        model.train()
        total_loss, total_acc, total = 0, 0, 0
        
        for batch in tqdm(loader, desc="Training", leave=False):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            optimizer.zero_grad()
            logits = model(**inputs).logits
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            preds = logits.argmax(dim=1)
            acc = (preds == labels).float().mean()
            
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_acc += acc.item() * bs
            total += bs
        
        return total_loss / total, total_acc / total
    
    def evaluate(
        self, 
        model: nn.Module, 
        loader: DataLoader, 
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Evaluate on validation or test set"""
        model.eval()
        total_loss, total_acc, total = 0, 0, 0
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating", leave=False):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
                labels = batch['labels'].to(self.device)
                
                logits = model(**inputs).logits
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                acc = (preds == labels).float().mean()
                
                bs = labels.size(0)
                total_loss += loss.item() * bs
                total_acc += acc.item() * bs
                total += bs
        
        return total_loss / total, total_acc / total
    
    def print_metrics(
        self, 
        epoch: int, 
        train_loss: float, 
        train_acc: float,
        val_loss: float, 
        val_acc: float, 
        test_loss: float, 
        test_acc: float
    ):
        """Print training metrics"""
        print(f"Epoch {epoch:02d} | "
              f"Train: {train_loss:.4f}/{train_acc:.4f} | "
              f"Val: {val_loss:.4f}/{val_acc:.4f} | "
              f"Test: {test_loss:.4f}/{test_acc:.4f}")
    
    def plot_results(self, save_path: str = None):
        """Plot training curves"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', marker='o')
        axes[0].plot(self.history['val_loss'], label='Val Loss', marker='s')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[1].plot(self.history['train_acc'], label='Train Acc', marker='o')
        axes[1].plot(self.history['val_acc'], label='Val Acc', marker='s')
        axes[1].plot(self.history['test_acc'], label='Test Acc', marker='^')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Training, Validation, and Test Accuracy')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        
        plt.show()
    
    @abstractmethod
    def run(self) -> Dict:
        """Run training - to be implemented by subclasses"""
        pass


class BaselineTrainer(AGNewsTrainer):
    """Baseline trainer without PAI"""
    
    def run(self) -> Dict:
        print("\n" + "="*60)
        print("BASELINE TRAINING")
        print("="*60)
        
        # Setup
        train_loader, val_loader, test_loader = self.load_data()
        model = self.create_model().to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=self.config['lr'], 
            weight_decay=self.config['weight_decay']
        )
        
        # Training loop
        best_val_acc = 0.0
        best_test_acc = 0.0
        patience_counter = 0
        
        for epoch in range(1, self.config['max_epochs'] + 1):
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)
            test_loss, test_acc = self.evaluate(model, test_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            self.print_metrics(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_baseline.pt')
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"\nEarly stopping at epoch {epoch}")
                    break
        
        # Final results
        print("\n" + "="*60)
        print(f"Best Val Acc:  {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print(f"Test Acc at Best Val: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
        print("="*60)
        
        # Plot
        self.plot_results('baseline_results.png')
        
        return {
            'best_val_acc': best_val_acc,
            'best_test_acc': best_test_acc,
            'total_epochs': epoch
        }


class PAITrainer(AGNewsTrainer):
    """PAI trainer with dendrites"""
    
    def run(self) -> Dict:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA
        
        print("\n" + "="*60)
        print("PAI TRAINING")
        print("="*60)
        
        # Setup
        train_loader, val_loader, test_loader = self.load_data()
        model = self.create_model()
        
        # Initialize PAI
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_max_dendrites(self.config['max_dendrites'])
        GPA.pc.set_input_dimensions([-1, -1, 0])  # BERT has 3D tensors: [batch, seq, hidden]
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_weight_decay_accepted(True)
        
        model = UPA.initialize_pai(
            model, 
            save_name=self.config['pai_save_name'], 
            maximizing_score=True
        )
        model.to(self.device)
        
        # Fix input dimensions - BERT encoder layers are 3D, pooler/classifier are 2D
        print("\nSetting input dimensions...")
        for name, module in model.named_modules():
            if hasattr(module, 'set_this_input_dimensions'):
                # Check if this is a pooler or classifier layer (2D tensors)
                if 'pooler.dense' in name or 'classifier' in name:
                    module.set_this_input_dimensions([-1, 0])
                    print(f"  2D: {name}")
        print("Done setting dimensions.\n")
        
        # Setup optimizer through PAI
        criterion = nn.CrossEntropyLoss()
        GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
        optim_args = {
            'params': model.parameters(), 
            'lr': self.config['lr'], 
            'weight_decay': self.config['weight_decay']
        }
        optimizer = GPA.pai_tracker.setup_optimizer(model, optim_args)
        
        # Training loop
        best_val_acc = 0.0
        epoch = 0
        
        while True:
            epoch += 1
            
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc = self.evaluate(model, val_loader, criterion)
            test_loss, test_acc = self.evaluate(model, test_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            
            self.print_metrics(epoch, train_loss, train_acc, val_loss, val_acc, test_loss, test_acc)
            
            # Track metrics for PAI
            GPA.pai_tracker.add_extra_score(train_acc, 'Train Accuracy')
            GPA.pai_tracker.add_test_score(test_acc, 'Test Accuracy')
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            
            # PAI decision point
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
                val_acc, model
            )
            model.to(self.device)
            
            if training_complete:
                print('\n' + "="*60)
                print('PAI Training Complete!')
                print("="*60)
                break
            elif restructured:
                print('  >>> Dendrites added!')
                optim_args = {
                    'params': model.parameters(), 
                    'lr': self.config['lr'],
                    'weight_decay': self.config['weight_decay']
                }
                optimizer = GPA.pai_tracker.setup_optimizer(model, optim_args)
        
        # Final results
        final_test_loss, final_test_acc = self.evaluate(model, test_loader, criterion)
        print(f"\nFinal Test Acc: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
        
        # Display PAI-generated graphs
        self.display_pai_graphs()
        
        return {
            'best_val_acc': best_val_acc,
            'final_test_acc': final_test_acc,
            'total_epochs': epoch
        }
    
    def display_pai_graphs(self):
        """Display graphs generated by PAI"""
        from PIL import Image
        import glob
        
        pai_folder = self.config['pai_save_name']
        png_files = glob.glob(f"{pai_folder}/*.png")
        
        if not png_files:
            print(f"\nNo graphs found in {pai_folder}/")
            return
        
        print(f"\nDisplaying PAI graphs from {pai_folder}/:")
        for png_file in sorted(png_files):
            print(f"  - {os.path.basename(png_file)}")
            img = Image.open(png_file)
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.title(os.path.basename(png_file))
            plt.tight_layout()
            plt.show()


def load_config(config_path: str = "config.json") -> Dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("Configuration loaded:")
    print(json.dumps(config, indent=2))
    return config


def main():
    # Load config
    config = load_config()
    
    # Create and run trainer
    if config['mode'] == 'baseline':
        trainer = BaselineTrainer(config)
    elif config['mode'] == 'pai':
        trainer = PAITrainer(config)
    else:
        raise ValueError(f"Invalid mode: {config['mode']}. Must be 'baseline' or 'pai'")
    
    results = trainer.run()
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)

if __name__ == "__main__":
    main()