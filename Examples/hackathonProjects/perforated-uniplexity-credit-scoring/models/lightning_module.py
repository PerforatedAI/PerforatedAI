"""
PyTorch Lightning Module for Credit Scoring
🔥 Provides training, validation, and testing logic
"""
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import Adam
from models.dendritic_model import DendriticModel


class CreditScoringLightningModule(pl.LightningModule):
    """Lightning module wrapper for credit scoring models"""
    
    def __init__(self, model_type='dendritic', learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model
        if model_type == 'dendritic':
            self.model = DendriticModel(input_dim=5)
        elif model_type == 'hybrid':
            # Import here to avoid circular dependencies if any
            from models.hybrid_model import HybridModel
            self.model = HybridModel(input_dim=5)
        else:
            from models.baseline_model import BaselineModel
            self.model = BaselineModel(input_dim=5)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.learning_rate = learning_rate
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        # Calculate accuracy
        preds = torch.sigmoid(y_hat) > 0.5
        acc = (preds == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        preds = torch.sigmoid(y_hat) > 0.5
        acc = (preds == y).float().mean()
        self.log('test_loss', loss)
        self.log('test_acc', acc)
        return loss
    
    def configure_optimizers(self):
        # Use AdamW and Cosine Scheduler as per improved methods
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


if __name__ == "__main__":
    # Test the Lightning module
    module = CreditScoringLightningModule()
    print(f"Lightning module initialized with {sum(p.numel() for p in module.parameters())} parameters")
