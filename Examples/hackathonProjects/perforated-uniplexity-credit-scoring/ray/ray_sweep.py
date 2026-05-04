"""
Ray Hyperparameter Sweep
📈 Scalable hyperparameter tuning with Ray Tune
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search.bayesopt import BayesOptSearch
except ImportError:
    print("\n⚠️  Ray is not installed or not compatible with this Python version.")
    print("   Hyperparameter sweeps are disabled. Use standard training scripts instead.\n")
    sys.exit(0)

from models.dendritic_model import DendriticModel


def train_dendritic_tune(config):
    """Training function for Ray Tune"""
    
    # Load data (placeholder)
    features = torch.randn(100, 5)
    labels = torch.randn(100, 1)
    
    train_dataset = TensorDataset(features[:80], labels[:80])
    val_dataset = TensorDataset(features[80:], labels[80:])
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])
    
    # Initialize model
    model = DendriticModel(
        input_dim=5,
        hidden_dim=config["hidden_dim"],
        output_dim=1,
        num_dendrites=config["num_dendrites"]
    )
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    # Training loop
    for epoch in range(50):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Report to Ray Tune
        tune.report(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            epoch=epoch
        )


def run_ray_sweep():
    """Run hyperparameter sweep with Ray Tune"""
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    # Define search space
    search_space = {
        "lr": tune.loguniform(1e-4, 1e-2),
        "batch_size": tune.choice([16, 32, 64]),
        "hidden_dim": tune.choice([16, 24, 32]), # Optimized space around 24
        "num_dendrites": tune.choice([2, 4, 8])
    }
    
    # Bayesian optimization search
    search_alg = BayesOptSearch(
        metric="val_loss",
        mode="min"
    )
    
    # ASHA scheduler for early stopping
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=50,
        grace_period=10,
        reduction_factor=2
    )
    
    # Run tuning
    tuner = tune.Tuner(
        train_dendritic_tune,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            num_samples=20,
            scheduler=scheduler,
            search_alg=search_alg
        ),
        run_config=ray.train.RunConfig(
            name="dendritic_sweep",
            storage_path="../models/ray_tune_results"
        )
    )
    
    results = tuner.fit()
    
    # Get best result
    best_result = results.get_best_result(metric="val_loss", mode="min")
    
    print("\n" + "="*60)
    print("Ray Hyperparameter Sweep Completed!")
    print("="*60)
    print(f"Best config: {best_result.config}")
    print(f"Best validation loss: {best_result.metrics['val_loss']:.4f}")
    
    # Save results
    import pandas as pd
    results_df = results.get_dataframe()
    results_df.to_csv("../reports/ray_sweep_results.csv", index=False)
    print(f"\n✅ Results saved to reports/ray_sweep_results.csv")
    
    ray.shutdown()


if __name__ == "__main__":
    run_ray_sweep()
