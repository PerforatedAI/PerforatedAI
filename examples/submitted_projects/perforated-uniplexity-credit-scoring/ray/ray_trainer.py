"""
Ray Distributed Trainer for Dendritic Models
🚀 Scale training across multiple nodes/GPUs
"""
import sys
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Robust Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

data_path_features = os.path.join(project_root, "data", "processed", "features.pt")
data_path_labels = os.path.join(project_root, "data", "processed", "labels.pt")



try:
    import ray
    from ray import train
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
except ImportError as e:
    print(f"\n⚠️  Ray Configuration Error: {e}")
    print("   Ensure you are in the correct environment (Python 3.11).")
    print("   Run: pip install ray[default] torch numpy\n")
    sys.exit(1)
except Exception as e:
    print(f"\n⚠️  Unexpected Error: {e}")
    sys.exit(1)

from models.dendritic_model import DendriticModel


def train_func(config):
    """Training function to be distributed by Ray"""
    
    # Setup device
    device = train.torch.get_device()
    
    # Load data (Robust Absolute Paths)
    try:
        features = torch.load(data_path_features)
        labels = torch.load(data_path_labels)
    except Exception as e:
        print(f"❌ Error loading data from {data_path_features}: {e}")
        return

    dataset = TensorDataset(features, labels)
    
    # Prepare distributed data loader
    train_loader = train.torch.prepare_data_loader(
        DataLoader(dataset, batch_size=config.get("batch_size", 32), shuffle=True)
    )
    
    # Initialize model (Optimized "Efficiency King" Architecture)
    model = DendriticModel(
        input_dim=5,
        hidden_dim=config.get("hidden_dim", 24), # Optimized from 64 -> 24
        output_dim=1,
        num_dendrites=config.get("num_dendrites", 4)
    )
    
    # Prepare model for distributed training
    model = train.torch.prepare_model(model)
    # model to device is handled by prepare_model usually, but good to be safe if manual
    
    # Setup optimizer and loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
    
    # Training loop
    epochs = config.get("epochs", 20)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Report metrics to Ray
        train.report({"loss": avg_loss, "epoch": epoch})
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")


def run_distributed_training():
    """Run distributed training with Ray"""
    
    print(f"Initializing Ray Training...")
    print(f"Project Root: {project_root}")
    
    # Initialize Ray
    # Force 127.0.0.1 to avoid Windows/Docker DNS issues (kubernetes.docker.internal)
    ray.init(ignore_reinit_error=True, _node_ip_address="127.0.0.1")
    
    # Training configuration (Optimized)
    train_config = {
        "lr": 1e-3,
        "batch_size": 32,
        "hidden_dim": 24, # Optimized Efficiency King
        "num_dendrites": 4,
        "epochs": 20
    }
    
    # Scaling configuration
    scaling_config = ScalingConfig(
        num_workers=1,  # Single worker to avoid Windows/Gloo issues
        use_gpu=torch.cuda.is_available(),
        resources_per_worker={"CPU": 1, "GPU": 0.5} if torch.cuda.is_available() else {"CPU": 1}
    )
    
    # Run configuration
    run_config = RunConfig(
        name="dendritic_distributed_optimized",
        storage_path=os.path.abspath(os.path.join(project_root, "models", "ray_results")),
        checkpoint_config=CheckpointConfig(
            num_to_keep=1
        )
    )
    
    # Create Ray trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=train_config,
        scaling_config=scaling_config,
        run_config=run_config
    )
    
    # Train
    print("Starting Distributed Training...")
    result = trainer.fit()
    
    print("\n" + "="*60)
    print("Distributed Training Completed!")
    print("="*60)
    print(f"Checkpoint: {result.checkpoint}")
    print(f"Metrics: {result.metrics}")
    
    ray.shutdown()


if __name__ == "__main__":
    run_distributed_training()
