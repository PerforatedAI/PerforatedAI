"""
Ray Utilities
🧠 Dataset sharding, metrics aggregation, and distributed utilities
"""
import torch
import numpy as np
from typing import List, Dict, Any
import ray


@ray.remote
class MetricsAggregator:
    """Distributed metrics aggregation"""
    
    def __init__(self):
        self.metrics = []
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Add metrics from a worker"""
        self.metrics.append(metrics)
    
    def get_aggregated_metrics(self) -> Dict[str, float]:
        """Aggregate metrics across all workers"""
        if not self.metrics:
            return {}
        
        aggregated = {}
        keys = self.metrics[0].keys()
        
        for key in keys:
            values = [m[key] for m in self.metrics]
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)
            aggregated[f"{key}_min"] = np.min(values)
            aggregated[f"{key}_max"] = np.max(values)
        
        return aggregated
    
    def reset(self):
        """Reset metrics"""
        self.metrics = []


def shard_dataset(dataset, num_shards: int, shard_id: int):
    """
    Shard dataset for distributed training
    
    Args:
        dataset: PyTorch dataset
        num_shards: Total number of shards
        shard_id: ID of current shard (0-indexed)
    
    Returns:
        Sharded dataset
    """
    total_size = len(dataset)
    shard_size = total_size // num_shards
    start_idx = shard_id * shard_size
    
    if shard_id == num_shards - 1:
        # Last shard gets remaining data
        end_idx = total_size
    else:
        end_idx = start_idx + shard_size
    
    indices = list(range(start_idx, end_idx))
    return torch.utils.data.Subset(dataset, indices)


def create_data_shards(features: torch.Tensor, labels: torch.Tensor, num_shards: int) -> List[tuple]:
    """
    Create data shards for distributed processing
    
    Args:
        features: Feature tensor
        labels: Label tensor
        num_shards: Number of shards to create
    
    Returns:
        List of (feature_shard, label_shard) tuples
    """
    total_size = features.shape[0]
    shard_size = total_size // num_shards
    
    shards = []
    for i in range(num_shards):
        start_idx = i * shard_size
        if i == num_shards - 1:
            end_idx = total_size
        else:
            end_idx = start_idx + shard_size
        
        feature_shard = features[start_idx:end_idx]
        label_shard = labels[start_idx:end_idx]
        shards.append((feature_shard, label_shard))
    
    return shards


@ray.remote
def compute_shard_metrics(model_state_dict: Dict, features: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """
    Compute metrics on a data shard
    
    Args:
        model_state_dict: Model state dictionary
        features: Feature tensor for this shard
        labels: Label tensor for this shard
    
    Returns:
        Dictionary of metrics
    """
    from models.dendritic_model import DendriticModel
    import torch.nn as nn
    
    # Load model
    model = DendriticModel(hidden_dim=24)
    model.load_state_dict(model_state_dict)
    model.eval()
    
    # Compute predictions
    with torch.no_grad():
        predictions = model(features)
    
    # Compute metrics
    criterion = nn.MSELoss()
    mse = criterion(predictions, labels).item()
    mae = torch.mean(torch.abs(predictions - labels)).item()
    
    return {
        "mse": mse,
        "mae": mae,
        "num_samples": features.shape[0]
    }


def distributed_evaluation(model, features: torch.Tensor, labels: torch.Tensor, num_workers: int = 4) -> Dict[str, float]:
    """
    Evaluate model in a distributed manner
    
    Args:
        model: PyTorch model
        features: All features
        labels: All labels
        num_workers: Number of Ray workers
    
    Returns:
        Aggregated metrics
    """
    # Get model state
    model_state_dict = model.state_dict()
    
    # Create data shards
    shards = create_data_shards(features, labels, num_workers)
    
    # Distribute computation
    futures = []
    for feature_shard, label_shard in shards:
        future = compute_shard_metrics.remote(model_state_dict, feature_shard, label_shard)
        futures.append(future)
    
    # Gather results
    shard_metrics = ray.get(futures)
    
    # Aggregate metrics
    total_samples = sum(m["num_samples"] for m in shard_metrics)
    weighted_mse = sum(m["mse"] * m["num_samples"] for m in shard_metrics) / total_samples
    weighted_mae = sum(m["mae"] * m["num_samples"] for m in shard_metrics) / total_samples
    
    return {
        "mse": weighted_mse,
        "mae": weighted_mae,
        "rmse": np.sqrt(weighted_mse)
    }


def save_distributed_checkpoint(model, optimizer, epoch: int, metrics: Dict, checkpoint_dir: str):
    """
    Save checkpoint in distributed training
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Training metrics
        checkpoint_dir: Directory to save checkpoint
    """
    import os
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics
    }
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


if __name__ == "__main__":
    # Test utilities
    print("Testing Ray utilities...")
    
    # Test data sharding
    features = torch.randn(100, 6)
    labels = torch.randn(100, 1)
    
    shards = create_data_shards(features, labels, num_shards=4)
    print(f"Created {len(shards)} shards")
    for i, (f_shard, l_shard) in enumerate(shards):
        print(f"  Shard {i}: {f_shard.shape[0]} samples")
    
    print("\n✅ Ray utilities ready!")
