"""
Distributed Training Support for PAI Experiments

Provides utilities for multi-GPU and multi-node training with W&B.

Usage:
    # Launch with torchrun
    torchrun --nproc_per_node=4 ml/train.py --distributed
"""

import os
import logging
from typing import Optional, Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_distributed() -> int:
    """
    Initialize distributed training.
    
    Returns:
        Local rank (0 for main process)
    """
    if not dist.is_initialized():
        # Check if we're in a distributed environment
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                rank=rank,
                world_size=world_size,
            )
            
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
            
            logger.info(f"Distributed: rank={rank}, world_size={world_size}, local_rank={local_rank}")
            return local_rank
    
    return 0


def get_rank() -> int:
    """Get current process rank (0 if not distributed)."""
    if dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """Get world size (1 if not distributed)."""
    if dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wandb_init_distributed(
    project: str,
    config: dict,
    **kwargs,
) -> Optional[Any]:
    """
    Initialize W&B for distributed training.
    
    Only rank 0 initializes W&B to avoid duplicate logging.
    
    Args:
        project: W&B project name
        config: Configuration dict
        **kwargs: Additional wandb.init arguments
        
    Returns:
        wandb.Run on rank 0, None on other ranks
    """
    if not is_main_process():
        return None
    
    from utils.wandb_utils import wandb_init
    return wandb_init(project=project, config=config, **kwargs)


def log_distributed(run: Optional[Any], metrics: dict, step: int):
    """
    Log metrics in distributed training (only rank 0).
    
    Args:
        run: wandb.Run object (or None)
        metrics: Metrics dictionary
        step: Training step
    """
    if run is not None and is_main_process():
        from utils.wandb_utils import log_metrics
        log_metrics(step, metrics, run)


def all_reduce_metrics(metrics: dict) -> dict:
    """
    All-reduce metrics across processes.
    
    Args:
        metrics: Local metrics dictionary
        
    Returns:
        Averaged metrics across all processes
    """
    if not dist.is_initialized():
        return metrics
    
    world_size = get_world_size()
    reduced = {}
    
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            tensor = torch.tensor(value, device='cuda' if torch.cuda.is_available() else 'cpu')
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            reduced[key] = tensor.item() / world_size
        else:
            reduced[key] = value
    
    return reduced
