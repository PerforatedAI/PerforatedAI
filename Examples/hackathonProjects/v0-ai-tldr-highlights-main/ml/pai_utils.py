"""
PerforatedAI Utility Functions

Helper functions for common PAI operations:
- Tracker initialization
- Optimizer lifecycle management
- Safetensors workarounds
- Graph generation and archiving

Usage:
    from ml.pai_utils import init_pai_tracker, apply_safetensors_workaround
    
    tracker = init_pai_tracker("my_experiment", maximizing_score=True)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import json

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def init_pai_tracker(
    save_name: str,
    maximizing_score: bool = True,
    make_graphs: bool = True,
    do_pb: bool = True,
) -> Optional[Any]:
    """
    Initialize PerforatedAI tracker with explicit arguments.
    
    Args:
        save_name: Deterministic name for artifacts (from ExperimentConfig.save_name)
        maximizing_score: True for accuracy/ROUGE, False for loss
        make_graphs: Whether to generate dendritic learning graphs
        do_pb: Whether to enable perforated backpropagation
        
    Returns:
        Initialized tracker, or None if PAI not available
    """
    try:
        import perforatedai as PA
        
        tracker = PA.PerforatedBackPropagationTracker(
            do_pb=do_pb,
            save_name=save_name,
            maximizing_score=maximizing_score,
            make_graphs=make_graphs,
        )
        
        logger.info(f"PAI tracker initialized: {save_name}")
        logger.info(f"  do_pb={do_pb}, maximizing_score={maximizing_score}, make_graphs={make_graphs}")
        
        return tracker
        
    except ImportError:
        logger.warning("perforatedai not installed - tracker not initialized")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize PAI tracker: {e}")
        return None


def setup_pai_optimizer(
    tracker: Any,
    model: nn.Module,
    optimizer_class: type = None,
    scheduler_class: type = None,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    **scheduler_kwargs,
) -> Optional[torch.optim.Optimizer]:
    """
    Setup optimizer via PAI tracker for proper lifecycle management.
    
    The tracker manages optimizer reconstruction when dendritic rewiring occurs.
    
    Args:
        tracker: Initialized PAI tracker
        model: The model being trained
        optimizer_class: Optimizer class (default: AdamW)
        scheduler_class: Scheduler class (default: CosineAnnealingLR)
        lr: Learning rate
        weight_decay: Weight decay
        **scheduler_kwargs: Additional scheduler arguments
        
    Returns:
        Configured optimizer, or None if setup fails
    """
    if tracker is None:
        logger.warning("No tracker provided - returning standard optimizer")
        optimizer_class = optimizer_class or torch.optim.AdamW
        return optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    try:
        optimizer_class = optimizer_class or torch.optim.AdamW
        scheduler_class = scheduler_class or torch.optim.lr_scheduler.CosineAnnealingLR
        
        # Default scheduler kwargs
        if "T_max" not in scheduler_kwargs:
            scheduler_kwargs["T_max"] = 10000
        
        tracker.setup_optimizer(
            optimizer_class,
            scheduler_class,
            lr=lr,
            weight_decay=weight_decay,
            **scheduler_kwargs,
        )
        
        logger.info(f"PAI optimizer configured: {optimizer_class.__name__}, lr={lr}")
        
        return tracker.optimizer
        
    except Exception as e:
        logger.error(f"Failed to setup PAI optimizer: {e}")
        return None


def add_validation_score(
    tracker: Any,
    model: nn.Module,
    score: float,
) -> tuple:
    """
    Add validation score to PAI tracker and handle rewiring.
    
    This is the core PAI feedback mechanism. After each evaluation:
    1. Tracker decides if rewiring should occur
    2. If restructured, optimizer must be reinitialized
    3. If training_complete, training should stop
    
    Args:
        tracker: PAI tracker
        model: Current model
        score: Validation score (higher = better if maximizing_score=True)
        
    Returns:
        Tuple of (model, improved, restructured, training_complete)
    """
    if tracker is None:
        return model, False, False, False
    
    try:
        result = tracker.add_validation_score(model, score)
        
        if isinstance(result, tuple) and len(result) >= 4:
            model, improved, restructured, training_complete = result[:4]
            
            if restructured:
                logger.info("Model RESTRUCTURED by dendritic algorithm")
            if training_complete:
                logger.info("PAI signaled TRAINING COMPLETE")
            
            return model, improved, restructured, training_complete
        else:
            return model, False, False, False
            
    except Exception as e:
        logger.error(f"Error adding validation score: {e}")
        return model, False, False, False


def apply_safetensors_workaround(enabled: bool = True):
    """
    Apply safetensors shared tensor workaround.
    
    Some HuggingFace models have tied weights that trigger safetensors
    validation errors after PAI conversion. This workaround bypasses
    the check when appropriate.
    
    WARNING: Only enable when you understand the model's weight sharing.
    
    Args:
        enabled: Whether to apply the workaround
    """
    if not enabled:
        return
    
    try:
        import safetensors.torch as sft
        
        # Store original for potential restoration
        _original_find_shared = getattr(sft, '_find_shared_tensors', None)
        
        def _patched_find_shared_tensors(*args, **kwargs):
            """Patched version that returns empty mapping."""
            return {}
        
        if hasattr(sft, '_find_shared_tensors'):
            sft._find_shared_tensors = _patched_find_shared_tensors
            logger.warning("Applied safetensors shared tensor workaround")
            logger.warning("Tied weights will not be validated during save")
        
    except ImportError:
        pass  # safetensors not installed
    except Exception as e:
        logger.warning(f"Failed to apply safetensors workaround: {e}")


def dump_pai_graphs(tracker: Any, output_dir: Path):
    """
    Export PAI tracker graphs to output directory.
    
    Args:
        tracker: PAI tracker with generated graphs
        output_dir: Directory to save graphs
    """
    if tracker is None:
        return
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # The tracker generates graphs during training
        # Look for graph files in the tracker's save directory
        import glob
        
        graph_patterns = ["*.png", "*.svg", "*.pdf"]
        for pattern in graph_patterns:
            for graph_file in glob.glob(f"{tracker.save_name}*{pattern}"):
                import shutil
                dest = output_dir / Path(graph_file).name
                shutil.copy(graph_file, dest)
                logger.info(f"Copied graph: {dest}")
        
    except Exception as e:
        logger.warning(f"Failed to dump PAI graphs: {e}")


def save_run_metadata(
    config: Any,
    output_dir: Path,
    pai_globals: Optional[Dict[str, Any]] = None,
    converted_modules: Optional[list] = None,
    wandb_run_id: Optional[str] = None,
):
    """
    Save comprehensive run metadata for reproducibility.
    
    Args:
        config: ExperimentConfig
        output_dir: Output directory
        pai_globals: Snapshot of PAI global settings
        converted_modules: List of converted module names
        wandb_run_id: W&B run ID if applicable
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "config": config.to_dict() if hasattr(config, 'to_dict') else str(config),
        "pai_globals": pai_globals or {},
        "converted_modules": converted_modules or [],
        "wandb_run_id": wandb_run_id,
    }
    
    # Add PAI global snapshot if available
    try:
        from perforatedai import pb_globals as PBG
        metadata["pai_globals"] = {
            "switch_mode": str(getattr(PBG, 'switch_mode', None)),
            "n_epochs_to_switch": getattr(PBG, 'n_epochs_to_switch', None),
            "p_epochs_to_switch": getattr(PBG, 'p_epochs_to_switch', None),
            "output_dimensions": getattr(PBG, 'output_dimensions', None),
            "testingDendriteCapacity": getattr(PBG, 'testingDendriteCapacity', None),
        }
    except ImportError:
        pass
    
    path = output_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Run metadata saved to {path}")


def get_dendrite_stats(model: nn.Module) -> Dict[str, Any]:
    """
    Get dendritic statistics from a PAI-converted model.
    
    Args:
        model: PAI-converted model
        
    Returns:
        Dictionary with dendrite counts and activity stats
    """
    stats = {
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "dendrite_modules": 0,
        "active_dendrites": 0,
    }
    
    # Count PAI-specific modules
    for name, module in model.named_modules():
        if "dendrit" in name.lower() or "perforat" in name.lower():
            stats["dendrite_modules"] += 1
    
    return stats
