"""
W&B helper utilities for v0-ai-tldr-highlights.

Usage:
    from utils.wandb_utils import wandb_init, log_metrics, log_artifact_checkpoint, finish_wandb

    run = wandb_init(project="v0-ai-tldr-highlights", config=config_dict, tags=["baseline"])
    ...
    log_artifact_checkpoint(run, model, checkpoint_path, artifact_name="summarizer")
    finish_wandb(run)

SECURITY: Never commit API keys. Set WANDB_API_KEY as environment variable:
    export WANDB_API_KEY="<YOUR_KEY>"
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Only import wandb if available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed. Install with: pip install wandb")


def wandb_init(
    project: str,
    config: Dict[str, Any],
    run_name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    resume: bool = False,
    entity: Optional[str] = None,
) -> Optional[Any]:
    """
    Initialize a W&B run consistently.

    IMPORTANT: Do NOT set API key here. Export WANDB_API_KEY in the environment:
        export WANDB_API_KEY="<YOUR_KEY>"
    
    Or set WANDB_MODE=disabled to skip logging entirely.
    
    Args:
        project: W&B project name
        config: Configuration dictionary to log
        run_name: Optional run name (auto-generated if not provided)
        tags: Optional list of tags for filtering
        resume: Whether to resume a previous run
        entity: W&B entity (username or team)
    
    Returns:
        wandb.Run object or None if W&B is disabled/unavailable
    """
    if not WANDB_AVAILABLE:
        logger.warning("W&B not available - logging disabled")
        return None
    
    # Check if W&B is disabled via environment
    wandb_mode = os.getenv("WANDB_MODE", "online")
    if wandb_mode == "disabled":
        logger.info("W&B disabled via WANDB_MODE=disabled")
        return None
    
    # Check if already initialized
    if wandb.run is not None:
        logger.info("W&B run already initialized, returning existing run")
        return wandb.run
    
    # Generate deterministic run name if not supplied
    git_sha = os.getenv("GIT_COMMIT_SHA", os.getenv("GITHUB_SHA", "local"))[:7]
    run_name = run_name or f"{config.get('experiment_name', 'exp')}-{git_sha}"
    
    # Get entity from env if not provided
    entity = entity or os.getenv("WANDB_ENTITY")
    
    try:
        run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=config,
            tags=tags or [],
            resume="allow" if resume else None,
            settings=wandb.Settings(start_method="thread"),
            mode=wandb_mode,
        )
        logger.info(f"W&B initialized: {run.url}")
        return run
    except Exception as e:
        logger.error(f"W&B init failed: {e}")
        return None


def log_metrics(step: int, metrics: Dict[str, float], run: Optional[Any] = None):
    """
    Log a dictionary of metrics at a given step.
    
    Args:
        step: Training step number
        metrics: Dictionary of metric name -> value
        run: Optional wandb run (uses global run if not provided)
    """
    if not WANDB_AVAILABLE:
        return
    
    try:
        if run:
            run.log(metrics, step=step)
        elif wandb.run:
            wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"Failed to log metrics: {e}")


def log_artifact_checkpoint(
    run: Any,
    checkpoint_path: str,
    artifact_name: str,
    artifact_type: str = "model",
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """
    Save a model checkpoint as a W&B artifact.
    
    Args:
        run: wandb.Run object
        checkpoint_path: Path to checkpoint file or folder
        artifact_name: Name for the artifact (e.g., 'summarizer-v1')
        artifact_type: Type of artifact (default: 'model')
        metadata: Optional metadata dictionary
    
    Returns:
        wandb.Artifact object or None
    """
    if not WANDB_AVAILABLE or run is None:
        return None
    
    try:
        artifact = wandb.Artifact(
            name=artifact_name,
            type=artifact_type,
            metadata=metadata or {},
        )
        
        path = Path(checkpoint_path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        
        run.log_artifact(artifact)
        logger.info(f"Logged artifact: {artifact_name}")
        return artifact
    except Exception as e:
        logger.error(f"Failed to log artifact: {e}")
        return None


def log_table(
    run: Any,
    table_name: str,
    columns: List[str],
    data: List[List[Any]],
):
    """
    Log a table to W&B.
    
    Args:
        run: wandb.Run object
        table_name: Name for the table
        columns: List of column names
        data: List of rows (each row is a list of values)
    """
    if not WANDB_AVAILABLE or run is None:
        return
    
    try:
        table = wandb.Table(columns=columns, data=data)
        run.log({table_name: table})
    except Exception as e:
        logger.warning(f"Failed to log table: {e}")


def log_highlight_eval_table(run: Any, eval_examples: List[Dict[str, Any]]):
    """
    Log highlight evaluation results as a W&B table.
    
    Args:
        run: wandb.Run object
        eval_examples: List of dicts with keys:
            - thread_id: str
            - digest: str
            - gt_highlights: list
            - retrieved_highlights: list
            - precision: float
    """
    if not WANDB_AVAILABLE or run is None:
        return
    
    try:
        table = wandb.Table(columns=[
            "thread_id", "digest", "gt_highlights", "retrieved", "precision"
        ])
        
        for ex in eval_examples:
            table.add_data(
                ex.get("thread_id", ""),
                ex.get("digest", "")[:500],  # Truncate long text
                str(ex.get("gt_highlights", []))[:500],
                str(ex.get("retrieved_highlights", []))[:500],
                ex.get("precision", 0.0),
            )
        
        run.log({"highlight_eval": table})
        logger.info(f"Logged highlight eval table with {len(eval_examples)} examples")
    except Exception as e:
        logger.warning(f"Failed to log highlight eval table: {e}")


def log_pai_metrics(
    run: Any,
    model: Any,
    pai_tracker: Any,
    step: int,
):
    """
    Log PerforatedAI-specific metrics.
    
    Args:
        run: wandb.Run object
        model: PyTorch model
        pai_tracker: PAI tracker object
        step: Current training step
    """
    if not WANDB_AVAILABLE or run is None:
        return
    
    try:
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics = {
            "model/total_params": param_count,
            "model/trainable_params": trainable_params,
            "step": step,
        }
        
        # Get PAI-specific metrics if available
        if pai_tracker is not None:
            if hasattr(pai_tracker, "get_active_dendrite_ratio"):
                metrics["pai/active_dendrite_ratio"] = pai_tracker.get_active_dendrite_ratio()
            if hasattr(pai_tracker, "restructure_count"):
                metrics["pai/restructure_events"] = pai_tracker.restructure_count
            if hasattr(pai_tracker, "last_restructure_step"):
                metrics["pai/last_restructure_step"] = pai_tracker.last_restructure_step
        
        run.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"Failed to log PAI metrics: {e}")


def log_training_curve(run: Any, history: List[Dict[str, Any]]):
    """
    Log training curves as W&B plots.
    
    Args:
        run: wandb.Run object
        history: List of dicts with step, loss, rouge_l, etc.
    """
    if not WANDB_AVAILABLE or run is None:
        return
    
    try:
        import pandas as pd
        df = pd.DataFrame(history)
        
        # Log as table
        run.log({"training_history": wandb.Table(dataframe=df)})
        
        # Log as line plot if we have the right columns
        if "step" in df.columns and "rouge_l" in df.columns:
            run.log({
                "training_curve": wandb.plot.line_series(
                    xs=df["step"].tolist(),
                    ys=[df.get("rouge_l", []).tolist(), df.get("loss", []).tolist()],
                    keys=["rouge_l", "loss"],
                    title="Training Progress",
                    xname="step",
                )
            })
    except Exception as e:
        logger.warning(f"Failed to log training curve: {e}")


def save_eval_artifacts(
    run: Any,
    eval_results: Dict[str, Any],
    artifact_name: str = "eval_results",
) -> Optional[Any]:
    """
    Save evaluation results as a W&B artifact.
    
    Args:
        run: wandb.Run object
        eval_results: Dictionary of evaluation results
        artifact_name: Name for the artifact
    
    Returns:
        wandb.Artifact object or None
    """
    if not WANDB_AVAILABLE or run is None:
        return None
    
    try:
        import tempfile
        
        # Create temp directory and save JSON
        tmpdir = tempfile.mkdtemp()
        json_path = Path(tmpdir) / "eval_summary.json"
        
        with open(json_path, "w") as f:
            json.dump(eval_results, f, indent=2, default=str)
        
        # Create and log artifact
        artifact = wandb.Artifact(name=artifact_name, type="evaluation")
        artifact.add_file(str(json_path))
        run.log_artifact(artifact)
        
        logger.info(f"Saved eval artifacts: {artifact_name}")
        return artifact
    except Exception as e:
        logger.error(f"Failed to save eval artifacts: {e}")
        return None


def finish_wandb(run: Optional[Any]):
    """
    Safely finish a W&B run.
    
    Args:
        run: wandb.Run object (or None)
    """
    if run is None:
        return
    
    try:
        run.finish()
        logger.info("W&B run finished")
    except Exception as e:
        logger.warning(f"Error finishing W&B run: {e}")


# Convenience function for quick setup
def setup_wandb_for_training(
    experiment_name: str,
    config: Dict[str, Any],
    tags: Optional[List[str]] = None,
) -> Optional[Any]:
    """
    Quick setup for training with W&B.
    
    Args:
        experiment_name: Name of the experiment
        config: Training configuration
        tags: Optional tags
    
    Returns:
        wandb.Run object or None
    """
    project = os.getenv("WANDB_PROJECT", "v0-ai-tldr-highlights")
    
    return wandb_init(
        project=project,
        config=config,
        run_name=experiment_name,
        tags=tags,
    )
