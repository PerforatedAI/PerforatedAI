"""
PerforatedAI Experiment Configuration (Canonical)

This is the single source of truth for all experiment configuration.
All PAI globals are documented and set here for complete auditability.

Usage:
    from ml.config import ExperimentConfig, configure_experiment
    
    config = ExperimentConfig.from_cli()
    configure_experiment(config)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any, List
from pathlib import Path
import logging
import os
import json

logger = logging.getLogger(__name__)


class ExperimentType(str, Enum):
    """
    Canonical experiment types for controlled PAI comparison.
    
    BASELINE (A): Full model, no compression, no dendrites
    COMPRESSED_PAI (B): Compressed architecture WITH dendritic optimization
    COMPRESSED_CONTROL (C): Compressed architecture WITHOUT dendrites (ablation)
    """
    BASELINE = "baseline"
    COMPRESSED_PAI = "compressed_pai"
    COMPRESSED_CONTROL = "compressed_control"


@dataclass
class ExperimentConfig:
    """
    Complete configuration for PerforatedAI experiments.
    
    This config contains all settings needed to reproduce an experiment:
    - Experiment identification and type
    - Dataset and model configuration
    - PAI-specific settings (switching, capacity)
    - Training hyperparameters
    - Logging and artifact paths
    """
    
    # === Experiment Identification ===
    experiment_name: str = "dendritic_summarizer"
    experiment_type: ExperimentType = ExperimentType.COMPRESSED_PAI
    run_id: Optional[str] = None
    seed: int = 42
    
    # === Dataset Configuration ===
    dataset: str = "samsum"  # samsum, reddit_tldr, forums_export
    dataset_path: Optional[str] = None  # For local JSONL
    max_train_samples: Optional[int] = None
    max_eval_samples: Optional[int] = None
    
    # === Model Configuration ===
    model_family: str = "t5"  # t5, bart, bert
    model_name: str = "t5-small"
    compression_ratio: float = 1.0  # 1.0 = no compression, 0.5 = 50% smaller
    
    # === PerforatedAI Settings ===
    # Mode switching policy (see pb_globals documentation)
    n_epochs_to_switch: int = 5  # Epochs in NEURITE phase before switching
    p_epochs_to_switch: int = 2  # Epochs in DENDRITE phase per cycle
    output_dimensions: List[int] = field(default_factory=lambda: [-1, 0, -1, -1])
    pai_testing_capacity: bool = False  # Enable capacity testing mode
    dendrite_capacity: int = 8
    
    # === Training Hyperparameters ===
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    max_steps: int = 100000  # High limit; PAI controls actual stopping
    eval_steps: int = 500
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    warmup_steps: int = 500
    
    # === Logging ===
    wandb_project: Optional[str] = "v0-pai-experiments"
    wandb_entity: Optional[str] = None
    wandb_enabled: bool = True
    log_level: str = "INFO"
    
    # === Artifacts ===
    output_dir: str = "./artifacts"
    save_steps: int = 1000
    save_total_limit: int = 3
    
    @property
    def use_pai(self) -> bool:
        """Whether to enable PerforatedAI dendritic optimization."""
        return self.experiment_type == ExperimentType.COMPRESSED_PAI
    
    @property
    def use_compression(self) -> bool:
        """Whether to compress the model architecture."""
        return self.experiment_type in [ExperimentType.COMPRESSED_PAI, ExperimentType.COMPRESSED_CONTROL]
    
    @property
    def save_name(self) -> str:
        """Deterministic save name for artifacts."""
        name = f"{self.dataset}-{self.model_family}-{self.experiment_type.value}"
        name += f"-c{int(self.compression_ratio * 100)}"
        name += f"-pai{int(self.use_pai)}"
        if self.run_id:
            name += f"-{self.run_id}"
        return name
    
    @property
    def artifact_dir(self) -> Path:
        """Directory for this experiment's artifacts."""
        return Path(self.output_dir) / self.experiment_type.value / self.save_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type.value,
            "run_id": self.run_id,
            "seed": self.seed,
            "dataset": self.dataset,
            "model_family": self.model_family,
            "model_name": self.model_name,
            "compression_ratio": self.compression_ratio,
            "use_pai": self.use_pai,
            "use_compression": self.use_compression,
            "n_epochs_to_switch": self.n_epochs_to_switch,
            "p_epochs_to_switch": self.p_epochs_to_switch,
            "output_dimensions": self.output_dimensions,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "max_steps": self.max_steps,
            "eval_steps": self.eval_steps,
            "wandb_project": self.wandb_project,
            "save_name": self.save_name,
        }
    
    def save(self, path: Optional[Path] = None):
        """Save config to JSON."""
        path = path or (self.artifact_dir / "config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Config saved to {path}")
    
    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """Load config from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        if "experiment_type" in data:
            data["experiment_type"] = ExperimentType(data["experiment_type"])
        return cls(**data)
    
    @classmethod
    def from_cli(cls) -> "ExperimentConfig":
        """Create config from command line arguments."""
        import argparse
        parser = argparse.ArgumentParser(description="PerforatedAI Experiment")
        
        parser.add_argument("--experiment-type", type=str, default="compressed_pai",
                          choices=["baseline", "compressed_pai", "compressed_control"])
        parser.add_argument("--dataset", type=str, default="samsum")
        parser.add_argument("--model-name", type=str, default="t5-small")
        parser.add_argument("--compression-ratio", type=float, default=1.0)
        parser.add_argument("--batch-size", type=int, default=16)
        parser.add_argument("--learning-rate", type=float, default=3e-4)
        parser.add_argument("--max-steps", type=int, default=100000)
        parser.add_argument("--eval-steps", type=int, default=500)
        parser.add_argument("--n-epochs-to-switch", type=int, default=5)
        parser.add_argument("--p-epochs-to-switch", type=int, default=2)
        parser.add_argument("--wandb", type=str, default="enabled", choices=["enabled", "disabled"])
        parser.add_argument("--wandb-project", type=str, default="v0-pai-experiments")
        parser.add_argument("--output-dir", type=str, default="./artifacts")
        parser.add_argument("--config-file", type=str, default=None)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--resume-from-checkpoint", type=str, default=None)
        
        args = parser.parse_args()
        
        # Load from YAML if provided
        if args.config_file:
            config = cls.from_yaml(args.config_file)
        else:
            config = cls()
        
        # Override with CLI args
        config.experiment_type = ExperimentType(args.experiment_type)
        config.dataset = args.dataset
        config.model_name = args.model_name
        config.compression_ratio = args.compression_ratio
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.max_steps = args.max_steps
        config.eval_steps = args.eval_steps
        config.n_epochs_to_switch = args.n_epochs_to_switch
        config.p_epochs_to_switch = args.p_epochs_to_switch
        config.wandb_enabled = args.wandb == "enabled"
        config.wandb_project = args.wandb_project
        config.output_dir = args.output_dir
        config.seed = args.seed
        
        return config


def configure_experiment(config: ExperimentConfig) -> bool:
    """
    Configure PerforatedAI globals for an experiment.
    
    This function sets all PAI globals in one visible place.
    MUST be called BEFORE model conversion.
    
    Args:
        config: ExperimentConfig with desired settings
        
    Returns:
        True if PAI was configured, False if PAI not available
    """
    if not config.use_pai:
        logger.info(f"Experiment {config.experiment_type.value}: PAI disabled")
        return False
    
    try:
        from perforatedai import pb_globals as PBG
        
        # =====================================================
        # PERFORATED AI GLOBAL CONFIGURATION
        # All settings documented per official API
        # =====================================================
        
        # switch_mode: Controls phase alternation strategy
        # doingHistory is recommended for NLP experiments
        PBG.switch_mode = PBG.doingHistory
        
        # n_epochs_to_switch: Epochs in NEURITE phase (weight learning)
        # Higher = more weight training before dendritic optimization
        PBG.n_epochs_to_switch = config.n_epochs_to_switch
        
        # p_epochs_to_switch: Epochs in PERFORATED phase (dendrite learning)
        # Higher = more time for dendritic structure optimization
        PBG.p_epochs_to_switch = config.p_epochs_to_switch
        
        # output_dimensions: Required for proper gradient flow
        # [-1, 0, -1, -1] is default; adjust based on model architecture
        PBG.output_dimensions = config.output_dimensions
        
        # testingDendriteCapacity: Enable for capacity analysis experiments
        PBG.testingDendriteCapacity = config.pai_testing_capacity
        
        logger.info("=" * 60)
        logger.info("PerforatedAI Globals Configured")
        logger.info("=" * 60)
        logger.info(f"  switch_mode:            {PBG.switch_mode}")
        logger.info(f"  n_epochs_to_switch:     {PBG.n_epochs_to_switch}")
        logger.info(f"  p_epochs_to_switch:     {PBG.p_epochs_to_switch}")
        logger.info(f"  output_dimensions:      {PBG.output_dimensions}")
        logger.info(f"  testingDendriteCapacity:{PBG.testingDendriteCapacity}")
        logger.info("=" * 60)
        
        return True
        
    except ImportError:
        logger.warning("perforatedai not installed - PAI globals not configured")
        return False
