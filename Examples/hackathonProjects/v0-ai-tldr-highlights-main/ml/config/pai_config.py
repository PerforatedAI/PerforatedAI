"""
PerforatedAI Configuration Module

This module provides centralized configuration for dendritic optimization.
All PAI globals are set here for auditability and reproducibility.

Usage:
    from ml.config.pai_config import configure_pai_globals, PAIConfig
    
    config = PAIConfig(experiment_type="B", dendrite_capacity=8)
    configure_pai_globals(config)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ExperimentType(str, Enum):
    """
    Canonical experiment types for controlled comparison.
    
    A = Baseline: Full model, no compression, no dendrites
    B = Compressed + Dendrites: Architecture compressed WITH dendritic optimization  
    C = Compressed Control: Architecture compressed WITHOUT dendrites (ablation)
    """
    BASELINE = "A"
    COMPRESSED_DENDRITES = "B"
    COMPRESSED_CONTROL = "C"


class SwitchMode(str, Enum):
    """
    Dendritic training mode switching strategies.
    
    - NEURITE_FIRST: Start with neurite (weight) learning, then switch to dendrite
    - DENDRITE_FIRST: Start with dendrite learning, then switch to neurite
    - ALTERNATING: Alternate between neurite and dendrite phases
    """
    NEURITE_FIRST = "neurite_first"
    DENDRITE_FIRST = "dendrite_first"
    ALTERNATING = "alternating"


@dataclass
class PAIConfig:
    """
    Complete configuration for PerforatedAI dendritic optimization.
    
    This config controls the dendritic training regime including:
    - When to switch between neuron and dendrite learning phases
    - Dendrite capacity and saturation behavior
    - Output dimension handling for classification/generation
    - Checkpoint and graph generation settings
    
    Args:
        experiment_type: One of A (baseline), B (compressed+dendrites), C (compressed control)
        switch_mode: Strategy for switching between learning phases
        n_epochs_to_switch: Epochs in NEURITE phase before switching
        p_epochs_to_switch: Epochs in DENDRITE (perforated) phase before switching
        dendrite_capacity: Max capacity per dendritic unit (affects compression ratio)
        test_dendrite_capacity: Enable capacity testing mode for analysis
        output_dimensions: Model output dimensions (for classification heads)
        make_graphs: Generate dendritic learning behavior graphs
        checkpoint_frequency: Save checkpoints every N validation steps
    """
    
    # Experiment identification
    experiment_type: ExperimentType = ExperimentType.COMPRESSED_DENDRITES
    experiment_name: str = "dendritic_summarizer"
    run_id: Optional[str] = None
    
    # Mode switching configuration
    switch_mode: SwitchMode = SwitchMode.NEURITE_FIRST
    n_epochs_to_switch: int = 5  # Epochs in neurite phase
    p_epochs_to_switch: int = 3  # Epochs in dendrite/perforated phase
    
    # Dendrite configuration
    dendrite_capacity: int = 8  # Max capacity per dendritic unit
    test_dendrite_capacity: bool = False  # Enable capacity testing mode
    cap_n: bool = True  # Cap neurite learning when saturated
    
    # Model configuration
    output_dimensions: Optional[int] = None  # Set based on task (num_labels for classification)
    
    # Training behavior
    maximizing_score: bool = True  # True for accuracy/ROUGE, False for loss
    early_stopping_patience: int = 10
    
    # Checkpointing and monitoring
    make_graphs: bool = True  # Generate dendritic learning graphs
    checkpoint_frequency: int = 1  # Save every N validation cycles
    save_dir: str = "./outputs/pai_checkpoints"
    
    # Derived properties
    @property
    def do_pb(self) -> bool:
        """Whether to enable perforated backpropagation."""
        return self.experiment_type == ExperimentType.COMPRESSED_DENDRITES
    
    @property
    def save_name(self) -> str:
        """Deterministic save name from experiment metadata."""
        name = f"{self.experiment_name}_exp{self.experiment_type.value}"
        name += f"_cap{self.dendrite_capacity}"
        name += f"_n{self.n_epochs_to_switch}_p{self.p_epochs_to_switch}"
        if self.run_id:
            name += f"_{self.run_id}"
        return name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "experiment_type": self.experiment_type.value,
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "switch_mode": self.switch_mode.value,
            "n_epochs_to_switch": self.n_epochs_to_switch,
            "p_epochs_to_switch": self.p_epochs_to_switch,
            "dendrite_capacity": self.dendrite_capacity,
            "test_dendrite_capacity": self.test_dendrite_capacity,
            "cap_n": self.cap_n,
            "output_dimensions": self.output_dimensions,
            "maximizing_score": self.maximizing_score,
            "do_pb": self.do_pb,
            "save_name": self.save_name,
        }


def configure_pai_globals(config: PAIConfig) -> bool:
    """
    Configure PerforatedAI global settings.
    
    This function sets all PAI globals in one visible place for auditability.
    Must be called BEFORE model conversion and tracker initialization.
    
    Args:
        config: PAIConfig instance with desired settings
        
    Returns:
        True if PAI globals were successfully configured, False otherwise
    """
    try:
        import perforatedai.pb_globals as PBG
        
        # =====================================================
        # DENDRITIC OPTIMIZATION GLOBAL CONFIGURATION
        # =====================================================
        # These settings control how the dendritic algorithm
        # switches between neuron learning and dendrite learning,
        # and how capacity saturation is handled.
        # =====================================================
        
        # Switch mode: controls phase alternation strategy
        # 'neurite_first' = standard mode (learn weights, then prune/optimize)
        PBG.switch_mode = config.switch_mode.value
        
        # N epochs: number of epochs in NEURITE phase (weight learning)
        # Higher = more weight training before dendritic optimization kicks in
        PBG.n_epochs_to_switch = config.n_epochs_to_switch
        
        # P epochs: number of epochs in PERFORATED/DENDRITE phase  
        # Higher = more time for dendritic structure optimization
        PBG.p_epochs_to_switch = config.p_epochs_to_switch
        
        # Output dimensions: required for classification tasks
        # Set to num_labels for classifiers, None for generation tasks
        if config.output_dimensions is not None:
            PBG.output_dimensions = config.output_dimensions
        
        # Dendrite capacity testing: enables detailed capacity analysis
        # When enabled, training exits after capacity is reached and logs saturation behavior
        PBG.testingDendriteCapacity = config.test_dendrite_capacity
        
        # Cap N: whether to cap neurite learning when saturated
        # True = stop weight updates when dendrites are saturated (recommended)
        PBG.cap_n = config.cap_n
        
        logger.info("=" * 60)
        logger.info("PerforatedAI Global Configuration")
        logger.info("=" * 60)
        logger.info(f"  switch_mode:            {PBG.switch_mode}")
        logger.info(f"  n_epochs_to_switch:     {PBG.n_epochs_to_switch}")
        logger.info(f"  p_epochs_to_switch:     {PBG.p_epochs_to_switch}")
        logger.info(f"  output_dimensions:      {getattr(PBG, 'output_dimensions', 'auto')}")
        logger.info(f"  testingDendriteCapacity:{PBG.testingDendriteCapacity}")
        logger.info(f"  cap_n:                  {PBG.cap_n}")
        logger.info("=" * 60)
        
        return True
        
    except ImportError:
        logger.warning("perforatedai not installed - PAI globals not configured")
        return False
    except Exception as e:
        logger.error(f"Failed to configure PAI globals: {e}")
        return False


def get_default_config_for_task(task: str) -> PAIConfig:
    """
    Get recommended PAI configuration for common tasks.
    
    Args:
        task: One of 'sentence_classification', 'summarization', 'embedding'
        
    Returns:
        PAIConfig with task-appropriate defaults
    """
    if task == "sentence_classification":
        return PAIConfig(
            experiment_name="sentence_classifier",
            output_dimensions=6,  # 6 highlight categories
            n_epochs_to_switch=5,
            p_epochs_to_switch=3,
            dendrite_capacity=8,
            maximizing_score=True,  # Optimize for accuracy
        )
    
    elif task == "summarization":
        return PAIConfig(
            experiment_name="summarizer",
            output_dimensions=None,  # Generation task
            n_epochs_to_switch=3,
            p_epochs_to_switch=2,
            dendrite_capacity=16,  # Larger capacity for seq2seq
            maximizing_score=True,  # Optimize for ROUGE
        )
    
    elif task == "embedding":
        return PAIConfig(
            experiment_name="sentence_encoder",
            output_dimensions=None,  # Embedding task
            n_epochs_to_switch=5,
            p_epochs_to_switch=3,
            dendrite_capacity=8,
            maximizing_score=True,  # Optimize for retrieval accuracy
        )
    
    else:
        logger.warning(f"Unknown task '{task}', returning default config")
        return PAIConfig()
