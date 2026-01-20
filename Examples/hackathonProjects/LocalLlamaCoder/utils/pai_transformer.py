"""
PAI-Transformer Integration Utilities
Helper functions for integrating PerforatedAI with Transformer models
"""

import torch
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


def setup_pai_lora(model, forward_function="forward", correlation_threshold=0.001, save_name="PAI_LocalLlamaCoder"):
    """
    Initialize PerforatedAI on LoRA adapter layers
    
    Args:
        model: PEFT model with LoRA adapters
        forward_function: Name of the forward function
        correlation_threshold: Threshold for dendritic growth
        save_name: Name for PAI output files
        
    Returns:
        PAI-initialized model
    """
    print(f"🧠 Applying PerforatedAI to LoRA adapters...")
    
    # Configure PerforatedAI settings
    GPA.pc.set_weight_decay_accepted(True)
    
    # Initialize PAI on the model
    # Note: We target LoRA adapter layers specifically
    model = UPA.initialize_pai(
        model,
        forward_function=forward_function,
        correlation_threshold=correlation_threshold,
        save_name=save_name
    )
    
    print("✅ PAI initialization complete on LoRA layers")
    return model


def configure_transformer_tracker(model, optimizer_class, scheduler_class, optim_args, sched_args):
    """
    Configure PAI tracker for Transformer training with proper optimizer setup
    
    Args:
        model: PAI-initialized model
        optimizer_class: Optimizer class (e.g., torch.optim.AdamW)
        scheduler_class: Scheduler class (e.g., torch.optim.lr_scheduler.ReduceLROnPlateau)
        optim_args: Dictionary of optimizer arguments
        sched_args: Dictionary of scheduler arguments
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    print("📊 Configuring PAI tracker with optimizer and scheduler...")
    
    try:
        # Set optimizer and scheduler classes
        GPA.pai_tracker.set_optimizer(optimizer_class)
        GPA.pai_tracker.set_scheduler(scheduler_class)
        
        # Setup optimizer with PAI tracker
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
        
        print("✅ PAI tracker configured with optimizer and scheduler")
        return optimizer, scheduler
    except Exception as e:
        print(f"⚠️  Warning: Could not configure PAI tracker: {e}")
        print("   Falling back to manual optimizer setup")
        optimizer = optimizer_class(**optim_args)
        scheduler = scheduler_class(optimizer, **sched_args)
        return optimizer, scheduler


def add_validation_score_and_check_restructure(score, model):
    """
    Add validation score to PAI tracker and check for model restructuring
    
    Args:
        score: Validation score (higher is better, use negative loss if using loss)
        model: Current model
        
    Returns:
        Tuple of (model, restructured, training_complete)
    """
    try:
        # This is the critical PAI API call that handles dendritic growth
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(score, model)
        return model, restructured, training_complete
    except Exception as e:
        print(f"⚠️  Warning: Could not add validation score: {e}")
        return model, False, False


def add_training_score(score):
    """
    Add training score for tracking (optional but recommended)
    
    Args:
        score: Training score
    """
    try:
        GPA.pai_tracker.add_extra_score(score, 'Train')
    except Exception as e:
        print(f"⚠️  Warning: Could not add training score: {e}")


def add_test_score(score):
    """
    Add test score for tracking (optional but recommended)
    
    Args:
        score: Test score
    """
    try:
        GPA.pai_tracker.add_test_score(score, 'Test Accuracy')
    except Exception as e:
        print(f"⚠️  Warning: Could not add test score: {e}")


def optimize_attention_heads(model):
    """
    Target dendritic growth on self-attention layers
    
    This function identifies attention projection layers (Q, K, V, O)
    and marks them as priority targets for PAI optimization.
    
    Args:
        model: Transformer model
        
    Returns:
        List of targeted layer names
    """
    attention_layers = []
    
    for name, module in model.named_modules():
        # Look for attention projection layers
        if any(proj in name.lower() for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            attention_layers.append(name)
            # Mark as high-priority for PAI
            if hasattr(module, 'pai_priority'):
                module.pai_priority = 1.0
    
    print(f"🎯 Identified {len(attention_layers)} attention layers for PAI optimization")
    return attention_layers


def get_lora_parameters(model):
    """
    Extract LoRA parameters from model
    
    Args:
        model: PEFT model with LoRA
        
    Returns:
        Dictionary of LoRA parameters
    """
    lora_params = {}
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            lora_params[name] = param
    
    return lora_params


def estimate_speedup(baseline_tps, optimized_tps):
    """
    Calculate speedup percentage
    
    Args:
        baseline_tps: Baseline tokens per second
        optimized_tps: PAI-optimized tokens per second
        
    Returns:
        Speedup percentage
    """
    if baseline_tps == 0:
        return 0.0
    
    speedup = ((optimized_tps - baseline_tps) / baseline_tps) * 100
    return speedup


def print_pai_info():
    """Print PerforatedAI configuration information"""
    print("\n" + "=" * 60)
    print("PerforatedAI Configuration:")
    print("=" * 60)
    
    try:
        print(f"  Forward Function: {GPA.pc.forward_function}")
        print(f"  Correlation Threshold: {GPA.pc.correlation_threshold}")
        print(f"  Weight Decay Accepted: {GPA.pc.weight_decay_accepted}")
        print(f"  Tracker Enabled: {hasattr(GPA, 'pai_tracker')}")
    except Exception as e:
        print(f"  Could not retrieve PAI info: {e}")
    
    print("=" * 60 + "\n")
