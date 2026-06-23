"""
PerforatedAI Integration Helper Functions
"""

import torch
from perforatedai import globals_perforatedai as GPA


def setup_pai_model(model, config):
    """
    Initialize model with PerforatedAI
    
    Args:
        model: PyTorch model
        config: PAI configuration dictionary
    
    Returns:
        PAI-wrapped model
    """
    from perforatedai import utils_perforatedai as UPA
    
    # Configure PAI settings before initialization
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    # YOLO often shares activation modules, which PAI needs to know to avoid duplicate pointer errors
    # We add common shared activation paths in YOLOv8
    GPA.pc.append_module_names_to_not_save(['.model.1.act', '.model.4.act', '.model.6.act', '.model.9.act'])
    
    if 'forward_function' in config:
        forward_fn_map = {
            'sigmoid': torch.sigmoid,
            'relu': torch.relu,
            'tanh': torch.tanh
        }
        GPA.pc.set_pai_forward_function(forward_fn_map[config['forward_function']])
    
    if 'weight_init_multiplier' in config:
        GPA.pc.set_candidate_weight_initialization_multiplier(
            config['weight_init_multiplier']
        )
    
    if 'improvement_threshold' in config:
        GPA.pc.set_improvement_threshold(config['improvement_threshold'])
    
    # Initialize PAI
    model = UPA.initialize_pai(
        model,
        doing_pai=config.get('enabled', True),
        save_name=config.get('save_name', 'PAI'),
        making_graphs=config.get('making_graphs', True),
        maximizing_score=config.get('maximizing_score', True)
    )
    
    return model


def configure_pai_tracker(model, optimizer_class, scheduler_class, optim_args, sched_args):
    """
    Configure PAI tracker with optimizer and scheduler
    
    Args:
        model: PAI-wrapped model
        optimizer_class: Optimizer class (e.g., torch.optim.Adam)
        scheduler_class: Scheduler class (e.g., torch.optim.lr_scheduler.ReduceLROnPlateau)
        optim_args: Arguments for optimizer
        sched_args: Arguments for scheduler
    
    Returns:
        optimizer, scheduler
    """
    GPA.pai_tracker.set_optimizer(optimizer_class)
    GPA.pai_tracker.set_scheduler(scheduler_class)
    
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
        model, optim_args, sched_args
    )
    
    return optimizer, scheduler


def handle_restructure(model, optimizer_class, scheduler_class, optim_args, sched_args, device):
    """
    Handle model restructuring when dendrites are added
    
    Args:
        model: Current model
        optimizer_class: Optimizer class
        scheduler_class: Scheduler class
        optim_args: Optimizer arguments
        sched_args: Scheduler arguments
        device: Target device
    
    Returns:
        model, optimizer, scheduler (updated if restructured)
    """
    # Move model to device
    model = model.to(device)
    
    # Reinitialize optimizer and scheduler
    optimizer, scheduler = configure_pai_tracker(
        model, optimizer_class, scheduler_class, optim_args, sched_args
    )
    
    return model, optimizer, scheduler


def add_validation_score(model, score, device):
    """
    Add validation score and check for restructuring
    
    Args:
        model: Current model
        score: Validation score
        device: Target device
    
    Returns:
        model, restructured (bool), training_complete (bool)
    """
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        score, model
    )
    
    # Move model to device after potential restructuring
    model = model.to(device)
    
    return model, restructured, training_complete
