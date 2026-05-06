"""
PAI-MONAI Integration Helpers
"""

import torch
import torch.optim as optim
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

def setup_pai_monai(model, config):
    """
    Wraps a MONAI model with PerforatedAI dendritic optimization.
    """
    # Configure PAI settings before initialization
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    # Reduce switch delay from default 10 to 3 for fast demo
    GPA.pc.set_n_epochs_to_switch(3)
    
    # Set forward function
    forward_fn_map = {
        'sigmoid': torch.sigmoid,
        'relu': torch.relu,
        'tanh': torch.tanh
    }
    GPA.pc.set_pai_forward_function(forward_fn_map[config['pai']['forward_function']])
    
    # Initialize PAI on the model
    # Medical models often have complex structures, PAI handles standard PyTorch modules well.
    model = UPA.initialize_pai(
        model,
        doing_pai=config['pai']['enabled'],
        save_name=config['pai']['save_name'],
        making_graphs=True,
        maximizing_score=config['pai']['maximizing_score']
    )
    
    return model

def configure_tracker(model, config):
    """
    Configures the PAI tracker with optimizer and scheduler.
    """
    optimizer_class = getattr(optim, config['optimizer']['type'])
    scheduler_class = getattr(optim.lr_scheduler, config['optimizer']['scheduler'])
    
    optim_args = {
        'params': model.parameters(),
        'lr': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay']
    }
    
    sched_args = config['optimizer']['scheduler_kwargs']
    
    GPA.pai_tracker.set_optimizer(optimizer_class)
    GPA.pai_tracker.set_scheduler(scheduler_class)
    
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
        model, optim_args, sched_args
    )
    
    return optimizer, scheduler
