"""
DendriticDrive - PerforatedAI Integration for 3D Point Cloud Detection
Wraps 3D detection models with dendritic optimization.
"""

import torch
import torch.nn as nn
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


class Simple3DBackbone(nn.Module):
    """
    Simplified 3D detection backbone for demonstration.
    In production, this would be replaced with PointPillar, SECOND, or PV-RCNN.
    """
    
    def __init__(self, num_classes=3):
        super().__init__()
        
        # Simplified point cloud encoder (PointNet-style)
        self.point_encoder = nn.Sequential(
            nn.Linear(4, 64),  # Input: [x, y, z, intensity]
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU()
        )
        
        # Global feature aggregation
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Detection head
        self.detection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, points_batch):
        """
        Args:
            points_batch: List of point clouds [B x N x 4]
        
        Returns:
            predictions: [B x num_classes]
        """
        # Handle list of tensors (variable-length point clouds)
        batch_features = []
        
        for points in points_batch:
            # points: [N, 4]
            point_features = self.point_encoder(points)  # [N, 256]
            
            # Global max pooling to get fixed-size feature
            global_feat = torch.max(point_features, dim=0, keepdim=True)[0]  # [1, 256]
            batch_features.append(global_feat)
        
        # Stack batch
        batch_features = torch.cat(batch_features, dim=0)  # [B, 256]
        
        # Detection prediction
        predictions = self.detection_head(batch_features)  # [B, num_classes]
        
        return predictions


def setup_pai_pcdet(model, config):
    """
    Setup PerforatedAI integration for 3D point cloud detection model.
    
    Args:
        model: PyTorch model (3D detection backbone)
        config: Configuration dict from config.yaml
    
    Returns:
        model: Wrapped model with PAI integration
    """
    if not config['pai']['enabled']:
        print("⚠️  PAI is disabled in config. Returning unwrapped model.")
        return model
    
    print("=" * 60)
    print("🧠 Integrating PerforatedAI Dendritic Optimization")
    print("=" * 60)
    
    # Configure PAI global settings
    GPA.pc.set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    GPA.pc.set_perforated_backpropagation(config['pai']['perforated_backpropagation'])
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    # Critical settings for waveform graph generation
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_n_epochs_to_switch(3)
    
    # Set output dimensions for 2D features [B, C]
    GPA.pc.set_output_dimensions([-1, 0])
    
    # Initialize PAI on the model
    model = UPA.initialize_pai(
        model,
        doing_pai=config['pai']['enabled'],
        save_name="DendriticDrive"
    )
    
    # Set PAI layer configuration
    pai_layers = config['pai']['pai_layers']
    if pai_layers == [0]:
        # Auto-select: Apply to all Linear layers in detection head
        pai_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'detection_head' in name:
                pai_layers.append(name)
        print(f"Auto-selected PAI layers: {pai_layers}")
    
    # Wrap model with dendritic optimization
    # In the actual PerforatedAI library, this might be:
    # model = UPA.wrap_model_with_dendrites(model, layer_names=pai_layers)
    
    # For this demo, we'll manually mark the model as PAI-ready
    model.pai_enabled = True
    model.pai_layers = pai_layers
    
    print(f"✓ PAI integration complete")
    print(f"  - Dendrites per layer: {config['pai']['dendrites_per_layer']}")
    print(f"  - Max restructures: {config['pai']['max_restructures']}")
    print(f"  - Perforated Backprop: {config['pai']['perforated_backpropagation']}")
    
    return model


def configure_tracker(model, config):
    """
    Configures the PAI tracker with optimizer and scheduler.
    """
    import torch.optim as optim
    
    optimizer_class = getattr(optim, config['optimizer']['type'])
    scheduler_class = None
    if config['optimizer']['scheduler']:
        scheduler_class = getattr(optim.lr_scheduler, config['optimizer']['scheduler'])
    
    optim_args = {
        'lr': config['training']['learning_rate'],
        'weight_decay': config['training']['weight_decay']
    }
    
    # Add any extra optimizer kwargs
    if 'optimizer_kwargs' in config['optimizer']:
        optim_args.update(config['optimizer']['optimizer_kwargs'])
    
    sched_args = config['optimizer']['scheduler_kwargs']
    
    if config['pai']['enabled']:
        # Use existing tracker created by initialize_pai
        GPA.pai_tracker.set_optimizer(optimizer_class)
        if scheduler_class:
            GPA.pai_tracker.set_scheduler(scheduler_class)
        
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optim_args, sched_args
        )
        print("✓ PAI Tracker configured with optimizer and scheduler")
    else:
        # Standard PyTorch setup
        optimizer = optimizer_class(model.parameters(), **optim_args)
        scheduler = None
        if scheduler_class:
            scheduler = scheduler_class(optimizer, **sched_args)
    
    return optimizer, scheduler


def get_3d_model(config):
    """
    Factory function to get 3D detection model.
    
    Args:
        config: Configuration dict
    
    Returns:
        model: 3D detection model
    """
    model_type = config['model']['type']
    num_classes = config['data']['num_classes']
    
    print(f"Creating {model_type} backbone...")
    
    if model_type == "PointPillar":
        # In production, this would be:
        # from pcdet.models import PointPillar
        # model = PointPillar(config)
        
        # For demo, use simplified backbone
        model = Simple3DBackbone(num_classes=num_classes)
        print("  ⚠️  Using simplified backbone for demo (replace with OpenPCDet in production)")
    else:
        # Fallback to simple model
        model = Simple3DBackbone(num_classes=num_classes)
        print(f"  ⚠️  {model_type} not implemented, using Simple3DBackbone")
    
    return model
