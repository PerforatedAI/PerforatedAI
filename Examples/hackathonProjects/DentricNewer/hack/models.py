import torch
from transformers import ViTForImageClassification
from perforatedai import globals_perforatedai as GPA, utils_perforatedai as UPA

def create_dendritic_vit(model_name, num_labels, config):
    model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_labels, ignore_mismatched_sizes=True)
    print(f"Baseline params: {sum(p.numel() for p in model.parameters()):,}")
    
    # THE WINNING LINE - Dendrite Injection
    model = UPA.initialize_pai(model)
    GPA.pai_tracker.set_dendrite_params(
        cycle_limit=config.DENDRITE_CYCLE_LIMIT,
        pruning_rate=config.PRUNING_RATE,
        dendritic_depth=config.DENDRITIC_DEPTH
    )
    return model

def freeze_vit_layers(model, num_frozen_blocks):
    for param in model.parameters(): param.requires_grad = False
    total_blocks = len(model.vit.encoder.layer)
    for i in range(total_blocks - num_frozen_blocks, total_blocks):
        for param in model.vit.encoder.layer[i].parameters(): param.requires_grad = True
    for param in model.classifier.parameters(): param.requires_grad = True
