import torch.nn as nn
import torch
import wandb
from tqdm import tqdm
from transformers import ViTImageProcessor
from config import dendrite_config
from data_loader import load_medical_imaging_dataset, create_dataloaders
from dentmodel import create_dendritic_vit, freeze_vit_layers
from utils import set_seed, count_parameters, save_checkpoint

set_seed(dendrite_config.SEED)
wandb.init(project=dendrite_config.WANDB_PROJECT, name="vit-dendritic-v1")

print("[INFO] Loading dataset...")
dataset = load_medical_imaging_dataset()
processor = ViTImageProcessor.from_pretrained(dendrite_config.MODEL_NAME)
loaders = create_dataloaders(dataset, processor, dendrite_config.BATCH_SIZE)

print("[INFO] Creating DENDRITIC ViT...")
model = create_dendritic_vit(dendrite_config.MODEL_NAME, dendrite_config.NUM_CLASSES, dendrite_config)
model = model.to(dendrite_config.DEVICE)

# Freeze + dendrite setup
freeze_vit_layers(model, dendrite_config.FREEZE_LAYERS)

# Standard optimizer setup
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=dendrite_config.LEARNING_RATE,
    weight_decay=dendrite_config.WEIGHT_DECAY
)

def train_epoch(model, dataloader, optimizer, device):
    model.train(); total_loss, correct, total = 0, 0, 0
    for batch in tqdm(dataloader, desc="🌳 Dendritic Training"):
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(pixel_values=pixel_values, labels=labels)
        outputs.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += outputs.loss.item()
        predictions = torch.argmax(outputs.logits, 1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return total_loss/len(dataloader), 100*correct/total

best_acc = 0
for epoch in range(dendrite_config.NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, dendrite_config.DEVICE)

    # Log metrics
    current_params = count_parameters(model)
    wandb.log({
        'epoch': epoch+1, 'train_loss': train_loss, 'train_acc': train_acc,
        'params': current_params
    })
    print(f"Epoch {epoch+1}: {train_acc:.1f}% acc, {current_params:,} params")
    
    if train_acc > best_acc:
        best_acc = train_acc
        save_checkpoint(model, optimizer, epoch, best_acc, "models/dendritic_best.pt")

print(f"✅ DENDRITIC COMPLETE! Best: {best_acc:.1f}% | Params: {current_params:,}")
wandb.finish()
