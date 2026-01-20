import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor
from config import base_config
from data_loader import load_medical_imaging_dataset, create_dataloaders
from utils import set_seed, count_parameters, save_checkpoint

import os
os.environ["WANDB_MODE"] = "offline"

set_seed(base_config.SEED)
wandb.init(project=base_config.WANDB_PROJECT, name="vit-baseline-medical", config=base_config.__dict__)

dataset = load_medical_imaging_dataset()
processor = ViTImageProcessor.from_pretrained(base_config.MODEL_NAME)
loaders = create_dataloaders(dataset, processor, base_config.BATCH_SIZE)

model = ViTForImageClassification.from_pretrained(base_config.MODEL_NAME, num_labels=base_config.NUM_CLASSES).to(base_config.DEVICE)
wandb.log({'baseline_parameters': count_parameters(model)})

optimizer = optim.AdamW(model.parameters(), lr=base_config.LEARNING_RATE, weight_decay=base_config.WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

def train_epoch(model, dataloader, optimizer, device):
    model.train(); total_loss, correct, total = 0, 0, 0
    for batch in tqdm(dataloader, desc="Training"):
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
    return total_loss / len(dataloader), 100 * correct / total

# Main training loop (50 epochs, early stopping)
best_val_acc = 0
for epoch in range(base_config.NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, base_config.DEVICE)
    # Validation, logging, checkpointing...
    wandb.log({'epoch': epoch+1, 'train_accuracy': train_acc})

wandb.finish()
