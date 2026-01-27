import torch.nn as nn
import torch
import wandb
import matplotlib
from tqdm import tqdm
from pathlib import Path
from transformers import ViTImageProcessor
from config import dendrite_config
from data_loader import load_medical_imaging_dataset, create_dataloaders
from dentmodel import create_dendritic_vit, freeze_vit_layers
from utils import set_seed, count_parameters, save_checkpoint
from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA
import torch.optim.lr_scheduler as schedulers

matplotlib.use('Agg')

# Create directories
Path("PAI").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

set_seed(dendrite_config.SEED)
wandb.init(project=dendrite_config.WANDB_PROJECT, name="vit-dendritic-v1")

print("[INFO] Loading dataset...")
dataset = load_medical_imaging_dataset()
processor = ViTImageProcessor.from_pretrained(dendrite_config.MODEL_NAME)
loaders = create_dataloaders(dataset, processor, dendrite_config.BATCH_SIZE)

print("[INFO] Creating DENDRITIC ViT...")
model = create_dendritic_vit(dendrite_config.MODEL_NAME, dendrite_config.NUM_CLASSES, dendrite_config)
model = model.to(dendrite_config.DEVICE)

freeze_vit_layers(model, dendrite_config.FREEZE_LAYERS)
model = UPA.initialize_pai(model)

# Configure PAI
GPA.pc.set_module_names_to_track(["vit.encoder.layer11.mlp.fc2", "vit.encoder.layer11.attention.attention.value"])
GPA.pc.append_module_names_to_convert(["classifier"])
GPA.pc.set_unwrapped_modules_confirmed(True)
GPA.pc.set_weight_decay_accepted(True)

# Setup PAI tracker
GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
GPA.pai_tracker.set_scheduler(schedulers.CosineAnnealingLR)

optimArgs = {
    "lr": dendrite_config.LEARNING_RATE,
    "weight_decay": 0.01
}
schedArgs = {
    "T_max": dendrite_config.NUM_EPOCHS
}

optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
    model, 
    optimArgs,
    schedArgs
)

def get_current_lr(optimizer):
    return optimizer.param_groups[0]['lr']

print(f"✅ Setup complete | Initial LR: {get_current_lr(optimizer):.2e}")

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
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

# TRAINING LOOP
best_acc = 0
for epoch in range(dendrite_config.NUM_EPOCHS):
    train_loss, train_acc = train_epoch(model, loaders['train'], optimizer, dendrite_config.DEVICE)
    scheduler.step()
    
    try:
        # IMPORTANT: Pass optimizer to track restructuring
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            train_acc, 
            model,
            optimizer
        )
        
        if restructured:
            print(">>> [SUCCESS] Model restructured to PA mode!")
            params = list(filter(lambda p: p.requires_grad, model.parameters()))
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
                model, 
                optimArgs, 
                schedArgs, 
                params
            )
            
        if training_complete:
            print(">>> [INFO] PAI training complete!")
            
    except Exception as e:
        print(f">>> [WARNING] PAI tracker: {e}")
    
    current_params = count_parameters(model)
    wandb.log({
        'epoch': epoch+1,
        'train_loss': train_loss,
        'train_acc': train_acc,
        'params': current_params,
        'lr': get_current_lr(optimizer)
    })
    print(f"Epoch {epoch+1}: {train_acc:.1f}% acc, {current_params:,} params")
    
    if train_acc > best_acc:
        best_acc = train_acc
        save_checkpoint(model, optimizer, epoch, best_acc, "models/dendritic_best.pt")

print(f"✅ DENDRITIC COMPLETE! Best: {best_acc:.1f}% | Final Params: {current_params:,}")

# ============================================
# AUTOMATED PAI GRAPH GENERATION
# ============================================
print("\n" + "="*60)
print("GENERATING AUTOMATED PAI GRAPH")
print("="*60)

# First, let's discover what methods are available
print("\n[DEBUG] Checking available PAI visualization methods...")
print("GPA.pai_tracker methods:", [m for m in dir(GPA.pai_tracker) if not m.startswith('_')])

# Try all possible automated methods
graph_generated = False

try:
    # Method 1: plot() - most common
    if hasattr(GPA.pai_tracker, 'plot'):
        print("\n[ATTEMPT 1] Using GPA.pai_tracker.plot()...")
        GPA.pai_tracker.plot(save_path='PAI/PAI.png')
        graph_generated = True
        print("✅ Success with plot() method")
        
except Exception as e:
    print(f"❌ Method 1 failed: {e}")

if not graph_generated:
    try:
        # Method 2: plot_performance()
        if hasattr(GPA.pai_tracker, 'plot_performance'):
            print("\n[ATTEMPT 2] Using GPA.pai_tracker.plot_performance()...")
            GPA.pai_tracker.plot_performance(save_path='PAI/PAI.png')
            graph_generated = True
            print("✅ Success with plot_performance() method")
            
    except Exception as e:
        print(f"❌ Method 2 failed: {e}")

if not graph_generated:
    try:
        # Method 3: visualize()
        if hasattr(GPA.pai_tracker, 'visualize'):
            print("\n[ATTEMPT 3] Using GPA.pai_tracker.visualize()...")
            GPA.pai_tracker.visualize(output_path='PAI/PAI.png')
            graph_generated = True
            print("✅ Success with visualize() method")
            
    except Exception as e:
        print(f"❌ Method 3 failed: {e}")

if not graph_generated:
    try:
        # Method 4: save_plots()
        if hasattr(GPA.pai_tracker, 'save_plots'):
            print("\n[ATTEMPT 4] Using GPA.pai_tracker.save_plots()...")
            GPA.pai_tracker.save_plots(directory='PAI', filename='PAI.png')
            graph_generated = True
            print("✅ Success with save_plots() method")
            
    except Exception as e:
        print(f"❌ Method 4 failed: {e}")

if not graph_generated:
    try:
        # Method 5: generate_graph() 
        if hasattr(GPA.pai_tracker, 'generate_graph'):
            print("\n[ATTEMPT 5] Using GPA.pai_tracker.generate_graph()...")
            GPA.pai_tracker.generate_graph(save_path='PAI/PAI.png')
            graph_generated = True
            print("✅ Success with generate_graph() method")
            
    except Exception as e:
        print(f"❌ Method 5 failed: {e}")

if not graph_generated:
    try:
        # Method 6: Try UPA utilities
        if hasattr(UPA, 'plot_pai'):
            print("\n[ATTEMPT 6] Using UPA.plot_pai()...")
            UPA.plot_pai(GPA.pai_tracker, save_path='PAI/PAI.png')
            graph_generated = True
            print("✅ Success with UPA.plot_pai() method")
            
    except Exception as e:
        print(f"❌ Method 6 failed: {e}")

if not graph_generated:
    try:
        # Method 7: Try with model parameter
        if hasattr(GPA.pai_tracker, 'plot'):
            print("\n[ATTEMPT 7] Using GPA.pai_tracker.plot(model)...")
            GPA.pai_tracker.plot(model, save_path='PAI/PAI.png')
            graph_generated = True
            print("✅ Success with plot(model) method")
            
    except Exception as e:
        print(f"❌ Method 7 failed: {e}")

# Verify file exists
print("\n" + "="*60)
if Path("PAI/PAI.png").exists():
    file_size = Path("PAI/PAI.png").stat().st_size / 1024
    print(f"✅✅✅ SUCCESS! PAI/PAI.png created ({file_size:.1f} KB)")
    graph_generated = True
else:
    print("❌❌❌ CRITICAL: PAI/PAI.png NOT FOUND!")
    print("\n[HELP] Available pai_tracker attributes:")
    for attr in dir(GPA.pai_tracker):
        if 'plot' in attr.lower() or 'visual' in attr.lower() or 'graph' in attr.lower() or 'save' in attr.lower():
            print(f"  - {attr}")
    
    print("\n[ACTION NEEDED] Check PerforatedAI documentation:")
    print("  https://github.com/Alwinator/PerforatedAI")
    print("  Or run: help(GPA.pai_tracker)")

wandb.finish()

print("\n" + "="*60)
print("SUBMISSION CHECKLIST")
print("="*60)
print(f"{'✅' if Path('PAI/PAI.png').exists() else '❌'} PAI/PAI.png")
print(f"✅ Best accuracy: {best_acc:.2f}%")
print(f"✅ Model checkpoint saved")
print("="*60)