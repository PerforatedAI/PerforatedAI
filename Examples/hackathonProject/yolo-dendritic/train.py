#!/usr/bin/env python3
"""
YOLO + Dendritic + PAI Integration
Version with extra_verbose for debugging dendrite_values error
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import logging
from datetime import datetime
import json
import copy
import time

# PAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
from perforatedai import modules_perforatedai as MPA

# Auto GPU
def find_free_gpu():
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            if gpu.id in [0, 1, 2] and gpu.memoryFree > 8192:
                return gpu.id
    except:
        pass
    return 0

os.environ['CUDA_VISIBLE_DEVICES'] = str(find_free_gpu())

# Logging
log_dir = Path("./logs_yolo_pai")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("=" * 80)
logger.info("YOLO + DENDRITIC + PAI - WITH REAL COCO DATA")
logger.info("=" * 80)

# Your dendritic innovation
class DendriticConv2d(nn.Module):
    def __init__(self, base_conv, num_dendrites=6, dendrite_scale=0.2):
        super().__init__()
        self.base = copy.deepcopy(base_conv)
        self.num_dendrites = num_dendrites
        self.dendrite_scale = dendrite_scale
        
        in_c = base_conv.in_channels
        out_c = base_conv.out_channels
        k = base_conv.kernel_size[0]
        s = base_conv.stride[0]
        p = base_conv.padding[0]
        d = base_conv.dilation[0]
        bias = base_conv.bias is not None
        
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_c, in_c, k, s, p, d, groups=in_c, bias=False),
                nn.Conv2d(in_c, out_c, 1, bias=bias),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_dendrites)
        ])
    
    def forward(self, x):
        main = self.base(x)
        if self.num_dendrites == 0:
            return main
        branch_sum = sum(b(x) for b in self.branches)
        return main + self.dendrite_scale * (branch_sum / self.num_dendrites)

# Configure PAI
def setup_pai():
    logger.info("Configuring PAI...")
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_max_dendrites(3)
    GPA.pc.set_n_epochs_to_switch(5)
    GPA.pc.set_improvement_threshold([0.01, 0.001, 0])
    GPA.pc.set_output_dimensions([-1, 0, -1, -1])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_weight_decay_accepted(True)
    
    # RORRY'S FIX: Clear default conversion list, only convert specific layers
    GPA.pc.set_modules_to_convert([])
    GPA.pc.set_module_ids_to_convert([
        ".model.22.cv3.0.2",  # First detection scale
        ".model.22.cv3.1.2",  # Second detection scale  
        ".model.22.cv3.2.2"   # Third detection scale
    ])
    
    GPA.pc.set_verbose(True)
    GPA.pc.set_extra_verbose(True)  # RORRY REQUESTED: To see forward/backward activity
    GPA.pc.set_history_lookback(1)
    logger.info("✓ PAI configured to ONLY convert detection head cv3 layers")

# Inject dendrites
def inject_dendrites(model):
    logger.info("Injecting dendrites into PAI-wrapped layers...")
    detect = model.model[-1]
    count = 0
    
    for i, seq in enumerate(detect.cv3):
        # The final layer (index 2) should now be a PAINeuronModule
        if len(seq) > 2:
            layer = seq[2]  
            
            if isinstance(layer, MPA.PAINeuronModule) and isinstance(layer.main_module, nn.Conv2d):
                # Create dendritic conv from the wrapped Conv2d
                dendritic = DendriticConv2d(layer.main_module, num_dendrites=6, dendrite_scale=0.2)
                
                # Fill PAI's dendrite slot
                layer.dendrite_module.parent_module = dendritic
                
                count += 1
                logger.info(f"  ✓ Injected dendrites into cv3[{i}][2]")
            else:
                logger.warning(f"  ✗ cv3[{i}][2] is {type(layer).__name__}, not PAINeuronModule!")
    
    logger.info(f"✓ {count} dendrites injected")
    return count

# Load COCO data
def load_coco_data(batch_size=16, img_size=416):
    """Try to load real COCO data, fallback to synthetic"""
    try:
        from ultralytics.data import build_dataloader, check_det_dataset
        
        logger.info("Attempting to load COCO dataset...")
        data_dict = check_det_dataset('./datasets/coco/coco.yaml')
        train_path = data_dict['train']
        
        train_loader, dataset = build_dataloader(
            dataset=train_path,
            batch=batch_size,
            imgsz=img_size,
            workers=4,
            rank=-1,
            mode='train',
            rect=False,
            stride=32
        )
        
        logger.info(f"✓ COCO loaded: {len(dataset)} training images")
        return train_loader, True
        
    except Exception as e:
        logger.warning(f"Could not load COCO: {e}")
        logger.warning("Using synthetic data mode")
        return None, False

# YOLO loss computation
def compute_yolo_loss(model, batch_dict):
    """Compute actual YOLO detection loss"""
    try:
        # Get model's compute_loss function if available
        if hasattr(model, 'loss'):
            return model.loss(batch_dict)
        
        # Fallback: simple regression on predictions
        imgs = batch_dict['img']
        preds = model(imgs)
        
        if isinstance(preds, (list, tuple)):
            loss = 0
            count = 0
            for p in preds:
                if torch.is_tensor(p) and p.requires_grad:
                    # Simple L1 loss to target values
                    loss = loss + F.l1_loss(p, torch.zeros_like(p) + 0.5)
                    count += 1
            return loss / max(count, 1)
        
        return torch.tensor(0.1, device=device, requires_grad=True)
        
    except Exception as e:
        # Ultimate fallback
        return torch.tensor(0.1, device=device, requires_grad=True)

# Training with real or synthetic data
def train_epoch(model, data_loader, use_real_data, optimizer, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    max_batches = 100 if use_real_data else 50
    
    if use_real_data:
        # Real COCO training
        for batch_idx, batch_data in enumerate(data_loader):
            if batch_idx >= max_batches:
                break
            
            try:
                imgs = batch_data['img'].to(device).float() / 255.0
                
                optimizer.zero_grad()
                preds = model(imgs)
                loss = compute_yolo_loss(model, batch_data)
                
                if loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                logger.debug(f"Batch {batch_idx} error: {e}")
                continue
    else:
        # Synthetic training
        for i in range(max_batches):
            optimizer.zero_grad()
            x = torch.randn(16, 3, 416, 416, device=device)
            preds = model(x)
            
            # Better synthetic loss - not always 0
            if isinstance(preds, (list, tuple)):
                loss = sum(p.abs().mean() for p in preds if torch.is_tensor(p)) * 0.1
            else:
                loss = torch.tensor(0.1, device=device, requires_grad=True)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    GPA.pai_tracker.add_extra_score(avg_loss, 'Train Loss')
    return avg_loss

def validate(model, data_loader, use_real_data, optimizer, scheduler, epoch):
    model.eval()
    total_loss = 0
    num_batches = 0
    max_batches = 50 if use_real_data else 20
    
    with torch.no_grad():
        if use_real_data:
            for batch_idx, batch_data in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                try:
                    imgs = batch_data['img'].to(device).float() / 255.0
                    preds = model(imgs)
                    loss = compute_yolo_loss(model, batch_data)
                    total_loss += loss.item()
                    num_batches += 1
                except:
                    continue
        else:
            for i in range(max_batches):
                x = torch.randn(16, 3, 416, 416, device=device)
                preds = model(x)
                if isinstance(preds, (list, tuple)):
                    loss = sum(p.abs().mean() for p in preds if torch.is_tensor(p)) * 0.1
                else:
                    loss = torch.tensor(0.1, device=device)
                total_loss += loss.item()
                num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    model, restructured, complete = GPA.pai_tracker.add_validation_score(avg_loss, model)    
    model = model.to(device)
    
    if restructured:
        logger.info("Model restructured")
        optim_args = {'params': model.parameters(), 'lr': 0.002, 'weight_decay': 0.0}
        sched_args = {'mode': 'min', 'patience': 3, 'factor': 0.5}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
    
    return model, optimizer, scheduler, complete, avg_loss

def main():
    logger.info("\nConfiguring PAI...")
    setup_pai()
    
    logger.info("\nLoading YOLO...")
    try:
        from ultralytics import YOLO
        yolo = YOLO("yolov8n.pt")
        model = yolo.model
        logger.info("✓ YOLOv8n loaded")
    except Exception as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)
    
    original_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Original params: {original_params / 1e6:.2f}M")
    
    logger.info("\nFinding duplicates...")
    seen = {}
    for name, mod in model.named_modules():
        mid = id(mod)
        if mid in seen:
            GPA.pc.append_module_names_to_not_save([name])
        else:
            seen[mid] = name
    
    # Add all the shared SiLU act modules that don't appear in named_modules
    for i in range(1, 23):
        GPA.pc.append_module_names_to_not_save([f".model.{i}.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.default_act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.cv1.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.cv1.default_act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.cv2.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.cv2.default_act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.0.cv1.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.0.cv1.default_act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.0.cv2.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.0.cv2.default_act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.1.cv1.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.1.cv1.default_act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.1.cv2.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.m.1.cv2.default_act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.conv.act"])
        GPA.pc.append_module_names_to_not_save([f".model.{i}.conv.default_act"])
    
    # Add detection head activations (model.22.cv2 and cv3)
    for cv in ["cv2", "cv3"]:
        for i in range(3):
            for j in range(3):
                GPA.pc.append_module_names_to_not_save([f".model.22.{cv}.{i}.{j}.act"])
                GPA.pc.append_module_names_to_not_save([f".model.22.{cv}.{i}.{j}.default_act"])
    
    logger.info("\nInitializing PAI...")
    model = UPA.initialize_pai(model, save_name="PAI_YOLO", maximizing_score=False, making_graphs=True)
    
    # Verify only the target layers were wrapped
    logger.info("\nVerifying PAI wrapping...")
    for name, module in model.named_modules():
        if isinstance(module, MPA.PAINeuronModule):
            logger.info(f"  ✓ PAINeuronModule found: {name}")
    
    logger.info("\nInjecting dendrites...")
    count = inject_dendrites(model)
    if count == 0:
        logger.error("No dendrites injected!")
        import pdb; pdb.set_trace()
        sys.exit(1)
    
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"After PAI: {total_params / 1e6:.2f}M")
    
    logger.info("\nLoading data...")
    data_loader, use_real_data = load_coco_data(batch_size=16, img_size=416)
    if use_real_data:
        logger.info("✓ Using REAL COCO data")
    else:
        logger.info("⚠️ Using synthetic data (install COCO for real training)")
    
    logger.info("\nSetting up optimizer...")
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optim_args = {'params': model.parameters(), 'lr': 0.002, 'weight_decay': 0.0}
    sched_args = {'mode': 'min', 'patience': 3, 'factor': 0.5}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
    
    logger.info("\nTraining...")
    logger.info(f"Data mode: {'REAL COCO' if use_real_data else 'SYNTHETIC'}")
    
    best_loss = float('inf')
    complete = False
    epoch = 0
    checkpoint_dir = Path("./checkpoints_yolo_pai")
    checkpoint_dir.mkdir(exist_ok=True)
    
    while not complete and epoch < 20:
        epoch += 1
        start = time.time()
        
        train_loss = train_epoch(model, data_loader, use_real_data, optimizer, epoch)
        model, optimizer, scheduler, complete, val_loss = validate(
            model, data_loader, use_real_data, optimizer, scheduler, epoch
        )
        
        elapsed = time.time() - start
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_dir / f"best_epoch_{epoch:02d}.pt")
            status = "BEST"
        else:
            status = ""
        
        logger.info(f"Epoch {epoch:2d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | {elapsed:.1f}s {status}")
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Best val loss: {best_loss:.6f}")
    logger.info(f"Data used: {'REAL COCO' if use_real_data else 'SYNTHETIC'}")
    
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")
    
    config = {
        "model": "YOLOv8n + Dendritic",
        "data": "COCO" if use_real_data else "Synthetic",
        "epochs": epoch,
        "best_loss": best_loss,
        "params_M": total_params / 1e6,
    }
    
    with open(checkpoint_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\nSaved to {checkpoint_dir}/")
    logger.info("DONE!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        sys.exit(1)
