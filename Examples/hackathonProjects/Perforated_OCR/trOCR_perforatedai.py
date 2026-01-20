"""
================================================================================
TrOCR + PerforatedAI: COMPLETE HACKATHON INTEGRATION & EVALUATION SUITE
================================================================================
Project: Perforated_OCR
Task: Handwritten Text Recognition Optimization
Fix: Correct implementation of add_validation_score and model restructuring.
================================================================================
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import json
import matplotlib.pyplot as plt
import time
from datetime import datetime
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

# ==============================================================================
# 1. ENVIRONMENT SETUP & AUTHENTICATION
# ==============================================================================
# These tokens are provided for the Hackathon integration
os.environ["PAIEMAIL"] = "hacker_token@perforatedai.com"
os.environ["PAITOKEN"] = "MdIq5V6gSmQM+sSak1imlCJ3tzvlyfHW8cUp+4FeQN9YxLKtwtl4HQIdmgQGmsJalAyoMtWgQVQagVOe2Bjr2THpWrxqPaU9xDnvPvRMxtYn6/bOWDqsv0Hs7td5R83rG8BMVzF8neYtxiiqrWX9XEOGlfGF8NHZVzy64C7maoO3OJiM3vDrKfhpGrAWJVV6RcGZZt/qpcraH86A2erhBhMWEbLbWqp8SRPqdJxL3mQJVcKTSe3sixQ20B3rZrRMpsfsjl0aNhZBTDhGcHzba8VTEam4k2+Sb3G5T3pWk5v7gVnFu5RN0Z0lRHeHMZ+j4VqudaOlJuH10MIQWm9Uqg=="
os.environ["WANDB_DISABLED"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    PAI_AVAILABLE = True
    print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ PerforatedAI Module Successfully Authenticated.")
    
    # Global PAI Optimization Settings
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_debugging_output_dimensions(0)

except ImportError as e:
    PAI_AVAILABLE = False
    print(f"❌ CRITICAL: PerforatedAI library failed to load. Check installation. {e}")

# ==============================================================================
# 2. GLOBAL HYPERPARAMETERS
# ==============================================================================
DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/PerforatedAI/Examples/hackathonProjects/Perforated_Scripts/data"
MODEL_NAME = "microsoft/trocr-base-handwritten"
BATCH_SIZE = 2
EPOCHS = 10  # Judges specifically requested a training trend over multiple epochs.
LEARNING_RATE = 5e-5
MAX_SAMPLES = 25
MAX_TARGET_LENGTH = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# 3. ROBUST DATASET ARCHITECTURE
# ==============================================================================
class TrOCRDataset(Dataset):
    """
    Custom Dataset for TrOCR. Extracts labels from filenames and 
    prepares pixel values and tokens.
    """
    def __init__(self, root_dir, processor, limit=MAX_SAMPLES):
        self.root_dir = root_dir
        self.processor = processor
        
        if not os.path.exists(root_dir):
            print(f"⚠ Warning: Data directory {root_dir} not found. Using placeholder logic.")
            self.image_files = []
        else:
            self.image_files = [
                f for f in os.listdir(root_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            ][:limit]
        
        print(f"📊 Dataset Status: {len(self.image_files)} samples loaded.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        path = os.path.join(self.root_dir, filename)
        
        # PIL Image handling
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new('RGB', (384, 384), color=(255, 255, 255))
            
        # Label Extraction: 'hello_world.png' -> 'Hello World'
        label = os.path.splitext(filename)[0].replace("_", " ").title()
        
        # Processor pipeline
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        
        # Tokenizer pipeline
        labels = self.processor.tokenizer(
            label, 
            padding="max_length", 
            max_length=MAX_TARGET_LENGTH, 
            truncation=True
        ).input_ids
        
        # CrossEntropyLoss ignore index is usually -100
        labels = [l if l != self.processor.tokenizer.pad_token_id else -100 for l in labels]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels),
            "original_text": label
        }

# ==============================================================================
# 4. PERFORATEDAI OPTIMIZED TRAINING (THE CORE FIX)
# ==============================================================================
def run_perforated_training():
    print("\n" + "="*80)
    print("PHASE 2: PERFORATEDAI INTEGRATED TRAINING & RESTRUCTURING")
    print("="*80)

    # Load Components
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    
    # Specific Transformer config
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    # Configure PAI exclusion modules (Safety for shared weights)
    GPA.pc.get_module_names_to_not_save().extend([
        '.decoder.model', '.decoder.base_model', '.decoder'
    ])

    # Convert model to PerforatedAI format
    print("[INIT] Converting model to Dendrite-ready architecture...")
    model = UPA.initialize_pai(model)

    # --------------------------------------------------------------------------
    # ARCHITECTURAL MAPPING (Crucial for TrOCR base model)
    # --------------------------------------------------------------------------
    print("[INIT] Mapping Vision (ViT) and Causal (RoBERTa) output dimensions...")
    try:
        # Encoder: 12 Layers of Vision Transformer
        for i in range(12):
            vit_layer = model.encoder.encoder.layer[i]
            # Mapping Attention Heads
            vit_layer.attention.attention.query.set_this_output_dimensions([-1, -1, 0])
            vit_layer.attention.attention.key.set_this_output_dimensions([-1, -1, 0])
            vit_layer.attention.attention.value.set_this_output_dimensions([-1, -1, 0])
            vit_layer.attention.output.dense.set_this_output_dimensions([-1, -1, 0])
            # Mapping Feed Forward Networks
            vit_layer.intermediate.dense.set_this_output_dimensions([-1, -1, 0])
            vit_layer.output.dense.set_this_output_dimensions([-1, -1, 0])

        # Decoder: 12 Layers of Causal Transformer
        for i in range(12):
            roberta_layer = model.decoder.model.decoder.layers[i]
            # Self-Attention Projection
            roberta_layer.self_attn.q_proj.set_this_output_dimensions([-1, -1, 0])
            roberta_layer.self_attn.k_proj.set_this_output_dimensions([-1, -1, 0])
            roberta_layer.self_attn.v_proj.set_this_output_dimensions([-1, -1, 0])
            roberta_layer.self_attn.out_proj.set_this_output_dimensions([-1, -1, 0])
            # Cross-Attention Projection (Encoder-Decoder)
            roberta_layer.encoder_attn.q_proj.set_this_output_dimensions([-1, -1, 0])
            roberta_layer.encoder_attn.k_proj.set_this_output_dimensions([-1, -1, 0])
            roberta_layer.encoder_attn.v_proj.set_this_output_dimensions([-1, -1, 0])
            roberta_layer.encoder_attn.out_proj.set_this_output_dimensions([-1, -1, 0])
            # Feed Forward Network
            roberta_layer.fc1.set_this_output_dimensions([-1, -1, 0])
            roberta_layer.fc2.set_this_output_dimensions([-1, -1, 0])
        print("✅ Success: All 24 Transformer layers mapped to PAI.")
    except Exception as e:
        print(f"⚠ Warning: Partial mapping achieved. Error: {e}")

    # Data Loader
    train_ds = TrOCRDataset(DATA_DIR, processor)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # PerforatedAI Optimizer and Tracker setup
    GPA.pai_tracker.set_optimizer(optim.AdamW)
    optimizer, _ = GPA.pai_tracker.setup_optimizer(model, {'params': model.parameters(), 'lr': LEARNING_RATE})

    model.to(DEVICE)
    results_history = {"loss": [], "accuracy": []}

    print(f"\n⚡ Commencing optimization for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        # Batch Training
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            optimizer.zero_grad()
            px = batch["pixel_values"].to(DEVICE)
            lb = batch["labels"].to(DEVICE)
            
            outputs = model(pixel_values=px, labels=lb)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_epoch_loss = running_loss / len(train_loader)
        
        # Validation Loop (Required for Dendrite trigger)
        model.eval()
        val_correct = 0
        total_validation = min(5, len(train_ds))
        
        with torch.no_grad():
            for i in range(total_validation):
                sample = train_ds[i]
                p_values = sample["pixel_values"].unsqueeze(0).to(DEVICE)
                output_ids = model.generate(p_values, max_length=MAX_TARGET_LENGTH)
                prediction = processor.decode(output_ids[0], skip_special_tokens=True).strip().lower()
                ground_truth = sample["original_text"].strip().lower()
                if prediction == ground_truth:
                    val_correct += 1
        
        epoch_accuracy = val_correct / total_validation
        
        # --- THE CRITICAL FIX: TRACKER + MODEL REASSIGNMENT ---
        # 1. Log the training loss to the tracker
        GPA.pai_tracker.add_extra_score(avg_epoch_loss, 'Train Loss')
        
        # 2. Call validation score and capture the returned "restructured_model"
        # This is the step that allows PAI to add dendrites to improve the model.
        restructured_model, model_was_changed, train_complete = GPA.pai_tracker.add_validation_score(epoch_accuracy, model)
        
        if model_was_changed:
            print(f"  ⚡ [RESTRUCTURE] PerforatedAI has added dendrites. Updating model architecture...")
            model = restructured_model  # REASSIGNMENT IS KEY

        results_history["loss"].append(avg_epoch_loss)
        results_history["accuracy"].append(epoch_accuracy)
        print(f"   📊 Epoch Summary: Loss={avg_epoch_loss:.4f} | Accuracy={epoch_accuracy:.2%}")

    return model, results_history

# ==============================================================================
# 5. BASELINE EVALUATION (FOR COMPARISON GRAPH)
# ==============================================================================
def run_baseline_training():
    print("\n" + "="*80)
    print("PHASE 1: BASELINE TrOCR TRAINING (NO OPTIMIZATION)")
    print("="*80)
    
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)

    dataset = TrOCRDataset(DATA_DIR, processor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    baseline_history = {"loss": [], "accuracy": []}

    for epoch in range(EPOCHS):
        model.train()
        total_l = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            px, lb = batch["pixel_values"].to(DEVICE), batch["labels"].to(DEVICE)
            loss = model(pixel_values=px, labels=lb).loss
            loss.backward()
            optimizer.step()
            total_l += loss.item()
        
        avg_l = total_l / len(dataloader)
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for i in range(5):
                item = dataset[i]
                gen = model.generate(item["pixel_values"].unsqueeze(0).to(DEVICE))
                if processor.decode(gen[0], skip_special_tokens=True).strip().lower() == item["original_text"].strip().lower():
                    correct += 1
        
        acc = correct / 5
        baseline_history["loss"].append(avg_l)
        baseline_history["accuracy"].append(acc)
        print(f"   Baseline Epoch {epoch+1}: Loss={avg_l:.4f} | Accuracy={acc:.2%}")

    return baseline_history

# ==============================================================================
# 6. REPORT GENERATION & VISUALIZATION
# ==============================================================================
def create_submission_report(baseline_h, pai_h):
    print("\n" + "="*80)
    print("PHASE 3: COMPILING PERFORMANCE ARTIFACTS")
    print("="*80)

    epochs_range = range(1, EPOCHS + 1)
    
    plt.figure(figsize=(16, 8))
    
    # Chart 1: Loss Reduction
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, baseline_h["loss"], 'r--', label='Baseline Loss')
    plt.plot(epochs_range, pai_h["loss"], 'g-', marker='o', linewidth=2, label='PerforatedAI Loss')
    plt.title('Training Loss Trend (Comparison)')
    plt.xlabel('Training Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Chart 2: Accuracy Improvements
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, baseline_h["accuracy"], 'r--', label='Baseline Accuracy')
    plt.plot(epochs_range, pai_h["accuracy"], 'b-', marker='s', linewidth=2, label='PerforatedAI Accuracy')
    plt.title('Validation Accuracy Growth (Dendrite Addition)')
    plt.xlabel('Training Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    # Save image for GitHub README
    plt.savefig("PAI_Integration_Trend_Report.png")
    plt.show()

    # Results table output (Markdown formatted)
    print("\n### QUANTITATIVE RESULTS TABLE")
    print("| Epoch | Baseline Loss | PAI Loss | Baseline Acc | PAI Acc |")
    print("|-------|---------------|----------|--------------|---------|")
    for i in range(EPOCHS):
        print(f"| {i+1:<5} | {baseline_h['loss'][i]:.4f}      | {pai_h['loss'][i]:.4f} | {baseline_h['accuracy'][i]:.2%}       | {pai_h['accuracy'][i]:.2%} |")

# ==============================================================================
# 7. MAIN EXECUTION ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    start_time = time.time()
    
    if PAI_AVAILABLE:
        # Step 1: Baseline run
        b_history = run_baseline_training()
        
        # Step 2: PerforatedAI run
        p_model, p_history = run_perforated_training()
        
        # Step 3: Reporting
        create_submission_report(b_history, p_history)
        
        total_time = (time.time() - start_time) / 60
        print("\n" + "="*80)
        print(f"✅ SUBMISSION GENERATED IN {total_time:.2f} MINUTES")
        print("ACTION: Upload 'PAI_Integration_Trend_Report.png' and update README table.")
        print("="*80)
    else:
        print("⚠ Aborting: PerforatedAI dependency missing.")
