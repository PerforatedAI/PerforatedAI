import os
import torch
import random  # <--- FIXED: Added missing import
import json
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from tqdm import tqdm

# ---------------- CONFIG ----------------
# Disable WandB completely to stop login prompts
os.environ["WANDB_DISABLED"] = "true"

# Use your specific Drive path
DATA_DIR = "/content/drive/MyDrive/Colab Notebooks/PerforatedAI/Examples/hackathonProjects/Perforated_Scripts/data"

# We use the official Microsoft hub path to ensure no missing tokenizer files
MODEL_PATH = "microsoft/trocr-base-handwritten"

BATCH_SIZE = 4
EPOCHS = 5
MAX_LABEL_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------

class MedicalDataset(Dataset):
    def __init__(self, root_dir, processor):
        self.root_dir = root_dir
        self.processor = processor
        # Safety check: Ensure directory exists
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory not found: {root_dir}")
            
        self.images = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        
        if len(self.images) == 0:
            print(f"WARNING: No images found in {root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        image_path = os.path.join(self.root_dir, img_name)
        image = Image.open(image_path).convert("RGB")

        # Simulate messy handwriting (Data Augmentation)
        if random.random() < 0.25:
            image = image.filter(ImageFilter.MaxFilter(3))

        # Weak label from filename (removes extension and underscores)
        label_text = os.path.splitext(img_name)[0].replace("_", " ")

        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()

        labels = self.processor.tokenizer(
            label_text,
            padding="max_length",
            max_length=MAX_LABEL_LEN,
            truncation=True
        ).input_ids

        # Replace padding token id with -100 so we don't calculate loss on padding
        labels = [
            l if l != self.processor.tokenizer.pad_token_id else -100
            for l in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(labels)
        }

def train_pipeline():
    print(f"Using device: {DEVICE}")
    print("Loading Model and Processor...")
    
    # Load from HuggingFace Hub to guarantee working files
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH).to(DEVICE)

    # --- CRITICAL FIX: Set special tokens to prevent ValueError Crash ---
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Fix for text generation settings
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id = processor.tokenizer.pad_token_id
    # ------------------------------------------------------------------

    dataset = MedicalDataset(DATA_DIR, processor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    print(f"Starting training on {len(dataset)} images for {EPOCHS} epochs...")
    
    # List to store loss for graphing
    loss_history = []

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            pixel_values = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # --- SAVE MODEL ---
    save_path = "./baseline_model"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f"Baseline model saved to {save_path}")

    # --- SAVE METRICS FOR COMPARISON ---
    # We save this to a file so we can load it later when we run the Perforated model
    with open("baseline_metrics.json", "w") as f:
        json.dump({"loss": loss_history}, f)
    
    # --- PLOT GRAPH ---
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, EPOCHS + 1), loss_history, marker='o', label='Baseline (Original)')
    plt.title("Training Loss: Baseline")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("baseline_graph.png")
    plt.show()
    print("Graph saved as baseline_graph.png")

if __name__ == "__main__":
    train_pipeline()