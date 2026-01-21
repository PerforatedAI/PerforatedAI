import torch
import os
import numpy as np
from PIL import Image, ImageDraw
from dataset import get_dataloaders
from model import VisionCNN
from train import train_model

def create_dummy_data(root_dir, num_classes=2, images_per_class=100):
    """Creates a synthetic dataset with geometric shapes for testing."""
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    
    for class_idx in range(num_classes):
        class_dir = os.path.join(root_dir, f'class_{class_idx}')
        os.makedirs(class_dir, exist_ok=True)
        
        for img_idx in range(images_per_class):
            img_path = os.path.join(class_dir, f'img_{img_idx}.jpg')
            
            # Create a background image
            img = Image.new('RGB', (224, 224), color='white')
            draw = ImageDraw.Draw(img)
            
            # Draw shapes based on class to make it learnable
            if class_idx == 0:
                # Class 0: Blue Square
                draw.rectangle([50, 50, 174, 174], fill='blue', outline='black')
            else:
                # Class 1: Red Circle
                draw.ellipse([50, 50, 174, 174], fill='red', outline='black')
            
            img.save(img_path)
    print(f"Generated synthetic data in {root_dir}")

def main():
    # --- Configuration ---
    # UPDATE THESE PATHS to point to your actual dataset folders
    TRAIN_DIR = './data/train' 
    VAL_DIR = './data/val'
    
    NUM_CLASSES = 2

    # Hyperparameters configuration for WandB
    config = {
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 32,
        "step_size": 2,  # Scheduler: decay LR every 2 epochs
        "gamma": 0.1     # Scheduler: decay rate
    }
    
    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # --- Data Loading ---
    # Generate/Overwrite data to ensure it is learnable (not random noise)
    create_dummy_data(TRAIN_DIR, NUM_CLASSES)
    create_dummy_data(VAL_DIR, NUM_CLASSES)

    print("Loading data...")
    train_loader, val_loader = get_dataloaders(TRAIN_DIR, VAL_DIR, batch_size=config["batch_size"])

    # --- Model Initialization ---
    model = VisionCNN(num_classes=NUM_CLASSES).to(device)

    # --- Training ---
    print("Starting training...")
    train_model(model, train_loader, val_loader, device, config)

if __name__ == "__main__":
    main()