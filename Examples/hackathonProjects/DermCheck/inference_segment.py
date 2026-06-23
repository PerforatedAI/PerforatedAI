"""
DermCheck Segmentation Inference
"""

import argparse
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from monai.networks.nets import UNet
import torchvision.transforms as transforms

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_segmentation(model_path, image_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading segmentation model from {model_path}...")
    
    # Initialize UNet
    model = UNet(
        spatial_dims=2,
        in_channels=3,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    )
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Running with uninitialized model for demo purposes.")
        
    model.to(device)
    model.eval()
    
    # Preprocess
    orig_image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor()
    ])
    input_tensor = transform(orig_image).unsqueeze(0).to(device)
    
    # Inference
    print("Segmenting lesion area...")
    with torch.no_grad():
        output = model(input_tensor)
        mask = torch.sigmoid(output).cpu().numpy()[0, 0]
        mask = (mask > 0.5).astype(np.uint8)
        
    # Visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(orig_image)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Lesion Segmentation")
    plt.imshow(orig_image.resize((config['data']['image_size'], config['data']['image_size'])))
    plt.imshow(mask, alpha=0.5, cmap='jet')
    plt.axis('off')
    
    out_path = "segmentation_result.png"
    plt.savefig(out_path)
    print(f"Segmentation result saved to {out_path}")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config()
    run_segmentation(args.model, args.image, config)
