"""
DermCheck Integrated Demo: Classification + Segmentation
"""

import argparse
import torch
import yaml
import cv2
import numpy as np
from PIL import Image
from monai.networks.nets import DenseNet121, UNet
import torchvision.transforms as transforms

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_demo(cls_model_path, seg_model_path, source, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting Integrated Medical AI Demo...")
    
    # Load Models
    cls_model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=config['data']['num_classes'])
    seg_model = UNet(
        spatial_dims=2, in_channels=3, out_channels=1,
        channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
    )
    
    # Silently handle missing model files for demo
    try:
        cls_model.load_state_dict(torch.load(cls_model_path, map_location=device))
        seg_model.load_state_dict(torch.load(seg_model_path, map_location=device))
    except:
        print("Warning: Model weights not found. Running with uninitialized models for demo.")
        
    cls_model.to(device).eval()
    seg_model.to(device).eval()
    
    # Preprocessing
    size = config['data']['image_size']
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Image loading (simplified for single image in demo)
    img = Image.open(source).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv = cv2.resize(img_cv, (size, size))
    
    tensor = transform(img).unsqueeze(0).to(device)
    norm_tensor = norm(tensor)
    
    with torch.no_grad():
        # Classification
        cls_out = cls_model(norm_tensor)
        cls_idx = torch.argmax(cls_out, 1).item()
        
        # Segmentation
        seg_out = seg_model(tensor)
        mask = torch.sigmoid(seg_out).cpu().numpy()[0, 0]
        mask = (mask > 0.5).astype(np.uint8) * 255
    
    # Visualize overlay
    mask_colored = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_cv, 0.7, mask_colored, 0.3, 0)
    
    # Add text
    cv2.putText(overlay, f"Class: {cls_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(overlay, "DermCheck + PerforatedAI", (10, size-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    cv2.imshow("DermCheck Integrated Analysis", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Demo finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--classify-model', type=str, required=True)
    parser.add_argument('--segment-model', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config()
    run_demo(args.classify_model, args.segment_model, args.source, config)
