"""
DermCheck Classification Inference
"""

import argparse
import os
import torch
import yaml
from PIL import Image
from monai.networks.nets import DenseNet121
import torchvision.transforms as transforms

# Clinical Classes for HAM10000
CLASSES = {
    0: "Melanocytic nevi (nv)",
    1: "Melanoma (mel)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Basal cell carcinoma (bcc)",
    4: "Actinic keratoses (akiec)",
    5: "Vascular lesions (vasc)",
    6: "Dermatofibroma (df)"
}

def load_config(path="config.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def predict_single(model, image_path, device, transform, config):
    # Preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    class_name = CLASSES.get(predicted_idx.item(), "Unknown")
    risk = "HIGH" if class_name in ["Melanoma (mel)", "Basal cell carcinoma (bcc)"] else "LOW/NORMAL"
    return class_name, confidence.item(), risk

def run_inference(model_path, data_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path}...")
    
    # Initialize MONAI model
    model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=config['data']['num_classes'])
    
    # Load weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Using uninitialized model for demonstration.")
        
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((config['data']['image_size'], config['data']['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Check if path is a directory or a file
    if os.path.isdir(data_path):
        print(f"Processing directory: {data_path}")
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        image_files = [f for f in os.listdir(data_path) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"No images found in {data_path}")
            return

        print(f"Found {len(image_files)} images. Starting batch inference...")
        
        results = []
        for img_file in image_files:
            full_path = os.path.join(data_path, img_file)
            class_name, confidence, risk = predict_single(model, full_path, device, transform, config)
            results.append((img_file, class_name, confidence, risk))
            print(f"  {img_file}: {class_name} ({confidence:.2f}) - Risk: {risk}")
            
        print(f"\nBatch processing complete. Processed {len(results)} images.")
    else:
        # Single image
        class_name, confidence, risk = predict_single(model, data_path, device, transform, config)
        print(f"\nResults for {data_path}:")
        print(f"  Condition: {class_name}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  RISK LEVEL: {risk}")
        if risk == "HIGH":
            print("  ACTION: Consult a specialist immediately.")
        else:
            print("  ACTION: Monitor regularly.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()
    
    config = load_config()
    run_inference(args.model, args.image, config)
