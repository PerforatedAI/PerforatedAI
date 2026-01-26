"""
Export models to TorchScript for mobile/edge deployment
🚀 Enable deployment on mobile and edge devices
"""
import sys
sys.path.append('..')

import torch
from models.baseline_model import BaselineModel
from models.dendritic_model import DendriticModel


def export_to_torchscript(model, model_name, input_shape=(1, 5)):
    """Export model to TorchScript format"""
    model.eval()
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Save traced model
    output_path = f'../models/{model_name}_torchscript.pt'
    traced_model.save(output_path)
    
    print(f"✅ {model_name} exported to {output_path}")
    
    # Verify the exported model
    loaded_model = torch.jit.load(output_path)
    test_input = torch.randn(*input_shape)
    
    with torch.no_grad():
        original_output = model(test_input)
        loaded_output = loaded_model(test_input)
        
        # Check if outputs match
        if torch.allclose(original_output, loaded_output, rtol=1e-5):
            print(f"✅ {model_name} verification passed")
        else:
            print(f"⚠️  {model_name} verification failed - outputs don't match")
    
    return output_path


def main():
    """Main export function"""
    
    print("="*60)
    print("Exporting Models to TorchScript")
    print("="*60)
    
    # Export baseline model
    print("\nExporting Baseline Model...")
    baseline_model = BaselineModel()
    baseline_path = export_to_torchscript(baseline_model, 'baseline')
    
    # Export dendritic model
    print("\nExporting Dendritic Model...")
    dendritic_model = DendriticModel()
    dendritic_path = export_to_torchscript(dendritic_model, 'dendritic')
    
    print("\n" + "="*60)
    print("Export Summary:")
    print("="*60)
    print(f"Baseline Model: {baseline_path}")
    print(f"Dendritic Model: {dendritic_path}")
    print("\n📱 Models are ready for mobile/edge deployment!")


if __name__ == "__main__":
    main()
