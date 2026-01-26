import torch
import torch.nn as nn
from torchvision import models
# Removed direct import of DendriticLinear as it is not in the installed package.
# PAI will convert nn.Linear to dendritic logic via initialize_pai.

class DendriticVisionModel(nn.Module):
    def __init__(self, num_classes=10, dendrite_count=4):
        """
        Args:
            num_classes (int): Number of output categories.
            dendrite_count (int): The number of dendritic branches per neuron.
                                 (Used in training script to configure the PAI tracker)
        """
        super(DendriticVisionModel, self).__init__()
        
        # 1. Load the pre-trained MobileNetV3-Small backbone
        self.base_model = models.mobilenet_v3_small(weights="DEFAULT")
        
        # 2. Extract features and pooling
        self.features = self.base_model.features
        self.avgpool = self.base_model.avgpool
        
        # 3. Modify the classifier head
        in_features = self.base_model.classifier[3].in_features
        
        self.classifier_top = nn.Sequential(
            self.base_model.classifier[0],
            self.base_model.classifier[1],
            self.base_model.classifier[2]
        )
        
        # The Dendritic Layer replacement
        # We use a standard Linear layer here. 
        # UPA.initialize_pai(model) in train.py will convert this to a dendritic module.
        self.dendritic_output = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier_top(x)
        x = self.dendritic_output(x)
        return x

if __name__ == "__main__":
    test_input = torch.randn(1, 3, 224, 224)
    model = DendriticVisionModel(num_classes=10, dendrite_count=8)
    output = model(test_input)
    print(f"Output shape: {output.shape}") 
