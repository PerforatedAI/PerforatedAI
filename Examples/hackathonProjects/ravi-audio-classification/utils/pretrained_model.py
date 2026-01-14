"""
Pretrained model wrapper for ESC-50 classification.
Uses SpeechBrain's CNN14 model pretrained on ESC-50.
"""
import torch
import torch.nn as nn

# Lazy import for SpeechBrain to avoid compatibility issues
# from speechbrain.inference.classifiers import EncoderClassifier


class SpeechBrainESC50(nn.Module):
    """
    Wrapper for SpeechBrain's pretrained CNN14 model for ESC-50.
    
    This model achieves ~82-90% accuracy on ESC-50 test set.
    """
    
    def __init__(self, freeze_encoder=False):
        super(SpeechBrainESC50, self).__init__()
        
        # Import here to avoid issues when not using this model
        from speechbrain.inference.classifiers import EncoderClassifier
        
        # Load pretrained model from SpeechBrain
        # Using the ESC-50 fine-tuned checkpoint
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/cnn14-esc50",
            savedir="pretrained_models/cnn14-esc50"
        )
        
        # Optionally freeze encoder layers for faster fine-tuning
        if freeze_encoder:
            for param in self.classifier.mods.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, 1, height, width) - mel-spectrograms
               or (batch, time) - raw audio
        
        Returns:
            Logits of shape (batch, 50)
        """
        # SpeechBrain expects audio waveforms, but we can also pass spectrograms
        # For mel-spectrograms, we need to adapt the input
        
        # If input is spectrogram (batch, 1, mel_bins, time)
        if len(x.shape) == 4:
            # Remove channel dimension and transpose to (batch, time, mel_bins)
            x = x.squeeze(1).transpose(1, 2)
        
        # Get predictions
        # The classifier returns embeddings and predictions
        out = self.classifier.encode_batch(x)
        logits = self.classifier.mods.classifier(out)
        
        return logits
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_embeddings(self, x):
        """Get embeddings from the encoder (useful for visualization)"""
        if len(x.shape) == 4:
            x = x.squeeze(1).transpose(1, 2)
        return self.classifier.encode_batch(x)


class CNN14ESC50(nn.Module):
    """
    CNN14 implementation for ESC-50, optimized for PerforatedAI.
    Based on the PANNs (Pretrained Audio Neural Networks) architecture.
    
    Uses Sequential blocks (Conv+BN+ReLU+Pool) which PAI can convert more effectively.
    """
    
    def __init__(self, num_classes=50):
        super(CNN14ESC50, self).__init__()
        
        # Convolutional blocks using Sequential for PAI compatibility
        # PAI works best when Conv+BN are grouped together
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2, 2)
        )
        
        # Global average pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input: (batch, 1, 128, 216)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        
        # Global pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
