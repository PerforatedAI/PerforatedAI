"""
Guardian-Pulse Ghana: Core AI Models
====================================
Two-model architecture for comprehensive cellular threat detection:

Model A (Signal-Geo): Detects IMSI catchers via signal triangulation
Model B (Cyber-Sec): Analyzes network logs for intrusion patterns

Both designed for edge deployment on resource-constrained devices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SignalGeoModel(nn.Module):
    """
    Model A: IMSI Catcher Detection via Signal Triangulation
    
    Mission: Identify rogue cell towers in Ghana's telecom infrastructure
    
    Architecture:
    - Input: RF spectrogram images (128x128x3) from SDR receivers
    - Output: Binary classification (legitimate/rogue tower) + confidence score
    
    Key Features:
    - Spatial feature extraction for signal pattern analysis
    - Handles interference and multipath fading common in urban Ghana
    - Optimized for Mali-G series GPUs found in African smartphones
    """
    
    def __init__(self, num_classes: int = 2):
        super(SignalGeoModel, self).__init__()
        
        # === Convolutional Feature Extractor ===
        # Stage 1: Initial signal feature extraction
        # Captures basic RF signatures (frequency peaks, power distribution)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, 
                               kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stage 2: Pattern recognition layer
        # Identifies anomalous signal behaviors (sudden power spikes, LAC changes)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, 
                               kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stage 3: Deep feature extraction
        # Distinguishes between legitimate base stations and IMSI catchers
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, 
                               kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Stage 4: High-level abstraction
        # Combines multiple signal characteristics for final decision
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, 
                               kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # === Classification Head ===
        # Adaptive pooling ensures consistent output regardless of input variations
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers for final threat assessment
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)  # Regularization for limited training data
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: RF Spectrogram → Threat Classification
        
        Args:
            x: Batch of RF spectrograms [B, 3, 128, 128]
        
        Returns:
            Logits for binary classification [B, 2]
        """
        # === Feature Extraction Pipeline ===
        # Stage 1: Scanning for basic signal anomalies
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Stage 2: Analyzing signal power distribution patterns
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Stage 3: Identifying IMSI catcher signatures
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Stage 4: Final feature refinement
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # === Classification ===
        # Global pooling and flattening
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        
        # Decision network with dropout for robustness
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # Raw logits for cross-entropy loss
        
        return x


class CyberSecModel(nn.Module):
    """
    Model B: Network Log Analysis for Intrusion Detection
    
    Mission: Detect cybersecurity threats in Ghana's telecom networks
    
    Architecture:
    - Input: Sequential network log embeddings (max 512 tokens)
    - Output: Multi-class threat classification (normal/malware/dos/injection)
    
    Key Features:
    - Lightweight transformer architecture for sequence analysis
    - Detects SS7 exploits, SIM swap attacks, and SMS phishing
    - Designed for continuous monitoring on edge gateways
    """
    
    def __init__(self, input_dim: int = 768, num_classes: int = 4, 
                 hidden_dim: int = 256, num_heads: int = 4, num_layers: int = 2):
        super(CyberSecModel, self).__init__()
        
        # === Input Processing ===
        # Projects raw embeddings to model dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding for sequence order awareness
        # Critical for detecting time-based attack patterns
        self.positional_encoding = nn.Parameter(
            torch.randn(1, 512, hidden_dim) * 0.02
        )
        
        # === Transformer Encoder ===
        # Multi-head attention for capturing attack pattern correlations
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # === Classification Head ===
        # Multi-layer perceptron for threat categorization
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Network Logs → Threat Classification
        
        Args:
            x: Batch of log embeddings [B, seq_len, 768]
        
        Returns:
            Logits for multi-class classification [B, 4]
        """
        # === Embedding and Positional Encoding ===
        # Project to model dimension
        x = self.input_projection(x)  # [B, seq_len, hidden_dim]
        
        # Add positional information for temporal attack patterns
        seq_len = x.size(1)
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # === Transformer Processing ===
        # Analyzing log sequences for anomalous patterns:
        # - Repeated failed authentication (brute force)
        # - Unusual protocol sequences (SS7 exploits)
        # - Rapid location updates (SIM swap indicators)
        x = self.transformer(x)  # [B, seq_len, hidden_dim]
        
        # === Global Aggregation ===
        # Use mean pooling to create sequence-level representation
        # Captures overall threat context across entire log window
        x = torch.mean(x, dim=1)  # [B, hidden_dim]
        
        # === Classification ===
        # Final threat categorization
        x = self.classifier(x)  # [B, num_classes]
        
        return x


class HybridGuardianModel(nn.Module):
    """
    Hybrid Guardian System (Optional Ensemble)
    
    Combines both models for comprehensive threat detection:
    - Signal-Geo detects physical infrastructure threats
    - Cyber-Sec identifies network-level attacks
    
    Fusion strategy enables correlation of physical and digital threats
    (e.g., rogue tower + credential harvesting = coordinated attack)
    """
    
    def __init__(self, signal_model: SignalGeoModel, cyber_model: CyberSecModel):
        super(HybridGuardianModel, self).__init__()
        self.signal_model = signal_model
        self.cyber_model = cyber_model
        
        # Fusion layer combines both threat assessments
        self.fusion = nn.Linear(6, 3)  # 2 signal + 4 cyber → 3 final classes
        
    def forward(self, signal_input: torch.Tensor, 
                cyber_input: torch.Tensor) -> torch.Tensor:
        """
        Dual-threat analysis pipeline
        
        Returns:
            Combined threat classification [B, 3]:
            0: Safe, 1: Signal Threat, 2: Cyber Threat
        """
        signal_logits = self.signal_model(signal_input)
        cyber_logits = self.cyber_model(cyber_input)
        
        # Concatenate threat scores
        combined = torch.cat([signal_logits, cyber_logits], dim=1)
        
        # Final fusion decision
        return self.fusion(combined)


# === Model Factory ===
def create_signal_geo_model(num_classes: int = 2) -> SignalGeoModel:
    """Initialize Model A for IMSI catcher detection"""
    return SignalGeoModel(num_classes=num_classes)


def create_cyber_sec_model(input_dim: int = 768, num_classes: int = 4) -> CyberSecModel:
    """Initialize Model B for network intrusion detection"""
    return CyberSecModel(input_dim=input_dim, num_classes=num_classes)