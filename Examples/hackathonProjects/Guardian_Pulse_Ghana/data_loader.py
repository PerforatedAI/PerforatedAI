"""
Guardian-Pulse Ghana: Synthetic Data Generator
==============================================
Mock data loaders for testing and development without real datasets.

In production, this would connect to:
- SDR receivers capturing cellular RF signals (Model A)
- Network taps collecting SS7/Diameter logs (Model B)
- Ghana Telecom Authority's threat intelligence feeds

For hackathon purposes, generates realistic synthetic data matching
production input specifications.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class SignalGeoDataset(Dataset):
    """
    Synthetic RF Spectrogram Dataset for IMSI Catcher Detection
    
    Simulates real-world data characteristics:
    - RF spectrograms from 900 MHz and 1800 MHz GSM bands
    - Spatial resolution: 128x128 pixels (time x frequency)
    - 3 channels: Power, Phase, Doppler shift
    
    In production, spectrograms generated from:
    - HackRF One SDR receivers
    - GNU Radio signal processing pipeline
    - Real-time FFT with 10ms windows
    """
    
    def __init__(self, num_samples: int = 1000, image_size: Tuple[int, int] = (128, 128)):
        """
        Args:
            num_samples: Number of synthetic signal samples
            image_size: Spectrogram dimensions (height, width)
        """
        self.num_samples = num_samples
        self.image_size = image_size
        
        # Generate labels (balanced dataset)
        # 0: Legitimate cell tower
        # 1: IMSI catcher / rogue tower
        self.labels = torch.randint(0, 2, (num_samples,))
        
        # Pre-generate random seeds for reproducibility
        self.seeds = np.random.randint(0, 1000000, num_samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate single synthetic RF spectrogram
        
        Returns:
            image: [3, 128, 128] tensor (Power, Phase, Doppler channels)
            label: Binary classification (0=legitimate, 1=rogue)
        """
        # Set seed for reproducibility
        np.random.seed(self.seeds[idx])
        torch.manual_seed(self.seeds[idx])
        
        # === Simulate RF Spectrogram ===
        # Channel 1: Power Spectral Density
        # Legitimate towers: Stable power (-80 to -60 dBm)
        # IMSI catchers: Higher power to override legitimate signals (-50 to -30 dBm)
        power_base = -70 if self.labels[idx] == 0 else -40
        power_channel = torch.randn(1, *self.image_size) * 10 + power_base
        
        # Channel 2: Phase Information
        # IMSI catchers often have phase instability due to cheaper oscillators
        phase_noise = 0.1 if self.labels[idx] == 0 else 0.5
        phase_channel = torch.randn(1, *self.image_size) * phase_noise
        
        # Channel 3: Doppler Shift
        # Rogue towers may have frequency drift
        doppler_drift = 0.05 if self.labels[idx] == 0 else 0.3
        doppler_channel = torch.randn(1, *self.image_size) * doppler_drift
        
        # Combine channels
        image = torch.cat([power_channel, phase_channel, doppler_channel], dim=0)
        
        # Normalize to [0, 1] range for neural network input
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image, self.labels[idx]


class CyberSecDataset(Dataset):
    """
    Synthetic Network Log Dataset for Intrusion Detection
    
    Simulates embeddings from pre-trained language models:
    - Input: Tokenized network logs (SS7 messages, authentication attempts)
    - Embedding: 768-dimensional vectors (BERT/RoBERTa standard)
    - Sequence length: Variable (padded to 512 max)
    
    In production, logs sourced from:
    - Telecom carrier SIEM systems
    - Mobile Network Operator (MNO) authentication servers
    - Ghana Cyber Security Authority incident reports
    """
    
    def __init__(self, num_samples: int = 1000, seq_len: int = 128, 
                 embedding_dim: int = 768, num_classes: int = 4):
        """
        Args:
            num_samples: Number of synthetic log sequences
            seq_len: Sequence length (number of log entries)
            embedding_dim: Embedding dimensionality
            num_classes: Threat categories (Normal, Malware, DoS, Injection)
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        
        # Generate labels (balanced multi-class dataset)
        # 0: Normal traffic
        # 1: Malware/RAT activity
        # 2: Denial of Service attack
        # 3: SQL/Command injection attempt
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Pre-generate seeds
        self.seeds = np.random.randint(0, 1000000, num_samples)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate single synthetic log sequence embedding
        
        Returns:
            sequence: [seq_len, embedding_dim] tensor
            label: Multi-class classification (0-3)
        """
        # Set seed for reproducibility
        np.random.seed(self.seeds[idx])
        torch.manual_seed(self.seeds[idx])
        
        # === Simulate Log Embeddings ===
        # Different attack types have distinct embedding patterns
        
        label = self.labels[idx].item()
        
        if label == 0:  # Normal traffic
            # Low variance, centered around zero
            sequence = torch.randn(self.seq_len, self.embedding_dim) * 0.5
        
        elif label == 1:  # Malware/RAT
            # High variance in specific dimensions (C2 communication patterns)
            sequence = torch.randn(self.seq_len, self.embedding_dim) * 0.5
            sequence[:, :100] += torch.randn(self.seq_len, 100) * 2.0  # Anomalous features
        
        elif label == 2:  # DoS attack
            # Repetitive patterns (flood of similar requests)
            base_pattern = torch.randn(1, self.embedding_dim) * 0.5
            sequence = base_pattern.repeat(self.seq_len, 1)
            sequence += torch.randn(self.seq_len, self.embedding_dim) * 0.2  # Small noise
        
        elif label == 3:  # Injection attack
            # Outlier embeddings (malicious payloads in logs)
            sequence = torch.randn(self.seq_len, self.embedding_dim) * 0.5
            # Inject anomalous entries
            num_injections = np.random.randint(5, 20)
            injection_indices = np.random.choice(self.seq_len, num_injections, replace=False)
            sequence[injection_indices] += torch.randn(num_injections, self.embedding_dim) * 5.0
        
        # Normalize embeddings
        sequence = torch.nn.functional.normalize(sequence, p=2, dim=1)
        
        return sequence, self.labels[idx]


def get_signal_geo_loader(batch_size: int = 32, num_samples: int = 1000, 
                          num_workers: int = 2, shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader for Signal-Geo Model (IMSI Catcher Detection)
    
    Args:
        batch_size: Samples per batch (adjust based on GPU memory)
        num_samples: Total dataset size
        num_workers: Parallel data loading workers
        shuffle: Randomize batch order
    
    Returns:
        DataLoader yielding (images, labels) batches
    """
    dataset = SignalGeoDataset(num_samples=num_samples)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )
    
    print(f"✓ Signal-Geo DataLoader initialized")
    print(f"  - Samples: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input shape: [B, 3, 128, 128]")
    print(f"  - Classes: 2 (Legitimate / Rogue)\n")
    
    return loader


def get_cyber_sec_loader(batch_size: int = 32, num_samples: int = 1000,
                         seq_len: int = 128, num_workers: int = 2, 
                         shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader for Cyber-Sec Model (Intrusion Detection)
    
    Args:
        batch_size: Samples per batch
        num_samples: Total dataset size
        seq_len: Log sequence length
        num_workers: Parallel data loading workers
        shuffle: Randomize batch order
    
    Returns:
        DataLoader yielding (sequences, labels) batches
    """
    dataset = CyberSecDataset(num_samples=num_samples, seq_len=seq_len)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"✓ Cyber-Sec DataLoader initialized")
    print(f"  - Samples: {num_samples}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input shape: [B, {seq_len}, 768]")
    print(f"  - Classes: 4 (Normal / Malware / DoS / Injection)\n")
    
    return loader


def get_sample_batch(model_type: str = 'signal') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate single sample batch for testing
    
    Useful for:
    - Model architecture validation
    - Forward pass debugging
    - Output shape verification
    
    Args:
        model_type: 'signal' or 'cyber'
    
    Returns:
        (inputs, labels) tuple
    """
    if model_type == 'signal':
        # Single RF spectrogram batch
        images = torch.randn(8, 3, 128, 128)  # [B, C, H, W]
        labels = torch.randint(0, 2, (8,))
        return images, labels
    
    elif model_type == 'cyber':
        # Single log sequence batch
        sequences = torch.randn(8, 128, 768)  # [B, seq_len, embed_dim]
        labels = torch.randint(0, 4, (8,))
        return sequences, labels
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# === Production Data Integration Notes ===
"""
For deployment with real data:

1. Signal-Geo Model:
   - Replace SignalGeoDataset with RF capture pipeline
   - Use GNU Radio for real-time spectrogram generation
   - Integrate with HackRF/RTL-SDR hardware
   - Apply signal preprocessing (noise reduction, normalization)

2. Cyber-Sec Model:
   - Replace CyberSecDataset with log ingestion pipeline
   - Use sentence-transformers for log embedding
   - Connect to Kafka/Fluentd for real-time streaming
   - Implement sliding window for online detection

3. Data Augmentation (for production):
   - Frequency domain shifts (Doppler effects)
   - Time domain stretching (variable call durations)
   - Additive noise (urban interference simulation)
   - Mixup for regularization

4. Ghana-Specific Considerations:
   - Handle 2G (GSM 900/1800) and 3G (UMTS 2100) bands
   - Account for MTN/Vodafone/AirtelTigo tower patterns
   - Incorporate geographic features (rural vs urban)
   - Adapt to local attack signatures (SIM box fraud, etc.)
"""