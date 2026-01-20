"""
BERT-Tiny Model with Dendritic Optimization

This module defines the BERT-Tiny model architecture and provides utilities
for wrapping it with PerforatedAI dendritic nodes. The goal is to achieve
BERT-Base level performance with BERT-Tiny's speed.

Architecture:
- BERT-Tiny: 2 layers, 128 hidden size, ~4M parameters
- BERT-Base: 12 layers, 768 hidden size, ~110M parameters
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoConfig
from typing import Optional, Dict, Any

# PerforatedAI imports - these enable dendritic optimization
try:
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    PAI_AVAILABLE = True
except ImportError:
    PAI_AVAILABLE = False
    print("Warning: perforatedai not installed. Dendritic optimization disabled.")


class ToxicityClassifier(nn.Module):
    """
    Toxicity Classification model based on BERT-Tiny.
    
    This wrapper allows for custom classification heads and is designed
    to work seamlessly with PerforatedAI dendritic optimization.
    """
    
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        num_labels: int = 2,
        hidden_dropout_prob: float = 0.1,
    ):
        """
        Initialize the ToxicityClassifier.
        
        Args:
            model_name: HuggingFace model identifier
            num_labels: Number of classification labels
            hidden_dropout_prob: Dropout probability for classifier
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Load the base BERT model
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        
        # Custom classification head
        # This is where dendrites will add the most value
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, num_labels)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the classifier weights."""
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Optional labels for computing loss
            class_weights: Optional class weights for imbalanced dataset
            
        Returns:
            Dictionary containing:
                - logits: Classification logits
                - loss: Cross-entropy loss (if labels provided)
                - hidden_states: Optional hidden states
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout and classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if class_weights is not None:
                loss_fn = nn.CrossEntropyLoss(weight=class_weights)
            else:
                loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": outputs.last_hidden_state,
        }
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_bert_tiny_model(
    num_labels: int = 2,
    hidden_dropout_prob: float = 0.1,
) -> ToxicityClassifier:
    """
    Create a BERT-Tiny model for toxicity classification.
    
    BERT-Tiny specifications:
    - 2 transformer layers
    - 128 hidden size
    - 2 attention heads
    - ~4M parameters
    
    Args:
        num_labels: Number of classification labels
        hidden_dropout_prob: Dropout probability
        
    Returns:
        ToxicityClassifier model instance
    """
    model = ToxicityClassifier(
        model_name="prajjwal1/bert-tiny",
        num_labels=num_labels,
        hidden_dropout_prob=hidden_dropout_prob,
    )
    
    print(f"Created BERT-Tiny model with {model.get_num_parameters():,} parameters")
    return model


def create_bert_base_model(
    num_labels: int = 2,
) -> AutoModelForSequenceClassification:
    """
    Create a BERT-Base model for comparison benchmarking.
    
    BERT-Base specifications:
    - 12 transformer layers
    - 768 hidden size
    - 12 attention heads
    - ~110M parameters
    
    This is the "Giant" that our "Killer" aims to match.
    
    Args:
        num_labels: Number of classification labels
        
    Returns:
        BERT-Base model for sequence classification
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels,
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created BERT-Base model with {num_params:,} parameters")
    return model


def wrap_with_dendrites(model: nn.Module) -> nn.Module:
    """
    Wrap a PyTorch model with PerforatedAI dendritic nodes.
    
    This is the core of the "Giant-Killer" approach. The dendrites learn
    to correct the mistakes of the base model by:
    1. Detecting error patterns in the base model's predictions
    2. Adding corrective signals that improve accuracy
    3. Using Cascade Correlation to maximize correlation with residual errors
    
    Mathematical Objective:
    max θ_d Corr(D_θd(x), E)
    
    Where D is dendrite output and E is the residual error.
    
    Args:
        model: PyTorch model to wrap
        
    Returns:
        Model wrapped with dendritic optimization
    """
    if not PAI_AVAILABLE:
        print("Warning: PerforatedAI not available. Returning unwrapped model.")
        return model
    
    # Configure PAI to skip interactive prompts
    GPA.pc.set_unwrapped_modules_confirmed(True)
    
    # Add module types that should be tracked but not wrapped
    # LayerNorm and embedding layers don't need dendritic optimization
    module_names_to_track = GPA.pc.get_module_names_to_track()
    if "LayerNorm" not in module_names_to_track:
        module_names_to_track.append("LayerNorm")
    if "Embedding" not in module_names_to_track:
        module_names_to_track.append("Embedding")
    GPA.pc.set_module_names_to_track(module_names_to_track)
    
    # Initialize PAI on the model
    wrapped_model = UPA.initialize_pai(model)
    
    # Initialize PAI tracker for perforated backpropagation
    # This must be done AFTER initialize_pai but BEFORE dimension configuration
    try:
        if hasattr(GPA, 'pai_tracker') and GPA.pai_tracker is not None:
            print("PAI tracker detected and available for use")
        else:
            print("Warning: PAI tracker not available in this PerforatedAI version")
    except Exception as e:
        print(f"Note: PAI tracker check failed: {e}")
    
    # Configure output dimensions for BERT layers
    # BERT outputs are 3D: [batch_size, seq_length, hidden_size]
    # PAI dimension markers:
    #   -1: batch dimension (variable, not tracked)
    #    0: first tracked dimension (seq_length in BERT)
    #   >0: fixed dimension size
    try:
        print("Configuring dimensions for BERT encoder layers...")
        GPA.pc.set_debugging_output_dimensions(1)
        
        # BERT-Tiny: 2 layers, hidden_size=128, intermediate_size=512
        for layer_idx in range(2):
            layer = wrapped_model.bert.encoder.layer[layer_idx]
            
            # Self-attention Q, K, V projections: [batch, seq_len, hidden] = [32, 128, 128]
            # Format: [-1, 0, 128] = [batch(variable), seq_len(tracked), hidden(128)]
            if hasattr(layer.attention.self.query, 'set_this_output_dimensions'):
                layer.attention.self.query.set_this_output_dimensions([-1, 0, 128])
                print(f"  [OK] Layer {layer_idx} attention.self.query: [-1, 0, 128]")
            
            if hasattr(layer.attention.self.key, 'set_this_output_dimensions'):
                layer.attention.self.key.set_this_output_dimensions([-1, 0, 128])
                print(f"  [OK] Layer {layer_idx} attention.self.key: [-1, 0, 128]")
            
            if hasattr(layer.attention.self.value, 'set_this_output_dimensions'):
                layer.attention.self.value.set_this_output_dimensions([-1, 0, 128])
                print(f"  [OK] Layer {layer_idx} attention.self.value: [-1, 0, 128]")
            
            # Attention output dense: [batch, seq_len, hidden] = [32, 128, 128]
            if hasattr(layer.attention.output.dense, 'set_this_output_dimensions'):
                layer.attention.output.dense.set_this_output_dimensions([-1, 0, 128])
                print(f"  [OK] Layer {layer_idx} attention.output.dense: [-1, 0, 128]")
            
            # Intermediate (FFN first layer): [batch, seq_len, intermediate] = [32, 128, 512]
            if hasattr(layer.intermediate.dense, 'set_this_output_dimensions'):
                layer.intermediate.dense.set_this_output_dimensions([-1, 0, 512])
                print(f"  [OK] Layer {layer_idx} intermediate.dense: [-1, 0, 512]")
            
            # Output (FFN second layer): [batch, seq_len, hidden] = [32, 128, 128]
            if hasattr(layer.output.dense, 'set_this_output_dimensions'):
                layer.output.dense.set_this_output_dimensions([-1, 0, 128])
                print(f"  [OK] Layer {layer_idx} output.dense: [-1, 0, 128]")
        
        print("[OK] Dimension configuration complete")
    except Exception as e:
        print(f"[WARNING] Could not configure output dimensions: {e}")
        print("  Continuing with auto-detection (may cause training issues)")
    
    print("Model wrapped with dendritic optimization")
    print(f"Parameters after wrapping: {sum(p.numel() for p in wrapped_model.parameters()):,}")
    
    return wrapped_model


def setup_perforated_optimizer(
    model: nn.Module,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    scheduler_step_size: int = 1,
    scheduler_gamma: float = 0.1,
    use_pai: bool = True,
):
    """
    Setup the optimizer and scheduler for perforated training.
    
    IMPORTANT: The PAI tracker manages the optimizer and scheduler to handle
    the switching between neuron learning and dendrite learning phases.
    
    Do NOT call scheduler.step() manually in your training loop!
    
    Args:
        model: The model (should already be wrapped with dendrites)
        learning_rate: Learning rate for AdamW
        weight_decay: Weight decay for regularization
        scheduler_step_size: Steps before learning rate decay
        scheduler_gamma: Learning rate decay factor
        use_pai: Whether to use PAI tracker (set False for baseline)
        
    Returns:
        Tuple of (optimizer, scheduler) managed by PAI tracker
    """
    if not PAI_AVAILABLE or not use_pai:
        # Fallback to standard optimizer if PAI not available or not requested
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_step_size,
            gamma=scheduler_gamma,
        )
        print("Standard optimizer and scheduler initialized")
        return optimizer, scheduler
    
    # Check if PAI tracker is properly initialized
    try:
        if hasattr(GPA, 'pai_tracker') and GPA.pai_tracker is not None:
            # Verify tracker has required methods
            if not hasattr(GPA.pai_tracker, 'setOptimizer'):
                raise AttributeError("PAI tracker missing setOptimizer method")
            
            print("Setting up PAI tracker with optimizer and scheduler...")
            
            # Set optimizer and scheduler classes
            GPA.pai_tracker.setOptimizer(torch.optim.AdamW)
            GPA.pai_tracker.setScheduler(torch.optim.lr_scheduler.StepLR)
            
            # Define optimizer arguments
            optim_args = {
                "lr": learning_rate,
                "weight_decay": weight_decay,
            }
            
            # Define scheduler arguments
            sched_args = {
                "step_size": scheduler_step_size,
                "gamma": scheduler_gamma,
            }
            
            # Let PAI tracker create the optimizer and scheduler
            optimizer = GPA.pai_tracker.getOptimizer(model.parameters(), **optim_args)
            scheduler = GPA.pai_tracker.getScheduler(optimizer, **sched_args)
            
            print("Perforated optimizer and scheduler initialized via PAI tracker")
            print("Two-phase training enabled: Neuron learning -> Dendrite learning")
            print("IMPORTANT: Do NOT call scheduler.step() manually - PAI tracker manages this")
            
            return optimizer, scheduler
            
        else:
            raise AttributeError("PAI tracker not available")
            
    except Exception as e:
        print(f"Warning: PAI tracker initialization failed: {e}")
        print("Falling back to standard optimizer (dendrites still active but no phase switching)")
    
    # Fallback to standard optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )
    print("Standard optimizer initialized with dendritic model")
    
    return optimizer, scheduler


def quantize_model(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Quantize the model for edge deployment.
    
    Dynamic quantization reduces model size while maintaining accuracy.
    Target: ~2-5MB model size for real-time toxicity filtering.
    
    Args:
        model: The trained model
        dtype: Quantization data type
        
    Returns:
        Quantized model
    """
    model.eval()
    
    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=dtype,
    )
    
    # Calculate size reduction
    original_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    print(f"Model quantized successfully")
    print(f"Original size: {original_size / 1e6:.2f} MB")
    
    return quantized_model


if __name__ == "__main__":
    # Test model creation
    print("Testing model creation...")
    
    # Create BERT-Tiny
    tiny_model = create_bert_tiny_model()
    
    # Test forward pass
    dummy_input = torch.randint(0, 1000, (2, 128))
    dummy_mask = torch.ones(2, 128)
    
    output = tiny_model(dummy_input, dummy_mask)
    print(f"Output logits shape: {output['logits'].shape}")
    
    # Test dendrite wrapping if available
    if PAI_AVAILABLE:
        wrapped_model = wrap_with_dendrites(tiny_model)
        output = wrapped_model(dummy_input, dummy_mask)
        print(f"Wrapped output logits shape: {output['logits'].shape}")
