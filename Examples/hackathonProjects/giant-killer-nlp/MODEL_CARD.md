---
language: en
license: apache-2.0
tags:
- toxicity-detection
- text-classification
- dendritic-optimization
- perforated-backpropagation
- bert-tiny
- efficient-ml
datasets:
- jigsaw-toxic-comment-classification-challenge
metrics:
- f1
- accuracy
- precision
- recall
model-index:
- name: dendritic-bert-tiny-toxicity
  results:
  - task:
      type: text-classification
      name: Toxicity Detection
    dataset:
      name: Civil Comments
      type: jigsaw_toxicity_pred
    metrics:
    - type: f1
      value: 0.358
      name: F1 Score (Toxic Class)
    - type: accuracy
      value: 0.918
      name: Accuracy
    - type: inference_time
      value: 2.25
      name: Inference Latency (ms)
---

# Dendritic BERT-Tiny for Toxicity Detection

## Model Description

This model applies **Perforated Backpropagation with Dendritic Optimization** to enhance a compact BERT-Tiny model (4.8M parameters) for toxicity classification. It achieves performance comparable to BERT-Base (109M parameters) while maintaining **17.8x faster inference speed**.

**Key Features:**
-  **22.8x smaller** than BERT-Base (4.8M vs 109M parameters)
-  **17.8x faster** inference (2.25ms vs 40ms)
-  **Superior toxic detection** (F1=0.358 vs BERT-Base F1=0.05)
-  **Dendritic optimization** using PerforatedAI
-  **Edge-ready** for real-time deployment

## Model Details

- **Model Type:** BERT-Tiny with Dendritic Optimization
- **Language:** English
- **Task:** Binary Text Classification (Toxic/Non-Toxic)
- **Base Model:** [prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny)
- **Framework:** PyTorch 2.9.1 + PerforatedAI 3.0.7
- **Parameters:** 4,798,468
- **Model Size:** 19.3 MB

## Performance Metrics

| Metric | Dendritic BERT-Tiny | BERT-Base | Improvement |
|--------|---------------------|-----------|-------------|
| **Parameters** | 4.8M | 109M | 22.8x smaller |
| **F1 Score (Toxic)** | 0.358 | 0.050 | 7.16x better |
| **Accuracy** | 91.8% | 91.0% | +0.8% |
| **Inference Time** | 2.25ms | 40.1ms | 17.8x faster |
| **Throughput** | 444 samples/s | 25 samples/s | 17.8x higher |

## Intended Use

### Primary Use Cases
- Real-time content moderation in online forums and social media
- Edge device deployment for resource-constrained environments
- High-throughput toxicity screening systems
- Research in efficient NLP and dendritic optimization

### Limitations
- Trained on Civil Comments dataset (English only)
- May not generalize to all forms of toxicity or cultural contexts
- Class imbalance handling (94% non-toxic samples)
- Best suited for binary toxicity detection

## How to Use

### Installation

```bash
pip install transformers torch perforatedai
```

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_name = "your-username/dendritic-bert-tiny-toxicity"
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Prepare input
text = "This is a sample comment to classify"
inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)

# Inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1)
    
# Get result
label = "toxic" if prediction.item() == 1 else "non-toxic"
confidence = torch.softmax(logits, dim=-1)[0][prediction].item()
print(f"Prediction: {label} (confidence: {confidence:.2%})")
```

### Batch Processing for High Throughput

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "your-username/dendritic-bert-tiny-toxicity"
tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Example batch of comments
comments = [
    "This is a great discussion!",
    "You are absolutely terrible",
    "I disagree but respect your opinion",
]

# Tokenize batch
inputs = tokenizer(comments, return_tensors="pt", max_length=128, 
                   truncation=True, padding=True)

# Batch inference
model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    probabilities = torch.softmax(outputs.logits, dim=-1)

# Display results
for comment, pred, probs in zip(comments, predictions, probabilities):
    label = "toxic" if pred.item() == 1 else "non-toxic"
    confidence = probs[pred].item()
    print(f"'{comment}' -> {label} ({confidence:.2%})")
```

### Edge Deployment with Quantization

```python
import torch
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "your-username/dendritic-bert-tiny-toxicity"
)

# Dynamic quantization for edge devices
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Model is now ~75% smaller and faster on CPU
```

## Training Details

### Training Dataset
- **Source:** Civil Comments / Jigsaw Toxicity Dataset
- **Train samples:** 5,000 (4.54% toxic)
- **Validation samples:** 1,000 (6.60% toxic)
- **Test samples:** 1,000 (9.00% toxic)

### Training Procedure
- **Optimizer:** AdamW (lr=2e-5, weight_decay=0.01)
- **Epochs:** 10
- **Batch size:** 32
- **Max sequence length:** 128 tokens
- **Loss function:** Cross-entropy with class weights (1.0, 21.0)
- **Warmup steps:** 500
- **Early stopping:** Patience 3 epochs

### Dendritic Optimization
This model uses **Perforated Backpropagation** with dendritic nodes to enhance learning capacity:
- Correlation threshold: 0.95
- Perforated AI version: 3.0.7
- Configuration: 3D tensor output dimensions for transformer layers

## Technical Architecture

```
Input Text (max 128 tokens)
    ↓
BERT-Tiny Tokenizer
    ↓
Embedding Layer (vocab_size: 30522, hidden: 128)
    ↓
Transformer Layer 1 (2 attention heads) + Dendritic Nodes
    ↓
Transformer Layer 2 (2 attention heads) + Dendritic Nodes
    ↓
Pooler Layer (CLS token)
    ↓
Classification Head (128 → 2)
    ↓
Output: [non-toxic_logit, toxic_logit]
```

## Ethical Considerations

### Bias and Fairness
- Model may inherit biases from the Civil Comments dataset
- Performance may vary across demographic groups and cultural contexts
- Should not be the sole decision-maker in content moderation

### Recommended Practices
- Use as part of a human-in-the-loop moderation system
- Regularly evaluate for fairness across user demographics
- Combine with other signals for critical decisions
- Provide appeal mechanisms for users

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{dendritic-bert-tiny-2026,
  title={Giant-Killer NLP: Dendritic Optimization for Toxicity Classification},
  author={PROJECT-Z Team},
  year={2026},
  publisher={HuggingFace Hub},
  howpublished={\url{https://huggingface.co/your-username/dendritic-bert-tiny-toxicity}},
  note={PyTorch Dendritic Optimization Hackathon}
}
```

## Acknowledgments

- **Base Model:** BERT-Tiny by [prajjwal1](https://huggingface.co/prajjwal1)
- **Optimization Framework:** [PerforatedAI](https://perforatedai.com/)
- **Dataset:** Jigsaw/Google Civil Comments
- **Hackathon:** PyTorch Dendritic Optimization Challenge

## License

Apache 2.0

## Contact

For questions, issues, or collaboration opportunities, please open an issue on the [GitHub repository](https://github.com/your-username/dendritic-bert-tiny-toxicity).

---

**Developed for the PyTorch Dendritic Optimization Hackathon - January 2026**
