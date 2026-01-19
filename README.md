# Perforated Phi-2 Financial Sentiment Analysis

## Elevator Pitch
**A real-world NLP benchmark showing how dendritic optimization improves Phi-2 accuracy and efficiency on financial sentiment analysis using PerforatedAI.**

---

## 📌 About the Project

Financial sentiment analysis is a high-impact NLP task used in trading, risk management, and market intelligence.  
In this project, we apply **PerforatedAI’s dendritic optimization** to a modern large language model (**Microsoft Phi-2**) and evaluate its impact on **accuracy, F1 score, and inference latency**.

We compare:
- A **standard fine-tuned Phi-2 baseline**
- A **dendritic-optimized Phi-2 model** using PerforatedAI

The goal is to demonstrate that dendritic optimization can deliver **measurable gains** on a realistic NLP workload—not just toy benchmarks.

---

## 🔍 What It Does

- Trains a **baseline Phi-2** model on financial sentiment data  
- Trains a **dendritic Phi-2** model using PerforatedAI  
- Compares both models on:
  - Accuracy
  - F1 score
  - Inference latency
- Produces **clear visualizations** and **reproducible metrics**

---

## 🛠 How We Built It

### Dataset
- **Financial PhraseBank (All-Agree split)**
- 3-class sentiment classification:
  - Positive
  - Neutral
  - Negative

### Model
- `microsoft/phi-2`
- Parameter-efficient fine-tuning using **LoRA**
- 4-bit quantization (NF4) via **bitsandbytes**

### Dendritic Optimization
- Applied using **PerforatedAI**
- Dendrites initialized via `initialize_pai`
- Output dimensions explicitly configured for sequence classification

### Training
- Class-weighted loss to handle imbalance
- Identical data splits and hyperparameters for fair comparison
- Baseline and dendritic models trained separately

---

## 📊 Results

### Quantitative Metrics

| Model       | Accuracy | F1 Score | Inference Time (s) |
|------------|----------|----------|-------------------|
| Baseline   | ~0.73    | ~0.72    | ~0.12             |
| Dendritic  | **~0.77** | **~0.76** | ~0.12             |

### Visualizations
- **Accuracy comparison**
- **F1 score comparison**
- **Training dynamics**
- **Inference latency comparison**

See:
- `Accuracy_Improvement.png`
- `Dendrite_vs_Baseline_Training.png`

---

## 🚧 Challenges We Ran Into

- Managing GPU memory while combining:
  - Quantization
  - LoRA
  - Dendritic layers
- Avoiding mixed-precision instability during training
- Ensuring reproducibility across Colab and local GPUs
- Preventing PerforatedAI interactive prompts during automated training

---

## 🏆 Accomplishments We’re Proud Of

- Successfully applied dendrites to a **real-world NLP task**
- Demonstrated **consistent accuracy and F1 improvements**
- Built a **clean, reproducible benchmark**
- Matched the official PerforatedAI hackathon submission format

---

## 📚 What We Learned

- Dendritic optimization is not just theoretical—it delivers **practical gains**
- Careful configuration is essential when combining:
  - Quantization
  - PEFT
  - Dendritic layers
- Real benchmarks matter more than synthetic tasks

---

## 🚀 What’s Next

- Extend dendritic optimization to:
  - Larger financial datasets
  - Multi-task financial NLP
- Explore **latency-optimized dendrites** for real-time inference
- Apply PerforatedAI to:
  - Document classification
  - Risk detection
  - Regulatory NLP pipelines

---

## 🧰 Built With

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **PerforatedAI**
- **BitsAndBytes (4-bit quantization)**
- **PEFT (LoRA)**
- **scikit-learn**
- **Matplotlib & Seaborn**

---

## ▶️ How to Run

```bash
pip install transformers accelerate bitsandbytes datasets peft scikit-learn
git clone https://github.com/PerforatedAI/PerforatedAI.git
pip install -e ./PerforatedAI

python train.py
