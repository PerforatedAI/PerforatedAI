# 🇬🇭 Guardian-Pulse Ghana
**AI-Powered Defense System for Critical Telecommunications Infrastructure**

[![Status](https://img.shields.io/badge/Status-Prototype_Ready-success)]()
[![Optimization](https://img.shields.io/badge/PerforatedAI-Dendritic_Optimization-purple)]()
[![Framework](https://img.shields.io/badge/PyTorch-2.0-red)]()
[![Logging](https://img.shields.io/badge/Weights_&_Biases-Active-gold)]()

---

## 🚨 The Mission
**Guardian-Pulse** is a specialized Edge-AI system designed for deployment on resource-constrained devices (e.g., Raspberry Pi, Jetson Nano) within Ghana's telecommunications network.

**Objective:**
To secure regional infrastructure by detecting **Illicit Cell Towers (IMSI Catchers)** and **Cybersecurity Vulnerabilities** in real-time, without relying on expensive, centralized GPU clusters.

**The Challenge:**
Standard deep learning models are too heavy for edge deployment in remote regions of Ghana. To solve this, we utilize **Perforated AI’s Dendritic Optimization** to achieve a 30-50% reduction in model size while maintaining detection accuracy.

---

## 🧠 The Core Models

### 📡 Model A: Signal Geo-Spatial (Deployed)
* **Architecture:** CNN-based Signal Analyzer.
* **Function:** Analyzes RF signal strength and tower metadata to triangulate coordinates.
* **Target:** Flags "rogue" cell towers (IMSI catchers) used by criminals for interception.
* **Optimization:** Standard `nn.Conv2d` layers are dynamically swapped for **Dendritic Convolutional Layers**.

### 🛡️ Model B: Cyber-Sec NLP (In Development)
* **Architecture:** Transformer / Greenwash-XAI Engine (Adapted).
* **Function:** Scans network logs and code repositories.
* **Target:** Identifies "Cyber-deception" patterns and infrastructure backdoors.
* **Optimization:** Large `nn.Linear` projection layers are replaced with sparse **Dendritic Linear Layers**.

---

## ⚡ Technical Innovation: Dendritic Optimization
This project integrates the **Perforated AI Open Source Library** (Experimental `pep8` Branch) to enable "Dendritic Computation."

Unlike static compression, our model uses **Dynamic Dendritic Switching**:
1.  **Plastic Phase:** The network "grows" sparse connections to learn complex signal patterns (like spoofed headers).
2.  **Stable Phase:** The network solidifies efficient pathways and prunes the rest.
3.  **Result:** A "breathing" neural network that adapts its topology during training.

**Implementation Highlight:**
We successfully integrated the `initialize_pai` transformation to surgically replace standard PyTorch layers at runtime:

```python
# From train.py
self.model = UPA.initialize_pai(
    self.model, 
    doing_pai=True,          
    making_graphs=False,     
    maximizing_score=True    
)
