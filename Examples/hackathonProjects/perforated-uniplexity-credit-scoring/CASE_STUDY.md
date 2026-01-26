# Optimizing Credit Scoring with Dendritic AI
### *How Uniplexity AI achieved remarkable efficiency results with Perforated Backpropagation™*

![Hero Image](PAI.png)

This case study comes from **Uniplexity AI**, a team working on an SME Credit Scoring model originally based on a standard Multi-Layer Perceptron (MLP). Their goal was to bring "Big Bank" risk analysis to the **edge**, running complex scoring on restricted hardware without internet access. Over the course of the hackathon, the team was able to optimize the model for both accuracy and extreme compression using Dendritic Optimization.

## The Team

This project was a solo engineering effort by **Praise Mwanza** (Uniplexity AI). With a focus on efficient deep learning and financial inclusion, Praise set out to explore how bio-inspired neural networks could solve the "Thin File" problem in credit scoring—providing accurate risk assessments for applicants with sparse credit history.

Their project used a synthetic **SME Credit Dataset** with engineered non-linear logic traps (e.g., conditional default risks). The goal was to maintain >98% accuracy while shrinking the model enough to run on a microcontroller.

## The Results

The real game changer was **Dendritic Fine-Tuning**, which allowed the model to learn the logic rules directly rather than approximating them. On the core credit scoring task, they saw a massive parameter reduction while maintaining parity with the baseline.

#### Accuracy Comparison
> The Dendritic model matched the heavy baseline's accuracy within 0.2% while using a fraction of the resources.

*   **Baseline (MLP):** 98.58%
*   **Dendritic (Lite):** 98.30%
*   **Hybrid Ensemble:** **98.78%** (New SOTA)

#### Model Efficiency (The "Wow" Factor)
> *Adding Dendritic Optimization reduced the model size by 79%.*

After characterizing the accuracy trade-offs, the team selected the "Lite" dendritic variant for deployment. Using this new architecture, they brought the parameter count down from **2,497** to just **529** and still achieved >98% accuracy.

| Metric | Before (MLP) | After (Dendritic) | Impact |
| :--- | :--- | :--- | :--- |
| **Parameters** | 2,497 | **529** | **79% Smaller** 📉 |
| **Data Required** | 100% | **25%** | **4x More Data Efficient** 🧠 |
| **Disk Size** | 12 KB | **4 KB** | **3x Smaller** 💾 |

## Implementation Experience

The technology was straightforward to implement, expanding on the base **PyTorch** training pipeline. After the initial setup of the `DendriticLayer`, scaling up required minimal code changes.

> "The ability to integrate the `DendriticLayer` directly into our `nn.Module` definition meant we could keep our existing training loops and data loaders. The biggest surprise was how quickly the model converged—it found the logic rules in just a few epochs compared to the baseline."
> — *Praise Mwanza, Lead Engineer*

## Tools Used
*   **Ray Train:** For distributed training on Windows.
*   **PyTorch Lightning:** For structured experimentation.
*   **Dendritic Layers:** For bio-inspired optimization.
