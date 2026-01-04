
# Perforated Dendritic Inference Routing

This project demonstrates **dendritic routing as a drop-in inference-time optimization** for PyTorch models.  
The goal is to reduce unnecessary computation during inference while preserving model behavior, using a simple, reproducible, and offline-safe setup.

The submission focuses on **measured wall-clock latency, effective compute usage, and correctness trade-offs**, rather than training or accuracy benchmarks.

---

## Motivation

Modern neural networks typically execute all parameters for every input, even when many internal computations contribute little to the final output.  
This “always-on” execution model leads to unnecessary inference cost and limits scalability.

Dendritic routing provides a mechanism for **conditional computation**, inspired by biological neurons, where only a subset of computation is activated depending on input characteristics.

This project explores dendritic routing specifically as an **inference optimization primitive**, independent of training, datasets, or architectural changes.

---

## What This Project Does

- Implements a simple, deterministic PyTorch baseline model
- Wraps the model with a dendritic routing mechanism
- Applies **per-sample conditional execution**
- Measures:
  - Wall-clock latency
  - Throughput
  - Effective compute usage
  - Routing overhead
  - Output deviation from the baseline
- Visualizes the trade-off between compute usage and performance

The demo is:
- CPU-only
- Offline-safe
- Deterministic
- Reproducible with a single command

---

## Key Design Principles

### Drop-in Behavior
The baseline model, inputs, and outputs remain unchanged.  
Only the routing logic determines whether full computation is executed.

### Inference-Only Focus
No training loops, datasets, or accuracy benchmarks are included.  
This isolates the effect of dendritic routing on inference cost.

### Low Demo Risk
The system avoids GPU dependencies, external data, and randomness to ensure reliable evaluation.

### Honest Trade-offs
The demo explicitly reports both performance gains and overhead, avoiding optimistic assumptions.

---

## Project Structure

```text
perforated-inference-routing/
├── README.md
├── run_demo.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── baseline_model.py
│   └── dendritic_router.py
├── latency_tradeoff.png
└── speedup_tradeoff.png
````

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the demo

```bash
python run_demo.py
```

---

## Demo Metrics

The demo reports the following metrics for each routing threshold:

* **Latency (ms)**
  Mean wall-clock inference latency on CPU

* **Throughput (samples/sec)**
  Effective inference throughput

* **Speedup (×)**
  Relative to the baseline model

* **Active Computation (%)**
  Fraction of samples routed through full computation

* **Output Deviation**
  Maximum and mean absolute difference from baseline outputs

These metrics allow direct inspection of the **performance–correctness trade-off**.

---

## Interpreting the Results

For small CPU-bound models, absolute speedups are modest because Python overhead dominates execution time.
This is expected and intentional.

What the results demonstrate is that:

* Active computation decreases smoothly as routing becomes more selective
* Routing overhead remains bounded
* Output deviation is small at moderate thresholds
* Performance gains emerge as compute dominates routing cost

This behavior scales conceptually to larger models and deployment settings where computation, not Python overhead, is the bottleneck.

---

## Routing Overhead

The demo explicitly measures routing overhead when all computation is active.
This ensures that any observed speedup is due to reduced computation, not measurement artifacts.

Reporting overhead is critical to evaluating dendritic routing as a practical inference primitive.

---

## Limitations

* The baseline model is intentionally simple
* GPU benchmarks are out of scope
* No claims are made about accuracy improvements
* Power and memory measurements are not included

These limitations are deliberate to prioritize reproducibility and clarity.

---

## Why This Matters

Inference cost is a primary constraint in real-world model deployment.
Dendritic routing offers a general mechanism for:

* Reducing unnecessary computation
* Preserving model behavior
* Integrating with existing PyTorch systems
* Enabling conditional execution without retraining

This positions dendritic optimization as a **systems-level inference technique**, not just a training-time modification.

---

## Future Work

Potential extensions include:

* Applying routing to larger architectures
* Learned routing policies
* Hardware-aware routing decisions
* Integration with production inference pipelines

---

## Summary

This submission demonstrates dendritic routing as a **clean, drop-in inference optimization** with measurable effects on computation, latency, and throughput, while explicitly reporting correctness trade-offs and overhead.

The project emphasizes clarity, reproducibility, and honest evaluation over scale or complexity.



---
