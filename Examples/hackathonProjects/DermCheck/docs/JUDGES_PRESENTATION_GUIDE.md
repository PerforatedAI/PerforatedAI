# 🏆 Hackathon Judges' Presentation Guide

This guide outlines how to showcase **GuardianEdge** and **DermCheck** to project a winning impression. Your submission demonstrates the "New Framework Integration" bonus across two separate domains: Object Detection (YOLO) and Medical AI (MONAI).

---

## 🏗️ 1. Technical Showcase (PerforatedAI Optimization)

The core value proposition is **Dendritic Optimization**. Show the judges the technical proof:

### 📈 Optimization Graphs
Show the automatically generated PAI graphs to prove the model improved through dendrite addition.
- **DermCheck:** Show `PAI_DermCheck/PAI_DermCheck.png`
  - *Point out*: The "Validation Scores" increasing and the "PB Scores" (Perforated Backpropagation) showing which layers benefited most from dendrites.
- **GuardianEdge:** Show the equivalent graph in the GuardianEdge project folder.

### 🔬 Architecture Integration
Show the `README.md` Mermaid diagrams in both projects.
- Explain how PerforatedAI was initialized (`UPA.initialize_pai`) and how it managed the dendritic growth without manual hyperparameter tuning.

---

## 🛡️ 2. Project 1: GuardianEdge (YOLO Integration)

**Key Narrative:** "Local-first security that respects privacy while running on edge hardware."

### Live Demo Flow:
1. **Explain:** "We integrated PerforatedAI with Ultralytics YOLO to optimize threat detection for decentralized security cameras."
2. **Run:** `python demo_quick.py`
3. **Show:** Point the webcam at common objects. Explain that the model has been optimized to be smaller and faster while retaining accuracy through dendritic growth.
4. **Highlight:** Performance metrics (FPS/Inference Time) shown on the screen.

---

## 🏥 Project 3: DermCheck (MONAI Integration)

**Key Narrative:** "Clinical-grade Medical AI that operates entirely offline, keeping sensitive patient data secure."

### Live Demo Flow:
1. **Explain:** "This is a medical application using the MONAI framework. We implemented a dual-task network for both Classification and Segmentation of skin lesions."
2. **Show Proof of Scale:** "We benchmarked this on the HAM10000 dataset, processing over 10,000 dermatoscopic images locally."
3. **Run Inference:**
   ```powershell
   python .\inference_classify.py --model .\models\classification_best.pt --image .\melanoma_sample.jpg
   ```
4. **Run Integrated Demo:**
   ```powershell
   python .\demo_integrated.py --classify-model .\models\classification_best.pt --segment-model .\models\classification_best.pt --source .\sample_lesion.jpg
   ```
5. **Highlight:** The segmentation overlay and the risk assessment ("HIGH RISK/Consult Specialist").

---

## 🔥 10. The "Winning Edge" (Bonus Points)

When speaking to judges, emphasize these three points:

1. **Dual Framework Mastery:** "We didn't just optimize one model; we successfully integrated PerforatedAI into two industry-standard frameworks: **Ultralytics YOLO** and **MONAI**."
2. **Real-World Impact:** "One project solves a security problem (GuardianEdge), the other solves a healthcare privacy problem (DermCheck)."
3. **Privacy & Offline:** "Both projects run entirely local. We are using PerforatedAI to make AI powerful enough to be useful on the edge, not just in the cloud."

---

**Good luck at the hackathon! 🚀**
