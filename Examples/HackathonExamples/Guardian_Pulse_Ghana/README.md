# Guardian-Pulse Ghana

## Intro
**Description:**
This folder contains the Guardian-Pulse submission. The project integrates **Perforated AI Dendritic Optimization** into a custom Edge-AI model (`signal-geo`) to detect illicit IMSI Catchers (rogue cell towers) in Ghana. The code is self-contained in this folder.

**Team:**
Hadi - Lead Engineer / Developer

---

## Project Impact
**Description:**
Detecting illicit IMSI Catchers ("Stingrays") in real-time is a national security priority for Ghana. Standard deep learning models are often too computationally heavy or inaccurate for the edge devices (Raspberry Pi/Jetson) used by field teams. By utilizing Dendritic Optimization, we achieve **100% detection accuracy** on critical signal data. This allows us to deploy high-fidelity security systems on low-cost hardware, democratizing access to critical defense infrastructure.

---

## Usage Instructions
**Installation:**
```bash
# From the root of the repo:
pip install -e .
pip install -r Examples/HackathonExamples/Guardian_Pulse_Ghana/requirements.txt