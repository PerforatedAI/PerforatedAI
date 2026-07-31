# GuardianEdge - Project Submission

## Elevator pitch
**GuardianEdge**: Privacy-first, local-only security intelligence optimized for the edge with PerforatedAI.

---

## About the project

### Inspiration
The inspiration for GuardianEdge came from the growing concern over privacy in home and commercial security. Most modern smart cameras rely on cloud-based AI, which not only introduces latency in critical threat detection (like fire or weapons) but also uploads sensitive video data to remote servers. We wanted to build a "blind" security system—one that is intelligent enough to detect threats locally but never needs to send a single pixel to the cloud.

### What it does
GuardianEdge is a real-time security application that uses a PerforatedAI-optimized YOLO (You Only Look Once) model. It performs on-device object detection to identify specific threat classes—such as knives, firearms, or fire—while ignoring non-threatening activity. 

By leveraging dendritic optimization, the system can run on less powerful edge hardware (like high-end IoT devices or local workstations) with higher efficiency than standard large-scale models. When a threat is detected, it triggers a local alert and log without compromising the user's data privacy.

### How I built it
The project is built on the **Ultralytics YOLOv8** framework. The core innovation is the integration of **PerforatedAI's dendritic optimization**.
1. **Model Architecture**: I used YOLOv8n (nano) as the baseline for high speed on edge devices.
2. **PAI Integration**: Using `perforatedai`, I initialized the YOLO backbone with dendritic structures.
3. **Training**: I implemented a custom training loop that uses Perforated Backpropagation (PB) to find the most "important" neurons and grow dendrites around them. This is represented by the formula:
   $$W_{new} = W_{old} + \Delta W_{PAI}$$
   where $\Delta W_{PAI}$ is optimized via the Dendritic tracker.
4. **Inference**: A Python-based real-time wrapper using OpenCV handles the video stream and overlays the optimized detections.

### Challenges I ran into
The biggest challenge was integrating a custom optimization framework into the highly encapsulated `ultralytics` training pipeline. I had to bypass the standard `.train()` method and write a manual loop that correctly managed the `GPA.pai_tracker` callbacks at the end of each epoch to ensure the dendrites actually "grew" based on the validation scores.

### Accomplishments that I'm proud of
I am incredibly proud of achieving a stable dendritic growth pattern that showed measurable improvement in the PAI validation graphs. Seeing the "PB Scores" rise for specific convolutional layers proved that the optimization was surgically targeting the parts of the model that needed help.

### What I learned
I learned that "smaller is often better" when it comes to edge AI. By using dendritic optimization, you can make a nano-sized model behave with the precision of a much larger one, which is the "holy grail" for privacy-focused hardware.

### What's next for it
The next step is to port the optimized models to specialized hardware like the NVIDIA Jetson or Coral TPU. I also plan to implement "Secure Enclave" storage where even the local logs are encrypted with a key only the user holds.

---

## Built with
- **Frameworks**: Ultralytics YOLOv8, PerforatedAI
- **Language**: Python 3.10
- **Libraries**: PyTorch, OpenCV, PyYAML, Matplotlib
- **Payment**: polar.sh integration for decentralized licensing
