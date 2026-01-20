**Intro** - 


CIFAR-10 is a standard dataset used to benchmark image classification models. This hackathon submission tests adding Perforated AI dendrites (Dendritic Optimization) to a ResNet18-based CIFAR-10 classifier in PyTorch, and compares its performance against the same baseline ResNet18 model trained without dendrites.

Team:
Yatharth Khanna - Student/Founder@Meta Infinium (https://www.linkedin.com/in/yatharth-khanna-960ab224b/)

**Project Impact** - 


Improving the accuracy of an image classification model matters because it reduces incorrect predictions and improves reliability in real-world applications like robotics, smart cameras, edge-device vision, and automated quality inspection. Even small improvements in test accuracy can meaningfully reduce remaining error and increase confidence in deployment scenarios where consistent recognition performance is important.

**Usage Instructions** - 

Installation:
pip install -r requirements.txt

Run Baseline (No Dendrites):
python cifar10_resnet18_baseline.py

Run Dendritic Optimization (PerforatedAI):
python cifar10_resnet18_perforatedai.py

**Results**

This project compares the final test accuracy of the ResNet18 CIFAR-10 baseline model vs the same architecture trained with PerforatedAI dendrites.

Model Final Test Accuracy Notes
Traditional (No Dendrites) 85.71% Baseline ResNet18 CIFAR-10

<img width="615" height="457" alt="Screenshot (522)" src="https://github.com/user-attachments/assets/1a325958-975d-4127-8e91-4b5f3ccc7e48" />

Dendritic (PerforatedAI) 93.48% ResNet18 + Dendritic Optimization

<img width="1355" height="656" alt="Screenshot (523)" src="https://github.com/user-attachments/assets/388a1114-8d6a-4c5a-8612-d54c212bdc80" />

Remaining Error Reduction:

Baseline Error = 100 - 85.71 = 14.29
Dendritic Error = 100 - 93.48 = 6.52
Remaining Error Reduction = (14.29 - 6.52) / 14.29 × 100 = 54.37%

This provides a Remaining Error Reduction of 54.37%

**Raw Results Graph** - 
<img width="2800" height="1400" alt="PAI" src="https://github.com/user-attachments/assets/7d78e269-730e-47f5-9a94-ab44266950e1" />


