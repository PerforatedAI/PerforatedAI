@echo off
REM Run training in Dendritic mode
REM --mode dendritic: Enables PerforatedAI
REM --pai_switch_mode fixed: Forces dendrites to be added at a specific epoch
REM --pai_switch_epoch 5: The epoch to add dendrites (triggers the "branching" graph)
REM --epochs 20: Run long enough to see the effect

echo Starting Dendritic Demo Run...
# Run with recommended settings
python train.py \
    --mode dendritic \
    --epochs 50 \
    --pai_switch_epoch 15 \
    --arch resnet50 \
    --dataset cifar100 \
    --run_name "dendritic_fixed_v2"

pause
