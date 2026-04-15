---
name: perforatedai-distributed
description: "Multi-GPU setup for PerforatedAI with DataParallel or DistributedDataParallel (DDP). Invoked automatically by the perforatedai skill when multi-GPU training is detected. Handles initialization workflow, checkpoint loading, rank 0 handling, and shell script generation for DDP."
---

# PerforatedAI Multi-GPU (Distributed) Setup Skill

This skill handles DataParallel and DistributedDataParallel (DDP) setup for PerforatedAI. It is automatically invoked by the main perforatedai skill when multi-GPU training is detected.

**Prerequisites:**
- User has already completed detection in main perforatedai skill (Step 4.1)
- Model initialization with `UPA.initialize_pai()` has been added
- User confirmed they want to use DataParallel or DDP

---

## DataParallel Setup

**Use this section when user is using `torch.nn.DataParallel`.**

**Educate them about the additional complexity:**

Tell them:

> **Important: DataParallel Setup with PerforatedAI**
>
> DataParallel with PerforatedAI requires a simple two-step initialization process:
> 1. First run: Initialize multi-GPU settings (exits after one batch)
> 2. Second run: Actual training with multi-GPU support
>
> This is a one-time setup. After initialization, you just run your training normally.

**Steps:**

1. **Add command-line argument to their script:**

Find their argparse section (or add one if they don't have it) and add:

```python
parser.add_argument('--initialize_pai_parallel', action='store_true', 
                   help='Initialize PAI settings for multi-GPU (run once on single GPU)')
```

2. **Add conditional initialization code after `initialize_pai` at the location of their DataParallel line:**

Find their DataParallel line:
```python
model = torch.nn.DataParallel(model)
```

Replace it with:
```python
# PAI multi-GPU setup - initialize on single GPU, then use DataParallel
if not args.initialize_pai_parallel:
    # Normal training mode - load settings and wrap with DataParallel
    GPA.pai_tracker.initialize_tracker_settings()
    model = torch.nn.DataParallel(model)
# else: initialization mode - stay on single GPU, will save settings after first batch
```

3. **Add initialization mode handling in training loop:**

⚠️ **IMPORTANT: This must be added INSIDE the batch/iteration loop, immediately after `loss.backward()`, so it exits after processing just ONE batch (not after a full epoch).**

Find their batch loop (usually `for batch in train_loader:` or similar) and add this code immediately after `loss.backward()`:

```python
# Inside the batch loop:
for batch in train_loader:  # Their loop
    # ... their training code ...
    loss.backward()
    
    # PAI initialization mode - save settings and exit after FIRST batch
    if args.initialize_pai_parallel:
        GPA.pai_tracker.save_tracker_settings()
        print(f"PAI multi-GPU settings saved to {{save_name}}/")
        print("Initialization complete. Now run without --initialize_pai_parallel flag for training.")
        exit(0)
    
    optimizer.step()  # Their code continues
    # ...
```

Tell them:
> "I've set up your script for DataParallel. To train:
> 1. First run: `python train.py --initialize_pai_parallel` (runs on single GPU, exits after ONE batch)
> 2. Second run: `python train.py` (uses DataParallel for multi-GPU training)
> 3. If you change any PAI configuration settings later, re-run step 1"

---

## DistributedDataParallel (DDP) Setup

**Use this section when user is using `torch.nn.parallel.DistributedDataParallel`.**

**Educate them about the additional complexity:**

Tell them:

> **Important: DistributedDataParallel Setup with PerforatedAI**
>
> DistributedDataParallel with PerforatedAI requires a special workflow because when dendrites are added, the process needs to exit and restart. I'll create a shell script that automates this entire process:
> - The script handles initialization and continuous training automatically
> - When training is interrupted (e.g., dendrites added), it automatically restarts and resumes
> - You just run one command and let it handle everything

**Steps:**

1. **Add command-line arguments to their script:**

Find their argparse section (or add one if they don't have it) and add:

```python
parser.add_argument('--initialize_pai_parallel', action='store_true', 
                   help='Initialize PAI settings for DDP (run once)')
parser.add_argument('--pai_load_folder', type=str, default=None,
                   help='Folder to load PAI state from (for automatic resumption)')
```

2. **Add checkpoint loading logic BEFORE optimizer creation:**

⚠️ **CRITICAL: This must be added RIGHT AFTER model creation and `initialize_pai()`, BEFORE creating the optimizer.**

Find where they create their model and call `initialize_pai`:
```python
model = YourModel(...)
model = UPA.initialize_pai(model, save_name="...", maximizing_score=...)
model = model.to(device)
```

Add this loading logic immediately after (still BEFORE any optimizer creation):

```python
# Load checkpoint if resuming from dendrite restructure (DDP mode)
if args.pai_load_folder is not None:
    # Find the highest numbered switch_x.pt file
    import glob
    switch_files = glob.glob(f"{args.pai_load_folder}/switch_*.pt")
    if switch_files:
        # Extract switch numbers and find the maximum
        switch_numbers = []
        for f in switch_files:
            try:
                num = int(f.split('switch_')[1].split('.pt')[0])
                switch_numbers.append(num)
            except:
                pass
        if switch_numbers:
            max_switch = max(switch_numbers)
            model = UPA.load_system(model, args.pai_load_folder, f'switch_{max_switch}', True)
            print(f"Loaded PAI state from {args.pai_load_folder}/switch_{max_switch}.pt")
        else:
            print(f"Starting from beginning (no valid switch_x.pt found in {args.pai_load_folder})")
    else:
        print(f"Starting from beginning (no switch_x.pt files found in {args.pai_load_folder})")

# ⚠️ STOP - DO NOT add optimizer code yet! ⚠️
# The PAI setup_optimizer call (from main skill Step 5) MUST be placed HERE,
# AFTER the checkpoint loading above, BEFORE the DDP wrapper below.
# This is CRITICAL - if optimizer is created before loading, it will have wrong parameters!
```

**Why this order matters:** When dendrites are added, the model structure changes. `load_system` loads the new structure. The optimizer needs to be created with the correct model parameters, so loading must happen BEFORE optimizer creation.

**⚠️ CRITICAL ORDERING FOR YOUR SCRIPT:**
1. Model creation + initialize_pai() [already done]
2. Checkpoint loading [code block above]  ← YOU ARE HERE
3. **Optimizer creation** [`GPA.pai_tracker.setup_optimizer()`] ← GOES HERE NEXT (main skill Step 5)
4. DDP wrapper [step 3 below] ← AFTER optimizer
5. Training loop modifications [steps 4-5 below]

**When you return to the main skill's Step 5 (Optimizer Setup), place the `setup_optimizer` code RIGHT HERE in your script - after the checkpoint loading block above, before the DDP wrapper below.**

3. **Add conditional DDP wrapper at their DDP location:**

Find their DDP initialization:
```python
model = torch.nn.parallel.DistributedDataParallel(model, ...)
```

Replace it with:
```python
# PAI DDP setup - initialize on single GPU, then use DDP
if not args.initialize_pai_parallel:
    # Normal training mode - initialize settings and wrap with DDP
    GPA.pai_tracker.initialize_tracker_settings()
    
    # Wrap with DDP - IMPORTANT: find_unused_parameters=True is required for PerforatedAI
    model = torch.nn.parallel.DistributedDataParallel(
        model, 
        device_ids=[...],  # Their original device_ids if they had them
        find_unused_parameters=True  # Required for PerforatedAI dendrite growth
    )
# else: initialization mode - stay on single GPU, will save settings after first batch
```

**IMPORTANT:** If their original DDP call had other arguments (like `device_ids`, `output_device`, `broadcast_buffers`, etc.), preserve those arguments and add `find_unused_parameters=True` to them.

4. **Add initialization mode handling in training loop:**

⚠️ **IMPORTANT: This must be added INSIDE the batch/iteration loop, immediately after `loss.backward()`, so it exits after processing just ONE batch (not after a full epoch).**

Find their batch loop (usually `for batch in train_loader:` or similar) and add this code immediately after `loss.backward()`:

```python
# Inside the batch loop:
for batch in train_loader:  # Their loop
    # ... their training code ...
    loss.backward()
    
    # PAI initialization mode - save settings and exit after FIRST batch
    if args.initialize_pai_parallel:
        GPA.pai_tracker.save_tracker_settings()
        print(f"PAI DDP settings saved to {{save_name}}/")
        print("Initialization complete. Now run without --initialize_pai_parallel flag for training.")
        exit(0)
    
    optimizer.step()  # Their code continues
    # ...
```

5. **Modify the validation loop to handle DDP correctly:**

⚠️ **CRITICAL: DistributedDataParallel requires special handling for PAI tracker functions.**

In the section where they call `add_validation_score` (and any `add_extra_score` calls), replace it with this DDP-aware pattern:

**📋 TESTED WORKING PATTERN - Use this exact structure:**

```python
# After validation completes and you have val_score/val_acc/val_loss

# Support both DDP and single GPU modes
if args.distributed:
    # DDP mode - only rank 0 calls PAI tracker functions
    if torch.distributed.get_rank() == 0:
        # Unwrap model from DDP wrapper
        model_unwrapped = model.module
        model_unwrapped, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model_unwrapped)
        
        # If adding extra scores, do those here too:
        # model_unwrapped = GPA.pai_tracker.add_extra_score(extra_score, model_unwrapped, score_name="extra_metric")
        
        # Re-wrap the updated model
        model.module = model_unwrapped
        model.module = model.module.to(device)
    else:
        # Other ranks skip PAI calls
        restructured = False
        training_complete = False
    
    # Broadcast restructured and training_complete from rank 0 to all ranks
    restructured_tensor = torch.tensor([1 if restructured else 0], dtype=torch.int, device=device)
    training_complete_tensor = torch.tensor([1 if training_complete else 0], dtype=torch.int, device=device)
    torch.distributed.broadcast(restructured_tensor, src=0)
    torch.distributed.broadcast(training_complete_tensor, src=0)
    restructured = bool(restructured_tensor.item())
    training_complete = bool(training_complete_tensor.item())
else:
    # Single GPU mode
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
    model = model.to(device)

# Handle training completion
if training_complete:
    print("PAI training complete!")
    if args.distributed:
        # Create completion marker for shell script (rank 0 only, but all ranks will exit)
        if torch.distributed.get_rank() == 0:
            os.makedirs(save_name, exist_ok=True)
            with open(f"{save_name}/.training_complete", "w") as f:
                f.write("complete")
        # Clean up distributed process group (all ranks)
        torch.distributed.destroy_process_group()
    sys.exit(0)

# Handle model restructuring (dendrite added)
elif restructured:
    print("Model restructured! Exiting for restart...")
    if args.distributed:
        # Clean up distributed process group (all ranks)
        torch.distributed.destroy_process_group()
    exit(0)  # Shell script will restart training automatically
```

**IMPORTANT NOTES:**
- This pattern supports both DDP and single GPU modes using `args.distributed` flag
- Replace `val_acc` with their actual validation metric variable name
- Replace `save_name` with their actual save folder variable
- This same pattern applies to **both** `add_validation_score()` and `add_extra_score()` calls
- Only rank 0 calls PAI tracker functions in DDP mode
- Unwrap model from DDP wrapper (`.module`) before PAI calls
- Broadcast results to all ranks in DDP mode
- Call `destroy_process_group()` on all ranks when exiting

6. **Ask critical DDP configuration questions:**

Before creating the shell script, ask:

**Required:**
- "How many GPUs do you want to use for training?"

**Optional (only ask if relevant):**
- If they want to use specific GPU IDs (not all GPUs): "Do you want to use specific GPU IDs, or use all available GPUs? (e.g., to use GPUs 0,1,2 instead of 0,1,2,3)"
- If they're not using standard torchrun: "Are you using `torchrun` to launch DDP, or a different launcher like `python -m torch.distributed.launch`?"

Wait for their answers.

7. **Create shell script with their configuration:**

Based on their answers, create a file `train_distributed.sh` in the same directory.

**If they're using torchrun (most common):**

```bash
#!/bin/bash

# PerforatedAI DistributedDataParallel Training Script
# This script handles automatic restarting when dendrites are added

SAVE_NAME="[their_save_name]"  # Use the save_name from UPA.initialize_pai
PYTHON_SCRIPT="[their_script_name]"  # Their actual script filename
NUM_GPUS=[their_num_gpus]  # Number of GPUs they specified

echo "Step 1: Initializing PAI DDP settings..."
# Run initialization on single GPU (no DDP launcher)
python $PYTHON_SCRIPT --initialize_pai_parallel

echo ""
echo "Initialization complete. Starting continuous DDP training loop..."
echo "Press Ctrl+C to stop training"
echo ""

# Continuous training loop with DDP launcher
while true; do
    # Check if any switch_*.pt checkpoint files exist
    if ls "${SAVE_NAME}"/switch_*.pt 1> /dev/null 2>&1; then
        echo "Resuming training from checkpoint..."
        torchrun --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT --pai_load_folder $SAVE_NAME
    else
        echo "Starting training from beginning..."
        torchrun --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT
    fi
    
    # Check exit code
    EXIT_CODE=$?
    
    # Check if training completed successfully
    if [ -f "${SAVE_NAME}/.training_complete" ]; then
        echo "Training completed successfully!"
        break
    fi
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Model restructured. Restarting in 2 seconds..."
        sleep 2
    else
        echo "Error occurred. Exiting..."
        exit $EXIT_CODE
    fi
done
```

**If they specified GPU IDs:**

Add this environment variable at the top of the script:
```bash
export CUDA_VISIBLE_DEVICES=[their_gpu_ids]  # e.g., "0,1,2"
```

**If they're using a different launcher:**

Replace the `torchrun` commands with their launcher. For example, if using `python -m torch.distributed.launch`:
```bash
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT --pai_load_folder $SAVE_NAME
```

Fill in the actual values:
- Replace `[their_save_name]` with the save_name from `UPA.initialize_pai()` call
- Replace `[their_script_name]` with their Python script filename
- Replace `[their_num_gpus]` with the number they specified
- Replace `[their_gpu_ids]` if they specified specific IDs (e.g., "0,1")

Make it executable:
```bash
chmod +x train_distributed.sh
```

Tell them:
> "I've set up your script for DistributedDataParallel and created `train_distributed.sh` with your configuration ([NUM_GPUS] GPUs). To train:
> 1. Run `./train_distributed.sh` - this handles everything automatically
> 2. The script will initialize DDP settings (single GPU), then continuously train with [NUM_GPUS] GPUs
> 3. When dendrites are added (model restructured), the script automatically restarts and resumes
> 4. Press Ctrl+C to stop training"

Replace [NUM_GPUS] with the actual number they specified.

---

## Completion

After completing either DataParallel or DDP setup, return control to the main perforatedai skill to continue with Step 5 (Optimizer Setup).
