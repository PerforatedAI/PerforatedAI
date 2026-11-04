---
name: perforatedai-distributed
description: "Multi-GPU setup for PerforatedAI with DistributedDataParallel (DDP). Invoked automatically by the perforatedai skill when multi-GPU training is detected; DataParallel is not supported. Handles the DDP wrap, optimizer setup, the validation-loop pattern that re-wraps the model after dendrites are added, configuration confirmation, and the torchrun launcher."
---

# PerforatedAI Multi-GPU (Distributed) Setup Skill

This skill handles multi-GPU setup for PerforatedAI, which uses DistributedDataParallel (DDP). It is automatically invoked by the main perforatedai skill when multi-GPU training is detected. `torch.nn.DataParallel` is not supported: a DataParallel script must be converted to DDP before using this skill.

**Prerequisites:**
- User has already completed detection in main perforatedai skill (Step 5)
- Model initialization with `UPA.perforate_model()` has been added
- User confirmed they want multi-GPU training, and the script uses DDP (or has been converted from DataParallel to DDP)

**How PerforatedAI behaves with multiple GPUs (read before editing their script):**
- Multi-GPU training is supported through DDP only. `torch.nn.DataParallel` is not supported; PyTorch itself recommends DDP over DataParallel, even on a single machine.
- `GPA.pai_tracker.add_validation_score(score, model)` accepts the bare model or a `DistributedDataParallel` wrapper. Pass the model exactly as it is used for training - do NOT unwrap it with `.module`.
- When the model was **not** restructured, it is returned in the same DDP wrapper it came in.
- When the model **was** restructured (dendrites added or removed), the **unwrapped** inner model is returned. The script must move it to the device, wrap it in DDP again with the same arguments as the original wrap, and rebuild the optimizer and scheduler.
- If the script already has pieces of an older PerforatedAI multi-GPU setup (a `--perforate_model_parallel` initialization run, `save_tracker_settings` / `initialize_tracker_settings` calls, or gradient pre-initialization), remove them; none of them are part of the current setup.

---

## DistributedDataParallel (DDP) Setup

**Use this section for scripts that use `torch.nn.parallel.DistributedDataParallel`.**

Tell them:

> **DistributedDataParallel with PerforatedAI**
>
> DDP runs as a single `torchrun` launch. When dendrites are added, every GPU process picks up the new model during the run. PerforatedAI keeps the processes in sync itself: rank 0 makes the dendrite decisions, the new model is shared with every rank, and candidate gradients and optimizer/scheduler settings are synchronized automatically.
>
> Two things are needed: the PerforatedAI configuration has to be confirmed before launching with `torchrun` (the interactive configuration screen needs a terminal, which `torchrun` processes don't have), and the DDP wrapper needs two specific arguments.

**Steps:**

1. **Make sure the configuration is confirmed before the first `torchrun` launch:**

Check how their script reaches `perforate_model`. If the configuration has not been confirmed yet, `perforate_model` opens an interactive configuration screen, which cannot run under `torchrun` (every rank stops with an error explaining this).

Tell them to do ONE of the following before using `torchrun`:
- Run the script once as a single process in a terminal (without `torchrun`), complete the configuration screen, then stop it once training starts. The confirmed configuration is saved in the `save_name` folder and every rank reads it.
- Or, if the settings are already fully specified in the script or config file, add `GPA.pc.set_configuration_confirmed(True)` before `UPA.perforate_model(...)`.

2. **Optional - crash resume (only if the user wants it):**

Ask: "Do you want the ability to resume from the last dendrite checkpoint if training crashes?" Dendrite additions do NOT need this - they are handled in-process. Only add it if they say yes.

Add a command-line argument:
```python
parser.add_argument('--pai_load_folder', type=str, default=None,
                   help='Folder to resume PAI state from after a crash')
```

Add this logic RIGHT AFTER model creation and `perforate_model()`, BEFORE creating the optimizer:

```python
# Resume from the latest dendrite checkpoint after a crash
if args.pai_load_folder is not None:
    import glob
    switch_numbers = []
    for f in glob.glob(f"{args.pai_load_folder}/switch_*.pt"):
        try:
            switch_numbers.append(int(f.split('switch_')[1].split('.pt')[0]))
        except ValueError:
            pass  # e.g. switch_3_pai.pt
    if switch_numbers:
        max_switch = max(switch_numbers)
        model = UPA.load_system(model, args.pai_load_folder, f'switch_{max_switch}', True)
        print(f"Loaded PAI state from {args.pai_load_folder}/switch_{max_switch}.pt")
    else:
        print(f"Starting from beginning (no switch_x.pt files found in {args.pai_load_folder})")
```

**Why this order matters:** a checkpoint with dendrites has a different model structure. `load_system` rebuilds that structure, and the optimizer must be created from the rebuilt model's parameters, so loading must happen BEFORE optimizer creation.

3. **Setup Optimizer and Scheduler:**

Find where their optimizer and scheduler are currently defined in their script.

**🚨 CRITICAL RULE: PRESERVE USER'S EXACT OPTIMIZER AND SCHEDULER TYPES AND ARGUMENTS**

**If their setup is clean (2-5 lines in one place):**

Replace their optimizer/scheduler code with the PAI pattern **while keeping the EXACT SAME types and arguments.**

**Example - User has Adam optimizer with StepLR scheduler:**
Original:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

Correct PAI conversion (PRESERVES their choices):
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)  # SAME optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.StepLR)  # SAME scheduler type
optim_args = {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}  # SAME args
sched_args = {'step_size': 30, 'gamma': 0.1}  # SAME scheduler args
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
```

**If their setup is complex (scattered across many lines or conditional):**

Keep their code and register the optimizer with `GPA.pai_tracker.set_optimizer_instance(optimizer)` (the main skill's Step 6, second option, describes this). If you can't tell where it should go, ask the user: "Your optimizer/scheduler setup is complex. Can you point me to where you want PAI's optimizer setup to go, and confirm which optimizer and scheduler types you want to use?"

4. **Wrap the model in DDP with the PerforatedAI arguments:**

Find their DDP initialization:
```python
model = torch.nn.parallel.DistributedDataParallel(model, ...)
```

Replace it with a kwargs dict so the exact same arguments can be reused after a restructure:
```python
# Required for PerforatedAI: not every parameter receives a gradient in every training phase,
# and broadcast_buffers=False is the tested configuration
ddp_kwargs = dict(
    device_ids=[...],  # their original device_ids if they had them
    find_unused_parameters=True,
    broadcast_buffers=False,
)
model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
```

**IMPORTANT:** If their original DDP call had other arguments (like `output_device`), keep them in `ddp_kwargs`, and set `find_unused_parameters=True` and `broadcast_buffers=False`.

**PyTorch 2.13 and newer:** `broadcast_buffers` is deprecated there (it still works, with a FutureWarning). Its replacement, `forward_sync_buffers=False`, works the same way with PerforatedAI; the one difference is that DDP copies rank 0's buffers to the other ranks once when the wrapper is built, which is harmless because every rank holds identical buffers at that point. `forward_sync_buffers` doesn't exist before 2.13, so only use it if their PyTorch is 2.13 or newer.

5. **Use this validation-loop pattern** (it replaces the `add_validation_score` block from the main skill's Step 7):

⚠️ **CRITICAL: `add_validation_score` must be called on EVERY rank.** It coordinates the ranks internally. Do NOT guard it with `if rank == 0` - a rank-0-only call hangs training.

```python
# After validation completes and you have val_score/val_acc/val_loss

# Only rank 0's value is used for dendrite decisions. If each rank validated a different
# shard of the validation set, combine the metric first so the decision reflects all of it:
# val_tensor = torch.tensor([val_score], device=device)
# torch.distributed.all_reduce(val_tensor)
# val_score = val_tensor.item() / torch.distributed.get_world_size()

model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score, model)  # pass the DDP model as is

if training_complete:
    if torch.distributed.get_rank() == 0:
        print("PAI training complete!")
        # Optional: write a marker file so launch scripts can tell training finished
        os.makedirs(save_name, exist_ok=True)
        with open(f"{save_name}/.training_complete", "w") as f:
            f.write("complete")
    torch.distributed.barrier()  # all ranks finish their I/O before the process group goes away
    torch.distributed.destroy_process_group()
    sys.exit(0)

elif restructured:
    # Every rank now holds the identical restructured model, returned unwrapped:
    # move it, re-wrap it with the same DDP arguments, then rebuild the optimizer
    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)
    optim_args = {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}  # EXACT SAME as step 3
    sched_args = {'step_size': 30, 'gamma': 0.1}  # EXACT SAME as step 3
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
```

**IMPORTANT NOTES:**
- Replace `val_score` with their actual validation metric variable name and `save_name` with their actual save folder, and make sure `os` and `sys` are imported.
- Check whether their validation already averages the metric across ranks (for example a `synchronize_between_processes` or `all_reduce` call). If it does, don't add another reduction. If every rank validates the full set, no reduction is needed.
- `add_extra_score()` calls can stay exactly where they are; they work the same on every rank.
- If they used `set_optimizer_instance` in step 3, rebuild their optimizer exactly as originally defined in the `restructured` branch and call `GPA.pai_tracker.set_optimizer_instance(optimizer)` again.
- **Do NOT** add any rank-0-only PAI calls, manual broadcasts of `restructured` / `training_complete`, exits on restructure, or gradient pre-initialization - PerforatedAI handles all of that.
- If the model is not wrapped in DDP again after a restructure, the next `add_validation_score` call raises an error saying the model parameters differ between DDP ranks.

6. **Ask critical DDP configuration questions:**

Before creating the launcher, ask:

**Required:**
- "How many GPUs do you want to use for training?"

**Optional (only ask if relevant):**
- If they want to use specific GPU IDs (not all GPUs): "Do you want to use specific GPU IDs, or use all available GPUs? (e.g., to use GPUs 0,1,2 instead of 0,1,2,3)"
- If they're not using standard torchrun: "Are you using `torchrun` to launch DDP, or a different launcher like `python -m torch.distributed.launch`?"

Wait for their answers.

7. **Create a launcher script with their configuration:**

Based on their answers, create a file `train_distributed.sh` in the same directory.

**If they're using torchrun (most common):**

```bash
#!/bin/bash

# PerforatedAI DistributedDataParallel training
# Dendrite additions are handled inside the run, so a single launch is enough.
# Confirm the PAI configuration first (see step 1) - torchrun processes can't show the configuration screen.

PYTHON_SCRIPT="[their_script_name]"  # Their actual script filename
NUM_GPUS=[their_num_gpus]  # Number of GPUs they specified

torchrun --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT "$@"
```

**If they specified GPU IDs:**

Add this environment variable at the top of the script:
```bash
export CUDA_VISIBLE_DEVICES=[their_gpu_ids]  # e.g., "0,1,2"
```

**If they added crash resume (step 2):**

Tell them to relaunch after a crash with the save folder, e.g. `./train_distributed.sh --pai_load_folder [their_save_name]`.

**If they're using a different launcher:**

Replace the `torchrun` command with their launcher. For example, if using `python -m torch.distributed.launch`:
```bash
python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS $PYTHON_SCRIPT "$@"
```

Fill in the actual values:
- Replace `[their_script_name]` with their Python script filename
- Replace `[their_num_gpus]` with the number they specified
- Replace `[their_gpu_ids]` if they specified specific IDs (e.g., "0,1")
- Replace `[their_save_name]` with the save_name from `UPA.perforate_model()` call

**Instruct the user to make it executable:**
```bash
chmod +x train_distributed.sh
```

Tell them:
> "I've set up your script for DistributedDataParallel and created `train_distributed.sh` with your configuration ([NUM_GPUS] GPUs). Before the first launch, confirm the PerforatedAI configuration [by running the script once in a terminal without torchrun / already done via `set_configuration_confirmed(True)`]. Then run `./train_distributed.sh` in your terminal. Training runs as a single launch: when dendrites are added, every GPU switches to the new model and continues, and the run exits on its own when PerforatedAI reports training is complete. Press Ctrl+C to stop training early."

Replace [NUM_GPUS] with the actual number they specified, and keep only the configuration option that applies.

---

## Completion

After completing the DDP setup:

**✅ Multi-GPU setup is COMPLETE. You have:**
- Placed the DDP wrapper after `perforate_model` and `model.to(device)`, with `find_unused_parameters=True` and `broadcast_buffers=False`
- Added optimizer setup
- Written the validation-loop pattern that re-wraps the model after a restructure
- Made sure the configuration is confirmed before `torchrun`
- (Optional) Added crash resume with `--pai_load_folder`
- Created the `train_distributed.sh` launcher

**📍 RETURN TO MAIN SKILL:**
- **SKIP Step 6** (optimizer already added)
- Go directly to main skill **Step 7 (Update Training Loop)** for the remaining training-loop changes (such as the `while True` loop), but keep the `add_validation_score` block from this skill instead of Step 7's single-GPU patterns
