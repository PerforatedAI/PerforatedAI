---
name: perforatedai-demo
description: "One-line PerforatedAI demo: adds a single dendrite to a PyTorch training script with UPA.perforate_model(demo_mode=True) and no other PAI code. Triggers: 'Run the perforatedai demo', 'Demo perforatedai on my model', 'Try dendrites on my model quickly'. For a full integration with validation-driven dendrite growth, use the perforatedai skill ('Perforate my model') instead."
---

# PerforatedAI Demo Mode

Demo mode adds one dendrite to one layer of the user's model with a single call. The script first trains as usual (phase 1). Then the dendrite turns on, and the user's optimizer and learning rate schedule restart at 0.25x their starting learning rate (phase 2), replaying the same schedule. Comparing the validation score at the end of phase 1 with the score at the end of training is the demo.

**Your role is to analyze and edit the user's script. Never run their training script yourself; ask them to run it and share the output.**

## Step 1: Install

Ask the user to run:

```bash
pip install perforatedai
```

`perforatedbp` is not needed. If it is installed, demo mode turns Perforated Backpropagation off for the run.

## Step 2: Check the script is a good fit

Ask for the training script path and read it. Demo mode supports a single-process PyTorch training loop the user wrote themselves. Stop and recommend the full **perforatedai** skill ("Perforate my model") if the script uses any of these:

- HuggingFace `Trainer`, PyTorch Lightning, or another library that runs the training loop
- `DataParallel`, `DistributedDataParallel`, FSDP, or multiple processes
- More than one optimizer for the model (for example separate backbone and head optimizers)
- A Muon optimizer
- EMA or averaged copies of the model (`torch.optim.swa_utils.AveragedModel`, timm `ModelEma`, a hand-written EMA)
- Code that sets `param_group["lr"]` by hand every step instead of using a scheduler
- A model that is not called with a single tensor (`output = model(inputs)`) or does not return a single tensor (for example dict inputs, several arguments, or HuggingFace output objects)

Tell the user what you checked and whether you found any of these.

## Step 3: Ask how long training actually runs

**Ask:** "How many epochs does your training actually run? If you use early stopping, about how many epochs does it usually run before it stops?"

Call the answer `E`. It must not be larger than the epoch count the scheduler is built with (`OneCycleLR` raises an error if it is stepped past its end).

Then find in the script:

- The optimizer steps per epoch: usually `len(train_loader)`. If the script only calls `optimizer.step()` every `k` batches (gradient accumulation), it is `len(train_loader) // k`.
- The epoch variable (for example `args.epochs`) and every place it is used: the loop's `range(...)` and scheduler arguments such as `T_max`, `total_steps`, `epochs=`, `total_iters` or `num_training_steps`.
- Which part of each training batch the loop passes to the model. For `for data, target in train_loader: output = model(data)` it is the first item, so one batch of inputs is `next(iter(train_loader))[0]`.

## Step 4: Make the edits

1. **Import** at the top of the script:

   ```python
   from perforatedai import utils_perforatedai as UPA
   ```

2. **The one PAI line**, after the model and the training data loader are created (after `.to(device)` is fine) and **before the optimizer and scheduler are created**:

   ```python
   model = UPA.perforate_model(model, demo_mode=True, demo_phase_steps=E * len(train_loader),
                               demo_sample_input=next(iter(train_loader))[0])
   ```

   Put the user's actual `E` in (as a number or a new argument). `demo_phase_steps` counts optimizer steps, so this is the length of their actual training run. `demo_sample_input` is one batch of exactly what the loop passes to `model(...)`; it is run through the model once to create the dendrite. If the data loader is created after the model, move this line below the loader.

3. **Training loop:** run `2 * E` epochs with a count of its own, for example `for epoch in range(1, 2 * E + 1):`. **Do not change the epoch variable the scheduler is built with.** The scheduler keeps its original epoch count; demo mode rewinds it at the switch.

4. **Early stopping:** remove the early-stopping `break` (or raise its patience above `2 * E`). Explain why: stopping during phase 1 ends the run before the dendrite turns on, and the short dip right after the learning rate restarts can trigger it during phase 2.

5. **Leave everything else alone:** optimizer creation, scheduler creation and every `scheduler.step()` call stay exactly as they are.

Example, PyTorch's MNIST example with `E = 14` (see `examples/base_examples/mnist/mnist_perforatedai_demo.py`):

```python
model = Net().to(device)
model = UPA.perforate_model(model, demo_mode=True,
                            demo_phase_steps=args.epochs * len(train_loader),
                            demo_sample_input=next(iter(train_loader))[0])
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
for epoch in range(1, 2 * args.epochs + 1):
    train(args, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()
```

## Step 5: Tell the user what to expect

- `perforate_model` prints which layer got the dendrite and how long phase 1 is. It also prints a list of "not wrapped" parameters: those are the other layers, which demo mode tracks on purpose.
- After `E` epochs, a "PAI demo: phase 1 finished" message appears. Compare the validation score just before that message with the score at the end of training.
- Phase 2 is also extra training, so for an honest comparison offer a control run: add `doing_pai=False` to the same `perforate_model` call. It runs the identical phases and learning rate restart without a dendrite. If the dendrite run beats the control run, the dendrite helped.
- Demo mode switches after a fixed number of steps, not on validation scores. If the model is already overfitting by the end of phase 1, the dendrite adds capacity and can make validation worse.

## Step 6: After the demo

Demo mode is a one-off trial, not a starting point for the full integration. When the user is done:

1. Revert the demo edits so the script is back to its original code: remove the import and the `perforate_model(..., demo_mode=True, ...)` line, restore the original epoch count, and restore early stopping. Offer to do this for them.
2. For the full integration, where PAI decides from validation scores when to add dendrites and keeps adding them while they help, start the **perforatedai** skill ("Perforate my model") on the original script.
