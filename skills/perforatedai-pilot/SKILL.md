---
name: perforatedai-pilot
description: "Run a PerforatedAI pilot: a simple comparison of the user's model exactly as it stands today against the same model with one dendrite added, using UPA.perforate_model(demo_mode=True). No tuning or other optimization of the model or training script. Use for pilots, trials, evaluations, proofs of concept and quick demos. Triggers: 'Run a perforatedai pilot', 'Pilot perforatedai on my model', 'Compare my baseline against perforatedai', 'Run the perforatedai demo', 'Try dendrites on my model quickly'. For a full integration with validation-driven dendrite growth and tuning, use the perforatedai skill ('Perforate my model') instead."
---

# PerforatedAI Pilot

A pilot answers one question: **what does adding a dendrite do to this model, as it stands today?** The result is two validation scores from the same training script, one without the dendrite and one with it.

The pilot uses demo mode, which gets both scores from a single run. The script first trains as usual with the dendrite frozen (phase 1, the baseline). Then the dendrite turns on, and the user's optimizer and learning rate schedule restart at 0.25x their starting learning rate (phase 2), replaying the same schedule. The score at the end of phase 1 is the baseline; the score at the end of training is baseline plus PerforatedAI.

**Your role is to analyze and edit the user's script. Never run their training script yourself; ask them to run it and share the output.**

## Pilot rules

The comparison only means something if the dendrite is the single difference between the two scores. It is also only useful if it happens soon: a pilot that waits behind a list of other improvements gets pushed back, and by the time it runs the baseline has moved and the effect is harder to see. So for the length of the pilot:

- **Change only what Step 4 lists.** The model, its size, the data pipeline, augmentation, loss, optimizer, learning rate, schedule, batch size, regularization, precision and seeds all stay exactly as the user has them.
- **Do not suggest improving the baseline first.** Take the script as the baseline even if you would have written it differently or think it is under-tuned. Do not propose hyperparameter sweeps, a smaller or different architecture, refactors or cleanup as a step before or during the pilot.
- **Do not tune PerforatedAI either.** Demo mode picks the layer and the settings. No `GPA.pc.set_*` calls, no choosing a different layer, no extra dendrites.
- **Keep a list instead.** If you notice something that could be improved, write it down and give it to the user under "After the pilot" in the Step 6 report. That is the only place it appears.
- **Fix only what stops the pilot from running.** If a pilot edit causes an error, fix that error with the smallest change that works and say what you changed. An error the script already had without the pilot edits is the user's to fix: tell them, and do not combine the fix with other changes.
- **If the user asks for another change mid-pilot,** tell them in one sentence that it changes the baseline, so both scores would need to be measured again, and suggest doing it after the pilot. It is their script and their call.

## Step 1: Install

Ask the user to run:

```bash
pip install perforatedai
```

`perforatedbp` is not needed. If it is installed, demo mode turns Perforated Backpropagation off for the run.

Demo mode needs perforatedai 3.3.0 or newer. If `perforate_model` raises `unexpected keyword argument 'demo_mode'`, the installed version is older: ask the user to upgrade (`pip install -U perforatedai`) and check `pip show perforatedai`.

## Step 2: Check the script is a good fit

Ask for the training script path and the exact command or config file the user runs it with, and read both. Find the validation score it reports each epoch; that is the number the pilot compares. It may be printed, or only logged (`self.log("val_loss", ...)` in Lightning, a TensorBoard or W&B scalar); note where it ends up, because Step 5 has to tell the user where to read it. If there are several, ask the user which one they judge the model by. If there is none, ask them how they measure the model today before going further. Note whether higher or lower is better: the score is often a loss.

Demo mode supports a single-process PyTorch training loop the user wrote themselves, and a single-device PyTorch Lightning `Trainer` (see "PyTorch Lightning scripts" below). It does not support a script that uses any of these:

- HuggingFace `Trainer` or a subclass of it. It builds its scheduler on a wrapped optimizer, so demo mode turns the dendrite on but does not restart the learning rate schedule, and prints no warning. Its epoch count also sets the schedule length, so doubling the epochs changes phase 1.
- Another library that runs the training loop, other than PyTorch Lightning
- `DataParallel`, `DistributedDataParallel`, FSDP, or multiple processes (for example a `torchrun --nproc_per_node=N` launch with N above 1, which is how the torchvision reference recipes are usually run). A Lightning `Trainer` without `devices=1` uses every visible GPU with DDP, and a HuggingFace `Trainer` uses `DataParallel`, so for those scripts ask how many GPUs the machine shows.
- More than one optimizer for the model (for example separate backbone and head optimizers)
- A Muon optimizer
- EMA or averaged copies of the model (`torch.optim.swa_utils.AveragedModel`, timm `ModelEma`, a hand-written EMA)
- Code that sets `param_group["lr"]` by hand every step instead of using a scheduler
- Mixed precision with a `GradScaler` (`scaler.step(optimizer)`) together with a `OneCycleLR` scheduler stepped every batch. Demo mode counts optimizer steps, and the scaler skips a few of them, so `OneCycleLR` reaches its last step before the switch and raises `Tried to step N times`. A `GradScaler` with any other scheduler, or with `OneCycleLR` stepped once per epoch, is fine.
- A model that is not called with a single tensor (`output = model(inputs)`) or does not return a single tensor (for example dict inputs, several arguments, or HuggingFace output objects)

Judge these by the run the user actually does, not by what the script can do. A feature behind a flag or config entry that their command leaves off (for example `--model-ema`, `--distributed`, or a multi-GPU branch that only runs with more than one GPU) does not count. If you cannot tell from the command or config, ask.

Tell the user what you checked and whether you found any of these. If you found one, do not rewrite their script to remove it; go to "If demo mode does not fit" at the end of this skill.

Two more things to look for:

- **PerforatedAI code already in the script** (`GPA.pc.set_*` calls, `perforate_model`, `setup_optimizer`, `add_validation_score`, or a model loaded with dendrites already in it). The pilot's baseline is a model with no dendrites, so it runs on the version of the script from before that code was added. Ask the user for that version, or for the flag that selects the plain model if the script keeps both paths. If they want to work on the integration they already have, that is the **perforatedai** skill, not a pilot.
- **More than one model trained in one run** (a comparison loop, folds, seeds, a teacher and a student). Ask which model the pilot is for, and add the PAI line for that model only. A separate `perforate_model(..., demo_mode=True, ...)` call for each training run in the same process works; each call starts a new pilot.

## Step 3: Ask how long training actually runs

**Ask:** "How many epochs does your training actually run? If you use early stopping, about how many epochs does it usually run before it stops?"

Call the answer `E`. It must not be larger than the epoch count the scheduler is built with (`OneCycleLR` raises an error if it is stepped past its end).

Also ask whether they have a validation score from a recent run of this exact script. It is not required, but it gives Step 6 something to check phase 1 against. Ask how long that run takes too, since the pilot takes twice as long. For long runs (ImageNet-scale training can take days), Step 5 has to warn that a pilot run cannot be resumed.

Then find in the script:

- The optimizer steps per epoch: usually `len(train_loader)`. If the script only calls `optimizer.step()` every `k` batches (gradient accumulation), it is `len(train_loader) // k`.
- The epoch variable (for example `args.epochs`) and every place it is used: the loop's `range(...)` and scheduler arguments such as `T_max`, `total_steps`, `epochs=`, `total_iters` or `num_training_steps`.
- Which part of each training batch the loop passes to the model. For `for data, target in train_loader: output = model(data)` it is the first item, so one batch of inputs is `next(iter(train_loader))[0]`. For a loader that yields dicts (`images = batch['image']`), it is `next(iter(train_loader))['image']`.

## Step 4: Make the edits

These are the only edits the pilot makes.

1. **Import** at the top of the script:

   ```python
   from perforatedai import utils_perforatedai as UPA
   ```

2. **The one PAI line**, after the model and the training data loader are created (after `.to(device)` is fine) and **before the optimizer and scheduler are created**:

   ```python
   model = UPA.perforate_model(model, demo_mode=True, demo_phase_steps=E * len(train_loader),
                               demo_sample_input=next(iter(train_loader))[0])
   ```

   It also goes before any code that collects the model's parameters for the optimizer, such as parameter groups for weight decay (`parameters = utils.set_weight_decay(model, ...)` in the torchvision reference scripts). A parameter list built before this line does not contain the dendrite, so the dendrite never trains; demo mode then prints "this optimizer does not include the dendrite parameters". The fix is to move the line above that code, not to change how the optimizer is built.

   Put the user's actual `E` in (as a number or a new argument). `demo_phase_steps` counts optimizer steps, so this is the length of their actual training run. `demo_sample_input` is one batch of exactly what the loop passes to `model(...)`; it is run through the model once to create the dendrite. If the data loader is created after the model, move this line below the loader. If the model is created in one function and the loader and optimizer in another (for example a `train_model(model, ...)` function), put the line in the second one, after the loader and before the optimizer.

   Demo mode chooses the layer itself: the last `nn.Linear` or `nn.Conv2d` layer, or the one before it if the last one holds 25% or more of the model's parameters.

3. **Training loop:** run `2 * E` epochs with a count of its own, for example `for epoch in range(1, 2 * E + 1):`, or `range(args.start_epoch, 2 * args.epochs)` for a loop that starts at `args.start_epoch`. **Do not change the epoch variable the scheduler is built with.** The scheduler keeps its original epoch count; demo mode rewinds it at the switch.

4. **Early stopping:** remove the early-stopping `break` (or raise its patience above `2 * E`). Explain why: stopping during phase 1 ends the run before the dendrite turns on, and the short dip right after the learning rate restarts can trigger it during phase 2.

5. **Code that reads every parameter's gradient:** during phase 1 the dendrite's parameters are frozen, so their `.grad` is `None`. If the script reads `param.grad` for every parameter without checking (for example gradient histograms or gradient-norm logging over `model.named_parameters()`), add an `is not None` check there. Gradient clipping with `clip_grad_norm_` already skips them and needs no change.

6. **Leave everything else alone:** optimizer creation, scheduler creation and every `scheduler.step()` call stay exactly as they are. There is no `setup_optimizer`, `set_optimizer_instance` or `add_validation_score` call in a pilot.

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

### PyTorch Lightning scripts

Lightning creates the optimizer inside `trainer.fit`, which is after the PAI line, so demo mode works without touching `configure_optimizers`. The same steps apply, with these differences:

- **Does not fit** (use "If demo mode does not fit"): `automatic_optimization = False`, `configure_optimizers` returning more than one optimizer, more than one device, or a `StochasticWeightAveraging` or EMA callback.
- **The PAI line** goes on the `LightningModule`, after it and the training loader are created and before `trainer.fit`. `demo_sample_input` is one batch of what `training_step` passes to `self(...)`. Do not wrap the network in an inner module or change `configure_optimizers`; `examples/library_examples/pytorch_lightning` shows the full integration, not a pilot.
- **Steps per epoch** are `len(train_loader) // accumulate_grad_batches`, and fewer if `limit_train_batches` is set.
- **Epochs:** set `max_epochs=2 * E` on the `Trainer`. If `configure_optimizers` builds the scheduler from `self.trainer.max_epochs`, `self.trainer.estimated_stepping_batches` or `max_steps`, put the original number there so the schedule keeps its length.
- **Early stopping** is the `EarlyStopping` callback: take it out of `callbacks=[...]`.
- **Reading the score:** Lightning numbers epochs from 0, so the baseline is the value logged for epoch `E - 1` and the final score is epoch `2 * E - 1`. If the script only logs the score, ask the user for those two values from their logger (for example `lightning_logs/version_N/metrics.csv`). The "phase 1 finished" message can appear out of order with the progress bar, so go by the epoch number.
- **GPU:** this path was checked on CPU only. If training runs on a GPU, move the module there before the PAI line (`model = model.to("cuda")`) so the dendrite is created on the training device.

## Step 5: Ask the user to run it

Ask the user to run the script and share the full output. Tell them what they will see:

- `perforate_model` prints which layer got the dendrite, how many parameters that layer has out of the model's total, and how long phase 1 is. It also prints a list of "not wrapped" parameters: those are the other layers, which demo mode tracks on purpose.
- After `E` epochs, a "PAI demo: phase 1 finished" message appears. The validation score just before that message is the baseline. The score at the end of training is the baseline plus the dendrite. If the script numbers epochs from 0 (the torchvision reference scripts print `Epoch: [0]`), the baseline is the evaluation after epoch `E - 1` and the final score is the one after epoch `2 * E - 1`. With mixed precision (`GradScaler`) the message can appear a few batches into the next epoch, because steps the scaler skips are not counted. The baseline is still the last validation score before it.
- The run takes twice as long as their usual training run, and writes a `PAI_demo/` folder in the working directory.
- Anything the script prints about the whole run covers both phases: a "best score" summary at the end is the best of either phase, and a progress label such as "Epoch 3/2" counts past the original total. The baseline is only the score at the phase 1 message.

Before they run it, check where the script saves checkpoints. The pilot run saves to the same paths, and those weights include the dendrite, so the original model class cannot load them with `load_state_dict`. If files already exist at those paths that the user wants to keep, tell them to move them first. For the same reason, code after training that loads a checkpoint back into the original class (`Model.load_from_checkpoint(best_path)`, `trainer.test(ckpt_path="best")`, `model.load_state_dict(torch.load(...))`) raises an error in the pilot run. Training has finished by then and both scores are valid; tell the user before they run it so the traceback at the end is expected. A "best" checkpoint is also the best of either phase. A script that saves a checkpoint every epoch (`model_{epoch}.pth`) writes twice as many files, so check there is disk space.

A pilot run cannot be resumed. Demo mode's step count and phase are not saved in the checkpoint, so a run restarted from one (`--resume checkpoint.pth`) starts phase 1 again with the dendrite frozen. A checkpoint from the original script does not load into the pilot model either. If the command already has a resume flag, take it off for the pilot. For a long run, tell the user that an interrupted pilot has to start again from the beginning.

Phase 2 is also extra training at a lower learning rate, which can raise the score by itself. To separate that from the dendrite, offer a control run: add `doing_pai=False` to the same `perforate_model` call and run again. It runs the identical phases and learning rate restart without a dendrite. This is part of the comparison, not tuning.

## Step 6: Report the comparison

When the user shares the output, report the numbers and stop. Use this shape:

| | Validation score |
| --- | --- |
| Baseline (end of phase 1) | |
| With dendrite (end of training) | |
| Control, no dendrite (end of training), if run | |

Under the table, give:

- The layer that got the dendrite and the parameters it added, from the lines `perforate_model` printed.
- One or two sentences on what the numbers show. Say whether higher or lower is better when the score is a loss. If the control was run, the dendrite's effect is the dendrite run against the control. If it was not, say that the difference includes the extra training in phase 2.
- **After the pilot:** the list you kept under the pilot rules, if there is anything on it. One line per item, no edits.

Things to check before reading meaning into the result:

- If the user gave a score from an earlier run of the original script, the end of phase 1 should be close to it. If it is far off, an edit changed their training; find which one before reporting.
- Look at how much the validation score moves from one evaluation to the next late in each phase. If the difference between the rows is no bigger than that movement, say the result is within the run's noise and that one run cannot separate them.
- Demo mode picks the layer near the output, which can be very small (a final 1x1 convolution in a segmentation model can be a few dozen parameters). If the dendrite is a tiny share of the model, say so next to the result: a small difference is what a dendrite that size can show. Do not move it to another layer; that is tuning, and it belongs to the full integration.
- Demo mode switches after a fixed number of steps, not on validation scores. If the model is already overfitting by the end of phase 1, the dendrite adds capacity and can make validation worse. Say so if the output shows it.

Report the result as it came out, including when the dendrite did not help. Do not start adjusting settings to get a better number: the pilot is one dendrite on one layer with default settings, and the full integration is where growth and tuning happen.

## After the pilot

The pilot edits are a one-off trial, not a starting point for the full integration. When the user is done:

1. Revert the pilot edits so the script is back to its original code: remove the import and the `perforate_model(..., demo_mode=True, ...)` line, restore the original epoch count, restore early stopping, and delete the `PAI_demo/` folder. Offer to do this for them.
2. For the full integration, where PAI decides from validation scores when to add dendrites and keeps adding them while they help, start the **perforatedai** skill ("Perforate my model") on the original script. Other optimizations, including the items from your "After the pilot" list, belong there.

## If demo mode does not fit

When Step 2 finds something demo mode does not support, the pilot is still the same comparison under the same pilot rules. It takes two runs and uses the standard integration:

1. **Baseline run.** Ask the user to run their script unchanged and share the final validation score, or to give you that score from a recent run of this exact script. If the integration needs a different build of a library (HuggingFace `Trainer` needs `transformers-perforated` in place of `transformers`), have them install it first and do the baseline run in that environment, so both runs use the same library. A score from before the switch is not the baseline.
2. **Add PerforatedAI.** Load the sibling skill **perforatedai** and follow its "Perforate my model" setup, with these differences:
   - Skip its "Ask About Optimization Goal" step. The goal is the user's existing validation metric, and the model stays the architecture and size it is now. Do not move to a smaller or different model.
   - Skip its "Optional Configuration Tuning" step. Wherever a step offers a choice of settings, keep the default.
   - The library and multi-GPU skills it loads (for example for HuggingFace `Trainer` or DDP) apply as written. The `GPA.pc.set_*` lines those steps prescribe are part of the integration, not tuning: add them with the values given.
   - PAI watches the score the user already reports, in its own direction: `"eval_loss"` with `maximizing_score=False` if the script only reports a loss. Do not add an accuracy metric, `compute_metrics`, or a change to the model's outputs or loss to get a different score. The scripts under `examples/library_examples` make changes like these and also replace the scheduler; do not copy them into a pilot.
3. **Perforated run.** After the integration passes that skill's verification step, ask the user to run full training and share the output.
4. **Report** as in Step 6, with the baseline run and the perforated run as the two rows. The standard integration keeps adding dendrites while they help, so this run trains for longer than the baseline; say that in the report.
