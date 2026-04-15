---
name: perforatedai
description: "Expert in PerforatedAI library for adding artificial dendrites to PyTorch neural networks. Triggers: 'Perforate my model' (start interactive setup), 'debug my perforated model' (debug/optimize existing integration). Also use when: debugging dendrite training, configuring PAI settings, troubleshooting dendrite issues, working with PAINeuronModule or PAIDendriteModule. For analyzing completed training results, use the perforatedai-analyze skill."
---

# PerforatedAI Dendrite Network Skill

## Available Resources

This skill has access to:
- **API Documentation**: [api-docs/](./api-docs/) - All API guides and references
- **Source Code**: [source-code/](./source-code/) - Complete PerforatedAI library implementation
- **Examples**: [examples/](./examples/) - Working code examples for various architectures

Feel free to reference these when helping users debug or understand implementation details.

## Entry Points

### 1. "Perforate my model" - Interactive Setup

When the user says **"Perforate my model"**, start the interactive setup process.

**IMPORTANT: Your role is to analyze code and make edits. Never run the user's training script - only ask them to run it and provide output.**

**First, check if PAI is already partially integrated:**

Ask for their training script path, then read it and check for:
- PAI imports: `from perforatedai import globals_perforatedai as GPA` and `utils_perforatedai as UPA`
- Configuration calls: `GPA.pc.set_*` functions
- Model initialization: `UPA.initialize_pai()` call
- Optimizer setup: `GPA.pai_tracker.setup_optimizer()` or `set_optimizer_instance()`
- Training loop: `GPA.pai_tracker.add_validation_score()` call

**If PAI integration is already present:**

Tell them: "I see you already have some PerforatedAI integration! Let me check what's complete and what's missing..."

Analyze what's been done and report:
- ✅ Completed steps (e.g., "Imports added", "Configuration set", "Model initialized")
- ❌ Missing steps (e.g., "Optimizer setup missing", "Training loop not updated")

Then ask: "Would you like me to:
1. Complete the missing integration steps
2. Debug/optimize your existing setup (say 'Debug my perforated model')
3. Start fresh with a different approach"

Based on their choice:
- **Option 1**: Continue from the first incomplete step
- **Option 2**: Jump to the "Debug My Perforated Model" workflow below
- **Option 3**: Confirm they want to replace existing code, then start from Step 1

**If no PAI integration found:**

Proceed with Step 1 below.

### Step 1: Discovery

#### 1.1 Get Training Script and Analyze Model

**Ask:** "What's the path to your training script?"

- If they provide a path: Read the script and analyze it to determine:
  - Model architecture type (CNN, Transformer, ResNet, Custom, etc.)
  - Model version/size (e.g., ResNet18 vs ResNet50, GPT2-small vs GPT2-large)
  - Input dimensions and data format
  - Training loop structure
  - Optimization metric being used
  - Whether the script has configurable model selection (command-line arguments for model, architecture parameters, etc.)
  
- If they say they don't have a script yet: Tell them:
  > "PerforatedAI is an optimization tool for existing models. Please build an initial training setup first and get a baseline working before integrating dendrites. Once you have a working training script, come back and say 'Perforate my model' to add dendritic optimization."
  
  Then stop - do not proceed with integration.

#### 1.2 Ask About Optimization Goal

**Ask:** "What are you optimizing for?"

Options:
- **Accuracy / Loss / Decision Making** - Improve model performance metrics
- **Efficiency / Model Size** - Reduce parameters while maintaining performance

Based on their answer:

**If Accuracy/Loss/Decision Making:**
- Proceed to Step 2 with standard dendrite growth settings (maximize metric or minimize loss)

**If Efficiency/Model Size:**
- **Educate them first:** "Important: Dendrites ADD parameters to your model, but they do so more efficiently than traditional approaches. To optimize for efficiency with PerforatedAI, you should start with a smaller base model and then add dendrites strategically.  This will allow you to achieve better performance than a larger model with fewer parameters. Let's work together to find the right balance for your use case."

- **Based on your earlier analysis of their training script, determine if model selection is configurable:**
  - Does it have command-line arguments that specify model architecture, variant, or size (e.g., `--model`, `--arch`, `--model_name`, `--resnet_version`, `--width`, `--depth`, `--num_heads`, `--num_layers`, `--hidden_dim`)?
  - Does it have config files that specify model architecture or size parameters?
  - Or is the model hardcoded in the script?

- **Then proceed based on what you find:**

  **If model IS configurable (has --model, --arch, or config arguments):**
  - **ASK what they're currently using, tailored to what's configurable:**
    - If they have `--model` or `--arch` argument: "I see your script has a `--model` argument. What model are you currently using?"
    - If they have architecture size parameters (e.g., `--width`, `--depth`, `--num_heads`, `--num_layers`): "I see your script supports configurable architecture. What are your current settings? (e.g., width, depth, number of heads/layers)"
    - If they have both model selection AND size parameters: "I see your script supports different configurations. What model and settings are you currently using? (e.g., model name, width, depth, number of layers/heads)"
  - Wait for their answer before proceeding
  
  **If model is NOT configurable (hardcoded in script):**
  - **Analyze the code** to identify what model they're using
  - Tell them: "I can see you're using [ModelName] in your script."
  - Proceed directly based on what you found (no need to ask)

- **Then determine the path based on their model (🚨 CRITICAL: For efficiency, the model MUST be smaller - always recommend downsizing):**

  **For Pretrained Models (e.g., ResNet50, BERT-base, GPT2-medium):**
  
  **If they're using a larger variant within a model family (ResNet50, ResNet34, BERT-base, GPT2-medium, etc.):**
  - Tell them: "You're using [ModelName]. To optimize for efficiency, I have two options for you.
    
    Important: Dendritic optimization often allows a smaller architecture to achieve the accuracy goals you would have previously needed the larger model for. The dendrites add targeted capacity exactly where needed, making efficient architectures surprisingly powerful.
    
    **Option 1: Smaller variant in the same family**
    - ResNet50 → ResNet18 + dendrites (or ResNet34 as intermediate step)
    - ResNet34 → ResNet18 + dendrites
    - BERT-base → BERT-small or DistilBERT + dendrites
    - GPT2-medium → GPT2-small + dendrites
    
    **Option 2: Switch to a more efficient architecture type**
    - ResNet50/34 → MobileNetV2 or EfficientNet-B0 + dendrites
    - BERT-base → DistilBERT + dendrites
    - ViT-Base → MobileViT or EfficientNet + dendrites
    
    Which would you like to try?"
  
  - **Wait for their choice before making any changes**
  
  - **After they choose**, make the appropriate changes:
    - **If model was configurable via arguments:** Modify their script's default argument or tell them which argument to change
    - **If model was hardcoded:** Modify their script to load the chosen model
  
  - **After making changes, tell them:**
    "I've updated your script to use [ChosenModel]. Before we add dendrites, please run a quick training test to confirm it loads correctly and trains without errors. This smaller model will have lower accuracy initially, which is expected. Dendrites will help recover performance."
  
  - Wait for confirmation before proceeding to Step 2
  
  **If they're already using the smallest variant within their model family (ResNet18, BERT-small, GPT2-small, etc.):**
  - Tell them: "You're using [CurrentModel], which is the smallest in its family. For maximum efficiency, I recommend switching to a fundamentally more efficient architecture type and adding dendrites.
    
    Important: Dendritic optimization often allows a smaller, more efficient architecture to achieve the accuracy goals you would have previously needed a larger model for. The dendrites add targeted capacity exactly where needed.
    
    Recommended switches:
    - ResNet18 → MobileNetV2 or EfficientNet-B0 + dendrites
    - BERT-small → DistilBERT + dendrites
    - ViT-Small → MobileViT or EfficientNet + dendrites
    
    Would you like to make this switch?"
  
  - **Wait for their decision before making any changes**
  
  - **If they agree**, make the appropriate changes:
    - **If model was configurable via arguments:** Modify their script's default argument or tell them which argument to change
    - **If model was hardcoded:** Modify their script to load the more efficient architecture
  
  - **After making changes, tell them:**
    "I've updated your script to use [MoreEfficientArchitecture]. Before we add dendrites, please run a quick training test to confirm it loads correctly and trains without errors. This architecture is designed for efficiency and will have lower accuracy initially. Dendrites will help recover performance."
  
  - Wait for confirmation before proceeding to Step 2
  
  **For Custom Models:**
  - **First, check if configuration is via arguments or hardcoded:**
    - Look for command-line arguments like `--num_layers`, `--hidden_dim`, `--width`, `--depth`, `--num_heads`, `--embed_dim`, etc.
    - Look for config file options for these settings
    
  - **If configurable via arguments/config:**
    - **ASK about their current configuration:** "What are your current model settings? For example, how many layers are you using? What are the hidden dimensions or channel counts?"
    - Wait for their answer
    
  - **If hardcoded:**
    - Analyze their model architecture to identify:
      - Number of layers (depth)
      - Hidden dimensions/channels (width)
    - Tell them: "I can see your model has [X] layers with hidden dimension [Y]."
  
  - **After understanding their current settings**, tell them: "To optimize for efficiency, let's make your model smaller first, then add dendrites.
    
    Important: Dendritic optimization often allows a smaller architecture to achieve the accuracy goals you would have previously needed a larger model for. The dendrites add targeted capacity exactly where needed, making smaller models surprisingly powerful.
    
    We can:
    - Reduce layer count (make it shallower)
    - Reduce hidden dimensions/channels (make it less wide)
    - Both"
  
  - Ask: "Would you like to reduce depth (fewer layers), width (fewer channels/neurons), or both? And what values would you like to use?"
  
  - **Wait for their decision on what to reduce and to what values**
  
  - **After they specify what they want**, implement the changes based on their code structure:
  
    **If they already have configurable width/depth parameters:**
    - Identify where these are set (config file, command-line args, hardcoded values)
    - Update them to the values they specified. For example:
      - `hidden_dim=512` → `hidden_dim=256`
      - `num_layers=12` → `num_layers=6`
    - Tell them what you changed
    
    **If they have command-line arguments but not for width/depth:**
    - Add new command-line arguments for the parameters they want to adjust
    - Example: Add `--hidden_dim`, `--num_layers`, `--num_channels`, etc.
    - Set defaults to the values they specified
    - Update their model instantiation to use these new arguments
    - Tell them what you changed
    
    **If they don't have command-line arguments:**
    - Add configurable settings at the top of their script or in a config section
    - Example:
      ```python
      # Model configuration (adjust these for efficiency)
      NUM_LAYERS = 6  # Original: 12
      HIDDEN_DIM = 256  # Original: 512
      NUM_CHANNELS = 32  # Original: 64
      ```
    - Update their model definition to use these variables instead of hardcoded values
    - Tell them what you changed
  
  - **Before proceeding to Step 2:**
    - Tell them: "I've updated your model to be smaller. Before we add dendrites, please run your training script now to confirm:
      1. Training runs without errors
      2. The model is indeed smaller (check parameter count)
      3. Accuracy is somewhat lower than your original model (expected)
      
      This establishes a baseline. After we add dendrites, we'll aim to recover or exceed your original accuracy with this more efficient architecture."
    
    - Wait for them to confirm they've tested the smaller model before proceeding to dendrite integration

### Step 2: Add Imports

Add these imports at the top of their training script:

```python
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA
```

These are the same for all model types. Make the change in their file.

### Step 3: Configure PAI Settings

Based on the analyzed model type, add the appropriate configuration **before** model initialization in their script:

#### For CNN/Vision Models

Add this configuration:

```python
# PAI Configuration for CNNs
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_module_names_to_convert(["Conv2d", "Linear"])
GPA.pc.set_output_dimensions([-1, 0, -1, -1])  # [batch, channels, height, width]
```

#### For Transformers/Sequence Models

Add this configuration:

```python
# PAI Configuration for Transformers
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_module_names_to_convert(["Linear"])
GPA.pc.set_output_dimensions([-1, -1, 0])  # [batch, sequence, features]
GPA.pc.set_module_ids_to_track([".output_projection"])  # Skip final layer
```

#### For ResNet Models

Add this configuration:

```python
# PAI Configuration for ResNet
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_module_names_to_convert(["BasicBlock", "Bottleneck", "Linear"])
GPA.pc.set_module_ids_to_track([".conv1", ".bn1"])
GPA.pc.set_output_dimensions([-1, 0, -1, -1])  # [batch, channels, height, width]
```

#### For Custom/Unknown Models

If the model doesn't match the above patterns, analyze it:

1. **Determine input dimensions:**
   - If images (4D tensors): `[-1, 0, -1, -1]` (batch, channels, height, width)
   - If sequences/text (3D tensors): `[-1, -1, 0]` (batch, sequence, features)

2. **Identify convertible layers:**
   - Look for `nn.Linear`, `nn.Conv2d`, `nn.Conv1d` in their model
   - Start with converting `Linear` layers only for safety
   - Can expand to Conv layers if needed

3. **Add configuration based on analysis:**
```python
# PAI Configuration - analyzed from your model
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
GPA.pc.set_max_dendrites(5)
GPA.pc.set_module_names_to_convert(["Linear"])  # Start conservative
GPA.pc.set_output_dimensions([...])  # Based on your tensor shape
# May need to skip certain layers - we'll see from debug output
```

Explain your reasoning for each choice based on what you saw in their model when you make the change.

**→ Next: Proceed to Step 4.**

---

### Step 4: Initialize Model

Find where their model is created. Before adding PAI initialization, **first analyze their validation loop** to determine if they're maximizing or minimizing a metric.

**Look for their validation code to identify:**
- Metrics like `accuracy`, `acc`, `f1`, `precision`, `recall`, `auc` → maximizing
- Metrics like `loss`, `error`, `mse`, `mae`, `rmse`, `cross_entropy` → minimizing
- Check if they're using `max()`, `min()`, or comparing with `best_acc`, `best_loss`, etc.

**Determine maximizing_score:**
- `maximizing_score=True` if tracking accuracy, F1, precision, or any "higher is better" metric
- `maximizing_score=False` if tracking loss, MSE, MAE, error, or any "lower is better" metric

**Then add PAI initialization:**

Find pattern like:
```python
model = YourModel(...)
model = model.to(device)
```

Change it to:
```python
model = YourModel(...)
model = UPA.initialize_pai(model, save_name="your_model_dendritic", maximizing_score=True)  # or False
model = model.to(device)
```

**Set maximizing_score based on what you found:**
- If they track `val_acc`, `accuracy`, `val_f1`, etc.: use `maximizing_score=True`
- If they track `val_loss`, `loss`, `val_mse`, `error`, etc.: use `maximizing_score=False`

Tell them: "I analyzed your validation loop and found you're tracking [metric_name]. I've set `maximizing_score=[True/False]` accordingly."

Make this change in their script.

**→ Next: Proceed to Step 5.**

---

### Step 5: Detect and Handle Multi-GPU Setup

🚨 **MANDATORY: Check for DataParallel or DistributedDataParallel BEFORE proceeding to optimizer setup.**

**Analyze their script to check if they're using DataParallel or DistributedDataParallel:**

**Search for these patterns:**
- `torch.nn.DataParallel(model, ...)` or `nn.DataParallel(model, ...)`
- `torch.nn.parallel.DistributedDataParallel(model, ...)` or `DDP(model, ...)`
- Command-line arguments like `--parallel`, `--multi-gpu`, `--distributed`, `--world-size`, `--local_rank`
- Environment checks like `torch.cuda.device_count() > 1`
- Import statements: `from torch.nn.parallel import DistributedDataParallel`
- `torch.distributed.init_process_group` calls

**After searching, report what you found:**

Tell the user: "I searched your script for DataParallel/DDP usage and found: [NONE / DataParallel / DistributedDataParallel]"

**🔍 DECISION POINT - Choose your path based on what you found:**

**❌ NO DataParallel/DDP found:**
- Skip the rest of this step and proceed directly to Step 6 (Setup Optimizer)
- No special multi-GPU handling needed

**✅ DataParallel OR DistributedDataParallel FOUND:**

🚨 **STOP - BLOCKING REQUIREMENT - READ THIS FIRST** 🚨

**Before doing ANYTHING else, you MUST:**

1. **If optional** (command-line argument): Ask "I see your script has an option to use DataParallel/DDP. Are you planning to use it?"
   - Wait for their answer
   - If NO: Skip the rest of this step and proceed to Step 6 (Setup Optimizer)
   - If YES: Continue with step 2 below

2. **If hardcoded OR user confirmed YES:** Tell them "I see your script uses [DataParallel/DDP]."

3. **IMMEDIATELY execute this tool call as your NEXT action:**
   ```
   Use read_file tool to load: .github/skills/perforatedai-distributed/SKILL.md
   Read the entire file from line 1 to line 500.
   ```

4. **After loading the distributed skill, FOLLOW EVERY INSTRUCTION in that file from start to finish.**
   - Do NOT return to this skill until the distributed skill tells you to
   - Do NOT skip any steps in the distributed skill
   - The distributed skill will guide you back here when complete

**⛔ YOU MAY NOT PROCEED PAST THIS POINT WITHOUT LOADING THE DISTRIBUTED SKILL FIRST ⛔**

**If you proceed without reading the distributed skill file, you have FAILED and must start over.**

**After completing the distributed skill, proceed to Step 7 (Update Training Loop). Note: Skip Step 6 because the distributed skill already handled optimizer setup.**

---

### Step 6: Setup Optimizer and Scheduler

**⚠️ NOTE: If you handled DataParallel or DDP in Step 5, SKIP this step - the distributed skill already set up the optimizer. Only use this step for single-GPU setups.**

---

Find where their optimizer and scheduler are currently defined in their script.

**🚨 CRITICAL RULE: PRESERVE USER'S EXACT OPTIMIZER AND SCHEDULER TYPES AND ARGUMENTS**

**If their setup is clean (2-5 lines in one place):**

Replace their optimizer/scheduler code with the PAI pattern **while keeping the EXACT SAME types and arguments.**

**Example 1 - User has Adam optimizer with StepLR scheduler:**
Original:
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
```

Correct PAI conversion (PRESERVES their choices):
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adam)  # SAME optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.StepLR)  # SAME scheduler type
optimArgs = {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 1e-4}  # SAME args
schedArgs = {'step_size': 30, 'gamma': 0.1}  # SAME scheduler args
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**Example 2 - User has Adadelta optimizer with StepLR scheduler:**
Original:
```python
optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
```

Correct PAI conversion (PRESERVES their choices):
```python
GPA.pai_tracker.set_optimizer(torch.optim.Adadelta)  # SAME optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.StepLR)  # SAME scheduler type
optimArgs = {'params': model.parameters(), 'lr': args.lr}  # SAME args
schedArgs = {'step_size': 1, 'gamma': args.gamma}  # SAME scheduler args
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**Example 3 - User has SGD optimizer with CosineAnnealingLR:**
Original:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
```

Correct PAI conversion (PRESERVES their choices):
```python
GPA.pai_tracker.set_optimizer(torch.optim.SGD)  # SAME optimizer type
GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR)  # SAME scheduler type
optimArgs = {'params': model.parameters(), 'lr': 0.1, 'momentum': 0.9, 'weight_decay': 5e-4}  # SAME args
schedArgs = {'T_max': 200}  # SAME scheduler args
optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**MANDATORY RULES:**
1. **DO NOT** change the optimizer type (keep Adam/SGD/AdamW/Adadelta/etc. exactly as user had)
2. **DO NOT** change the scheduler type (keep StepLR/CosineAnnealingLR/ExponentialLR/etc. exactly as user had)
3. **DO NOT** change optimizer arguments (preserve lr, weight_decay, momentum, betas, etc.)
4. **DO NOT** change scheduler arguments (preserve step_size, gamma, T_max, patience, etc.)
5. **DO** remove any `scheduler.step()` calls in their training loop - PAI handles this automatically
6. If user has NO scheduler, use `set_scheduler(None)` or omit the set_scheduler call

**If their setup is complex (scattered across functions, custom classes, framework-managed, etc.):**

Don't try to replace it. Instead, add this single line after their optimizer is fully created:

```python
# Their existing optimizer/scheduler setup stays unchanged
optimizer = ...  # Their code
scheduler = ...  # Their code (if they have one)

# Add only this line after optimizer creation
GPA.pai_tracker.set_optimizer_instance(optimizer)
```

Tell them: "Your optimizer setup is complex, so I'm using the simpler integration method. PAI will work with your existing optimizer configuration."

---

### Step 7: Update Training Loop

Find their validation step in the training loop. You need to update it to use PAI's `add_validation_score` function.

**Pattern 1 - If they used PAI optimizer setup (Step 6, first option):**

Find where validation completes and they have a validation score. Add the PAI score tracking and restructuring logic.

Look for something like:
```python
val_acc = validate(model, val_loader)
# or
val_loss = compute_loss(model, val_loader)
```

After this, add:
```python
# Add PAI score tracking
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)  # Pass actual value (val_acc or val_loss)
model = model.to(device)  # Re-apply device settings

if training_complete:
    print("PAI training complete!")
    break

elif restructured:
    # Model was restructured (dendrites added/incorporated)
    # Reinitialize optimizer with EXACT SAME settings from Step 5
    # Example: if Step 5 used Adadelta with StepLR:
    optimArgs = {'params': model.parameters(), 'lr': args.lr}  # EXACT SAME as Step 5
    schedArgs = {'step_size': 1, 'gamma': args.gamma}  # EXACT SAME as Step 5
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
```

**🚨 CRITICAL:** 
- Use the EXACT SAME optimArgs and schedArgs from Step 6
- If Step 6 used StepLR, use StepLR args here (step_size, gamma)
- If Step 6 used CosineAnnealingLR, use those args here (T_max, etc.)
- DO NOT change optimizer/scheduler types or arguments in the restructured block
- Just pass the actual validation value (val_acc or val_loss). PAI handles maximization/minimization internally.

**Pattern 2 - If they used `set_optimizer_instance` (Step 6, second option):**

Find where validation completes. Add the PAI score tracking and restructuring logic.

After validation, add:
```python
# Add PAI score tracking
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_score, model)  # Pass actual value
model = model.to(device)  # Re-apply device settings

if training_complete:
    print("PAI training complete!")
    break

elif restructured:
    # Model was restructured - reinitialize optimizer exactly as they had it
    # Copy their ENTIRE original optimizer setup (including any complex initialization)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Their exact setup
    # If they had scheduler, recreate it too
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30)  # If they had one
    # Then tell PAI about it
    GPA.pai_tracker.set_optimizer_instance(optimizer)
```

When `restructured=True`, recreate the optimizer exactly as it was originally defined - if their setup was complex with function calls or multiple steps, copy all of that.

**IMPORTANT: Modify their training loop to handle dynamic completion**

PAI automatically decides when training is complete (when dendrites stop improving validation score). Because of this, their training loop needs to be able to run indefinitely or for many epochs.

**Find their training loop:**
```python
for epoch in range(1, args.epochs + 1):
    # training code
```

**🚨 CRITICAL: Change it to this EXACT pattern (DO NOT use epoch = 0):**

```python
epoch = -1  # 🚨 MUST BE -1, NOT 0! (will increment to 0 at loop start)
while True:
    epoch += 1
    # training code
    # ... validation ...
    # PAI's training_complete will break the loop
```

**Why epoch = -1 and not 0:**
- The loop immediately increments `epoch += 1` BEFORE any training code
- Starting at -1 means the first epoch will be 0 (or 1 if they increment before use)
- This matches their original loop behavior where `range(1, args.epochs + 1)` starts at 1
- **NEVER use `epoch = 0` - this would make the first epoch 1, breaking compatibility**

Tell them: "I've modified your training loop to allow PAI to control when training ends via the `training_complete` flag. PAI will automatically break the loop when adding more dendrites no longer improves validation score."

**⚠️ WARNING:** If their code has epoch-dependent behavior (learning rate schedules, early stopping, etc.), make sure those still work correctly with the modified loop structure.

**After making all the code changes:** Tell them:
> "I've integrated PerforatedAI into your training script. Note: I set `set_testing_dendrite_capacity(True)` which is a debugging flag that helps verify dendrites are being added correctly. We'll change this to `False` for full training after confirming everything works."

### Step 8: Optional Configuration Tuning

Before running the first experiment, check with them about optional configurations that can improve results and analysis:

#### 7.1 Additional Score Tracking (Training and Test)

**Understanding the three dataset types:**
- **Training set**: Used for training the model (every epoch)
- **Validation set**: Used to track performance and make dendrite decisions (every epoch) - this is the MAIN metric
- **Test set**: Typically evaluated ONLY at the end for final results (optional)

**🚨 CRITICAL: The validation metric is ALREADY tracked by PAI via `add_validation_score()`. DO NOT duplicate it.**

**First, check if they have a test set:**

Look for a `test_loader`, `test_dataset`, or separate test evaluation function in their code. 

**If they have a test set:**

Ask: "I see you have a test set that's normally evaluated at the end. Would you like me to add:
1. Training score tracking every epoch (helps detect overfitting)
2. Test score tracking every epoch (lets you see test scores for EVERY architecture/dendrite count in `_best_arch_scores.csv`, not just your final architecture)"

**If they DON'T have a test set (only train/val):**

Ask: "Would you like me to add training score tracking every epoch? This helps me make better optimization recommendations by comparing training vs validation trends (e.g., detecting overfitting)."

**If yes, modify their training loop:**

**Pattern with all three datasets (train, validation, test):**

```python
# Each epoch
train_acc = train(model, train_loader)
val_acc = validate(model, val_loader)
test_acc = test(model, test_loader)  # Usually only done at end, but can track every epoch

# Track extra scores (training and test)
GPA.pai_tracker.add_extra_score(train_acc, "train")
GPA.pai_tracker.add_extra_score(test_acc, "test")

# Main validation metric for dendrite decisions (NOT duplicated in add_extra_score)
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
```

**Pattern with only two datasets (train, validation):**

```python
# Each epoch
train_acc = train(model, train_loader)
val_acc = validate(model, val_loader)

# Track training score only
GPA.pai_tracker.add_extra_score(train_acc, "train")

# Main validation metric for dendrite decisions
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
```

**Implementation: For training scores** - Track during the training loop (don't run a separate evaluate):
```python
# Inside the training loop, accumulate metrics
train_loss_total = 0
train_correct = 0
train_total = 0

for batch in train_loader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    # Accumulate stats during training
    train_loss_total += loss.item()
    train_correct += (outputs.argmax(1) == targets).sum().item()  # For classification
    train_total += targets.size(0)

# After training epoch, calculate training score
train_acc = train_correct / train_total  # Or use train_loss_total / len(train_loader) for loss
GPA.pai_tracker.add_extra_score(train_acc, "train")
```

**Implementation: For test scores** - Use their existing test/evaluate function:
```python
# After validation, evaluate on test set (if they have one)
test_acc = test(model, test_loader)  # Or evaluate(model, test_loader)
GPA.pai_tracker.add_extra_score(test_acc, "test")
```

**Advanced: If they track MULTIPLE metrics on the same dataset:**

For example, top-1 and top-5 accuracy:
```python
val_acc1 = evaluate_top1(model, val_loader)
val_acc5 = evaluate_top5(model, val_loader)

# Main metric for dendrite decisions
model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc1, model)

# Secondary metric for analysis only
GPA.pai_tracker.add_extra_score(val_acc5, "validation_top5")
```

Make these changes in their script.

#### 7.2 Convert Higher-Level Modules

Analyze their model architecture. If they have structured modules (like ResNet blocks, Transformer layers, etc.):

Ask: "I see your model has [BlockType] modules (e.g., BasicBlock, TransformerLayer). Converting at the block level along with individual layers can sometimes work better. Would you like to add block-level conversion?"

**Examples:**
- ResNet: Add `["BasicBlock", "Bottleneck"]` to existing `["Linear", "Conv2d"]`
- Transformers: Add `["TransformerEncoderLayer"]` to existing `["Linear"]`
- Custom architectures: Add their custom module classes

If yes, update the configuration:
```python
# Add to existing module conversions (don't replace)
GPA.pc.append_module_names_to_convert(["BasicBlock"])  # Or their block type
# This adds to the list - Linear and Conv2d layers inside BasicBlock won't be converted
# because BasicBlock will be converted first
```

Make this change and explain: "I've added [BlockType] to the conversion list. PAI will convert these modules first, so the layers inside them won't be separately converted. This captures higher-level feature patterns."

#### 7.3 Convert Only Top Layers

Ask: "For parameter efficiency, would you like to convert only the top (deeper) layers of your network? Top layers often benefit more from dendrites than early layers."

If yes, analyze their model structure and identify deeper layers to convert while skipping early ones.

**For sequential models:**
```python
# Skip early layers, only convert later ones
GPA.pc.append_module_ids_to_track([".layer1", ".layer2", ".conv1", ".conv2"])  # Skip these
# Keep layer3, layer4, fc for conversion
```

**For named architectures (ResNet, VGG, etc.):**
```python
# ResNet example: Only convert layer3, layer4 (skip layer1, layer2)
GPA.pc.append_module_ids_to_track([".layer1", ".layer2", ".conv1", ".bn1"])
```

Make the changes and explain: "I've configured PAI to only add dendrites to your top [N] layers. This focuses dendrite resources where they typically have the most impact."

**After optional configurations:**

Tell them: "Configuration complete! Now let's verify the integration works."

### Step 9: Verify Dendrite Integration

Tell them:

**"Now run your training script. PAI will automatically test dendrite integration for 7 epochs."**

**"Look for this success message from tracker_perforatedai:**
```
Successfully added 3 dendrites with GPA.pc.set_testing_dendrite_capacity(True) (default).
You may now set that to False and run a real experiment.
```

This message will appear automatically after 7 epochs if everything is working correctly."

**Wait for the user to run the script and report back. Do not run it yourself.**

**Once they see this success message:**

Say: "Great! The test completed successfully. I'll now switch to full training mode."

Find the line:
```python
GPA.pc.set_testing_dendrite_capacity(True)  # Debugging flag - start with True
```

Change it to:
```python
GPA.pc.set_testing_dendrite_capacity(False)  # Full training mode
```

Make this change in their script and tell them: "You can now run full training. PAI will add dendrites dynamically and manage the training process."

**If the success message does NOT appear:**

This means errors occurred during the test. Tell them: "The test didn't complete successfully. Let's debug the issue."

Switch to debugging mode:
- Check error messages in the console output
- Review the configuration (dimensions, module names, input shapes)
- Reference the [Debugging](#debugging-common-issues) section below for common issues
- Check the debugging documentation in [api-docs/debugging.md](./api-docs/debugging.md)

### Follow-up Support

After the initial setup and verification, you can now run full training.

**When your training is complete:**
- Say **"Analyze my perforated results"** to review outputs and get optimization recommendations (uses perforatedai-analyze skill)
- Say **"Debug my perforated model"** if you encounter issues during training

---

### 2. "Debug my perforated model" - Debug and Optimize

When the user says **"Debug my perforated model"**, help them debug or optimize an existing PerforatedAI integration.

---

**🚨 CRITICAL RULE: DO NOT RUN THE USER'S TRAINING SCRIPT 🚨**

**You are FORBIDDEN from:**
- Running their training script with `run_in_terminal`
- Running any Python scripts that train models
- Executing their code to "test" or "check for errors"

**You are ONLY allowed to:**
- Read their code files
- Analyze their code
- Make edits to their code
- Ask them to run the script and provide output/errors

**If you need to see errors:** ASK THE USER TO RUN THE SCRIPT and copy-paste the error to you.

---

**Step 1: Get their training script and the issue**

Ask: "What's the path to your training script, and what issue are you experiencing? If you're getting a crash or error, please copy-paste the full error message/traceback."

**Wait for their response. Do NOT run their script to find the issue yourself.**

Read the script and analyze the current PAI setup.

**Step 2: Check integration completeness**

Verify all required components are present:
- ✅ Imports (GPA, UPA)
- ✅ Configuration (set_testing_dendrite_capacity, set_max_dendrites, etc.)
- ✅ Model initialization (UPA.initialize_pai)
- ✅ Optimizer setup (setup_optimizer or set_optimizer_instance)
- ✅ Training loop (add_validation_score)

Report any missing components and offer to add them.

**Step 3: Enable debug mode if needed**

Check if `set_testing_dendrite_capacity` is currently set to `False` in their script.

**If set to False:**
- Ask: "I see you have `set_testing_dendrite_capacity(False)`. Does the crash/issue still occur when you set it to `True`?"

**Based on their answer:**
- **If they say "yes" or "not sure":** Change it to `True` in their script and tell them:
  > "I've temporarily set `testing_dendrite_capacity=True` for debugging. This runs a simplified 7-epoch test which helps isolate issues. We'll set it back to `False` after fixing the problem."
  
- **If they say "no, it works fine with True":** Tell them:
  > "The issue only occurs with `testing_dendrite_capacity=False`. This suggests the problem is related to full training mode. Let's debug with it set to `False` to reproduce the actual issue."
  > Keep it set to `False` for debugging.

**If already set to True:**
- Proceed with debugging - no changes needed

**Step 4: Analyze the issue**

**If the user hasn't provided the error/issue yet**, ask: "What specific issue are you experiencing? If you're getting a crash or error, please copy-paste the full error message/traceback."

**🚨 DO NOT run their script to find the error yourself. Wait for them to provide it. 🚨**

Once you have the error/issue description, analyze it:

Common scenarios:

**A. "Training errors / crashes"**
- Once you have the error traceback, analyze it for common issues:
  - Dimension mismatches in `set_output_dimensions()`
  - Wrong module names in `set_module_names_to_convert()`
  - Device placement issues (model not on correct device after restructuring)
  - Optimizer not reinitialized after restructuring
- Check [api-docs/debugging.md](./api-docs/debugging.md) for detailed debugging steps

**B. "Dendrites not being added"**
- Verify based on n_epochs_to_switch, improvement_threshold, and save_name_scores.csv that enough epochs have passed such that dendrites should have been added
- Check improvement_threshold - may be too strict
- Check epoch count that enough epochs have passed
- Verify validation scores are being passed to `add_validation_score()`

**C. "Dendrites added but performance worse"**
- Check if `maximizing_score` matches their metric (True for accuracy, False for loss)
- Verify they're passing the raw score value (not negated)

**Step 5: Make fixes**

Based on the identified issue, make the necessary code changes directly in their script.

**Step 6: Verification and restore settings**

After fixes, tell them what was changed and **ask them to run the training script**.

Say: "Please run your training script now and let me know:
- Does it run without errors?
- What output do you see?"

**Do not run the script yourself - wait for the user to run it and report back.**

**If you changed `testing_dendrite_capacity` to True in Step 3:**
- Tell them: "First, run a quick test with `testing_dendrite_capacity=True` to verify the fix works in debug mode."
- After they confirm it works, change it back to `False` and tell them:
  > "Great! I've set `testing_dendrite_capacity=False` back for full training. Run your training again to confirm everything works in production mode."

**If it was already True or you kept it False:**
- Just tell them what to look for when they run training with current settings

---

### 3. "Analyze my perforated results" - Review Training Outputs

**This functionality has been moved to a separate skill.**

When the user says **"Analyze my perforated results"**, they should use the **perforatedai-analyze** skill which provides:
- Comprehensive analysis of training CSV outputs
- Performance insights and dendrite impact assessment
- Optimization recommendations based on results
- Learning rate and scheduler tuning suggestions
- Module selection optimization

Tell them: "For analyzing your training results, please use the perforatedai-analyze skill. Just say 'Analyze my perforated results' and the analysis skill will be loaded automatically."

---

## Core Concepts

### What are Artificial Dendrites?

In biological neurons, dendrites perform computation before signals reach the cell body. PerforatedAI adds artificial dendrites to neural network layers, enabling:

- **Dynamic Architecture Growth**: Automatically adds dendrites where needed during training
- **Improved Accuracy**: Better feature representation through dendritic computation
- **Minimal Code Changes**: Wrap your existing PyTorch model with ~10 lines of code

### Architecture

- **PAINeuronModule**: Wrapper that converts a standard PyTorch module (Conv2d, Linear, etc.) into one that can have dendritic copies
- **PAIDendriteModule**: Container for all dendrite modules added to a neuron module
- **Dendrite-to-Neuron Weights**: Learned parameters controlling how dendrite outputs combine with main neuron output

---

**For detailed guidance:**
- Say **"Perforate my model"** to start the interactive setup process
- Say **"Debug my perforated model"** to debug or optimize an existing integration  
- Say **"Analyze my perforated results"** to review your training outputs and get optimization recommendations (uses perforatedai-analyze skill)
