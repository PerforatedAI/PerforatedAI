# Getting started with Weights and Biases

Weights and biases includes a great tool for doing hyperparameter sweeps.  This is strongly recommended to use when getting started with dendrites because the dendritic hyperparameters play an important role and with each new project there could be a high variation in optimal parameters.

## Initialization
Make sure you have already created an account and then add the import at the top of your file.

    import wandb

Note: wandb.login() is typically handled automatically or via environment variables. Manual login in code is optional.

## Set Up the Sweep
We typically recommend using the random sweep method and maximizing the Final Max Val metric (the best validation score achieved across all architectures).  But other sweep methods are available and minimizing loss is also an appropriate goal.

Define your sweep configuration with the hyperparameters you want to explore:

    sweep_config = {
        "method": "random",  # Random sampling
        "metric": {"name": "Final Max Val", "goal": "maximize"},
        "parameters": {
            # Standard hyperparameters
            "lr": {"values": [0.0001, 0.0003, 0.001, 0.003]},
            "weight_decay": {"values": [0.0, 1e-5, 1e-4, 1e-3]},
            "label_smoothing": {"values": [0.0, 0.05, 0.1]},
            # Dendritic hyperparameters
            "improvement_threshold": {"values": [[0.01, 0.001, 0.0001, 0], [0]]},
            "pai_forward_function": {"values": ["sigmoid", "relu", "tanh"]},
        }
    }

## Picking Hyperparameters
In addition to normal hyperparameters you may want to use, these are the ones most important for dendrites:

### Standard Training Hyperparameters
- **learning rate (lr)**: Start with lower values (0.0001-0.003) for transfer learning or fine-tuning
- **weight_decay**: Include 0.0 as an option; dendrites can sometimes perform better without weight decay
- **label_smoothing**: Often helpful for regularization (0.0-0.15)

### Dendritic Hyperparameters
- **improvement_threshold**: Speed of improvement required to prevent switching. For example, [0.01, 0.001, 0.0001, 0] means dendrites will switch after score stops improving by 1%, 0.1%, 0.01%, then 0% over recent history. [0] means no early switching (train until validation stops improving naturally).
- **pai_forward_function**: Forward function for dendrites - "sigmoid", "relu", or "tanh"

### Optional Parameters
- **cap_at_n**: If set to True, caps total dendrite training epochs to match neuron epochs. This is only used for Perforated Backpropagation training where they are separated. Defaults to False for best results, but True can cut epochs with slight performance trade-off.

    GPA.pc.set_cap_at_n(True)

## Getting the Run Setup

The modern approach uses a dedicated sweep function that WandB calls:

    def train_with_wandb():
        """Training function for WandB sweep - gets config from wandb.config."""
        import argparse
        
        parser = argparse.ArgumentParser(description="Training run within WandB sweep")
        parser.add_argument("--data-path", default="./data", type=str, help="Dataset path")
        parser.add_argument("--device", default="cuda", type=str, help="Device")
        args, unknown = parser.parse_known_args()  # Ignore unknown args from main script
        
        # Initialize wandb (project is inherited from sweep context)
        wandb.init()
        config = wandb.config
        
        # Override wandb sweep's silent mode (wandb.agent sets this to True)
        from perforatedai import globals_perforatedai as GPA
        GPA.pc.set_silent(False)
        
        # Set run name from config keys (for easier identification)
        # Start with run ID for uniqueness
        name_parts = [wandb.run.id]
        # Use all config keys (sorted for consistency)
        config_keys = sorted(config.keys())
        # Put any key containing 'model' first
        model_keys = [k for k in config_keys if 'model' in k.lower()]
        other_keys = [k for k in config_keys if 'model' not in k.lower()]
        config_keys = model_keys + other_keys
        name_parts.extend([f"{k}_{config[k]}" for k in config_keys])
        if name_parts:
            wandb.run.name = "_".join(name_parts)
        
        # Your training code here, using config values
        # ...
    
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="Train with WandB sweep support")
        parser.add_argument("--sweep-id", type=str, help="Join existing sweep by ID")
        parser.add_argument("--sweep-count", type=int, default=100, help="Number of runs")
        parser.add_argument("--wandb-project", type=str, required=True, help="WandB project name")
        parser.add_argument("--wandb-entity", type=str, default=None, help="WandB entity name")
        args = parser.parse_args()
        
        if args.sweep_id:
            # Join existing sweep
            wandb.agent(args.sweep_id, function=train_with_wandb, 
                       count=args.sweep_count, entity=args.wandb_entity, project=args.wandb_project)
        else:
            # Initialize new sweep
            sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_project)
            print(f"Sweep initialized: {sweep_id}")
            print(f"Project: {args.wandb_project}")
            entity_arg = f" --wandb-entity {args.wandb_entity}" if args.wandb_entity else ""
            print(f"To join from other machines: --sweep-id {sweep_id} --wandb-project {args.wandb_project}{entity_arg}")
            
            # Start agent on this machine
            wandb.agent(sweep_id, function=train_with_wandb, 
                       count=args.sweep_count, entity=args.wandb_entity, project=args.wandb_project)


## Using the config values

Access sweep parameters via `wandb.config`. Apply dendritic hyperparameters **inside your model loading function** before calling `perforate_model()`:

    def setup_model(model_name, num_classes, perforate=False):
        """Load model and configure PAI settings."""
        # Load your base model
        model = ...
        
        if perforate:
            # Configure PAI settings
            GPA.pc.set_modules_to_perforate([".fc", ".classifier"])  # Which layers to add dendrites to
            GPA.pc.set_output_dimensions([-1, 0])  # Output shape: [batch, features]
            
            # Apply dendritic hyperparameters from wandb config if in sweep
            if hasattr(wandb, "run") and wandb.run is not None and hasattr(wandb, "config"):
                if "improvement_threshold" in wandb.config:
                    GPA.pc.set_improvement_threshold(wandb.config.improvement_threshold)
                
                if "pai_forward_function" in wandb.config:
                    pai_fwd = wandb.config.pai_forward_function
                    if pai_fwd == "sigmoid":
                        GPA.pc.set_pai_forward_function(torch.sigmoid)
                    elif pai_fwd == "relu":
                        GPA.pc.set_pai_forward_function(torch.relu)
                    elif pai_fwd == "tanh":
                        GPA.pc.set_pai_forward_function(torch.tanh)
            
            # Build save_name from wandb config if available
            if hasattr(wandb, "run") and wandb.run is not None and hasattr(wandb.run, "name") and wandb.run.name:
                save_name = wandb.run.name  # Use wandb run name for consistency
            else:
                save_name = "default_run"
            
            # Perforate the model
            model = UPA.perforate_model(model, save_name=save_name, maximizing_score=True)
        
        return model
    
    # In train_with_wandb(), use config values for standard hyperparameters:
    def train_with_wandb():
        wandb.init()
        config = wandb.config
        
        lr = config.lr
        weight_decay = config.weight_decay
        label_smoothing = config.label_smoothing
        
        # Load model (dendritic configs applied inside load_model)
        model = setup_model(model_name, num_classes, perforate=True)
        
        # ... rest of your training code

### Retaining Perforated AI Logs

When calling perforate_model, set save_name to organize dendrite test results. 
For sweeps, use wandb.run.name directly to ensure save_name matches the WandB run name:

    # Use wandb run name as save_name for consistency
    if hasattr(wandb, "run") and wandb.run is not None and hasattr(wandb.run, "name") and wandb.run.name:
        save_name = wandb.run.name  # Ensures local folders match WandB run names
    else:
        save_name = "default_run"
    
    model = UPA.perforate_model(model, save_name=save_name, maximizing_score=True)

## Logging

Log training progress and metrics to WandB throughout training:

    # During training loop (each epoch)
    if hasattr(wandb, "run") and wandb.run is not None:
        wandb.log({
            "epoch": epoch + 1,
            "TrainAcc": train_acc,
            "TrainLoss": train_loss,
            "ValAcc": val_acc,
            "ValLoss": val_loss,
            "TestAcc": test_acc,  # Often same as ValAcc for transfer learning
            "Param Count": UPA.count_params(model),
            "Dendrite Count": GPA.pai_tracker.member_vars["num_dendrites_added"],
            "lr": optimizer.param_groups[0]["lr"],
        })

### Architecture Logging (Perforated Models)

For perforated models that dynamically add dendrites, track best performance at each architecture 
and the overall global best. The key insight: log arch scores when `num_dendrites_integrated` increases,
not when `num_dendrites_added` increases (since added dendrites may not be successfully integrated).

    # Outside the training loop - initialize tracking variables
    last_logged_integrated = -1  # Track last num_dendrites_integrated we logged
    max_val = 0   # Best validation for current architecture
    max_train = 0  # Training acc when max_val was achieved
    max_params = 0  # Params when max_val was achieved
    
    global_max_val = 0   # Best validation across all architectures
    global_max_train = 0  # Training acc when global_max_val was achieved
    global_max_params = 0  # Params when global_max_val was achieved
    
    # Inside the training loop (each epoch)
    
    # Track best for current architecture (IMPORTANT: only during neuron training, not dendrite training)
    current_mode = GPA.pai_tracker.member_vars.get("mode", "n")
    if current_mode == "n" and val_acc > max_val:
        max_val = val_acc
        max_train = train_acc
        max_params = UPA.count_params(model)
    
    # Track global best (only during neuron training)
    if current_mode == "n" and val_acc > global_max_val:
        global_max_val = val_acc
        global_max_train = train_acc
        global_max_params = UPA.count_params(model)
    
    # After add_validation_score
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(val_acc, model)
    model = model.to(device)
    
    # Log arch max when dendrites are successfully integrated
    if restructured and not training_complete:
        current_integrated = GPA.pai_tracker.member_vars["num_dendrites_integrated"]
        # Log if integrated count increased (new dendrites successfully integrated)
        if current_integrated > last_logged_integrated:
            if hasattr(wandb, "run") and wandb.run is not None:
                wandb.log({
                    "Arch Max Val": max_val,
                    "Arch Max Train": max_train,
                    "Arch Param Count": max_params,
                    "Arch Dendrite Count": current_integrated,
                })
            last_logged_integrated = current_integrated
            # Reset for next architecture
            max_val = 0
            max_train = 0
            max_params = 0
        
        # Reinitialize optimizer after restructuring
        optimArgs = {"params": model.parameters(), "lr": lr, 
                     "momentum": 0.9, "weight_decay": weight_decay}
        optimizer = GPA.pai_tracker.setup_optimizer(model, optimArgs)
    
    # When training completes
    if training_complete:
        # Log final arch if integrated count increased (max dendrites hit with successful last dendrite)
        current_integrated = GPA.pai_tracker.member_vars["num_dendrites_integrated"]
        if current_integrated > last_logged_integrated and hasattr(wandb, "run") and wandb.run is not None:
            wandb.log({
                "Arch Max Val": max_val,
                "Arch Max Train": max_train,
                "Arch Param Count": max_params,
                "Arch Dendrite Count": current_integrated,
            })
        
        # Always log Final Max scores
        if hasattr(wandb, "run") and wandb.run is not None:
            wandb.log({
                "Final Max Val": global_max_val,
                "Final Max Train": global_max_train,
                "Final Param Count": global_max_params,
                "Final Dendrite Count": GPA.pai_tracker.member_vars["num_dendrites_integrated"],
            })
        
        break  # Exit training loop

**Important Notes:**
- **Only track max values during neuron training (`mode == 'n'`)**: PAI alternates between training neurons (mode 'n') and training dendrites (mode 'd'). Only neuron training performance should count toward architecture maximums, as dendrite training is for correlation learning, not final performance.
- Use `num_dendrites_integrated` (successfully integrated dendrites) not `num_dendrites_added` (attempted dendrites) for arch logging
- Track `last_logged_integrated` to avoid duplicate logging
- Reset arch max trackers after logging each architecture
- Always log Final Max scores when training completes
- For non-perforated models, Arch Max = Final Max (no restructuring occurs)

### Final Metrics Logging

The logic for logging Final Max scores differs between perforated and non-perforated models:

**For perforated models**: Final metrics are logged inside the training loop when `training_complete=True`. Do NOT log them again in `train_with_wandb()` as this would duplicate or overwrite them.

**For non-perforated models**: Final metrics should be logged in `train_with_wandb()` after training completes, since there's no restructuring logic.

    # In train_with_wandb(), after training completes:
    best_acc1, best_epoch, model = train_single_run(args, train_loader, test_loader, num_classes)
    
    # Determine if model is perforated
    perforate = args.model in [model names that are perforated here]
    
    if not perforate:
        # Non-perforated models: log Final scores here
        wandb.log({
            "Final Max Val": best_acc1,
            "Final Max Train": best_acc1,
            "Final Param Count": UPA.count_params(model),
            "best_epoch": best_epoch,
        })
    # else: Perforated models already logged Final scores in training loop


## Running Sweeps

### Initialize a new sweep:
```bash
python your_script.py --wandb-project my-project --sweep-count 100
```

This will:
1. Create the sweep with your sweep configuration
2. Initialize the sweep on WandB with the specified project name
3. Print the sweep ID for joining from other machines
4. Start running sweep agents on this machine

### Join an existing sweep from another machine:
```bash
python your_script.py --sweep-id YOUR_SWEEP_ID --wandb-project PROJECT_NAME --sweep-count 100
```

Note: Always include `--wandb-project` to ensure metrics are logged to the correct project.

### View results:
Visit https://wandb.ai to see your sweep dashboard with all runs, metrics, and visualizations.

## Complete Example

See `Examples/imagenet_pretrained/train_from_hf_wandb_sweep.py` for a working implementation that includes:
- Model index mapping for better WandB visualization
- Sweep configuration with hyperparameter ranges
- Proper arch logging with `num_dendrites_integrated` tracking
- Silent mode override for verbose output during sweeps
- Transfer learning with pretrained models

## Optional: Parse Config String for Debugging

To re-run specific configurations from successful sweep runs:

    from types import SimpleNamespace

    def parse_config_string(name_str, config_keys):
        """Parse a config string back into a config object.
        
        Args:
            name_str: String like "model_resnet18_lr_0.001_weight_decay_1e-4"
            config_keys: List of expected keys in order
        
        Returns:
            SimpleNamespace with parsed config values
        """
        tokens = name_str.split("_")
        result = {}
        
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens):
                key = tokens[i]
                value = tokens[i + 1]
                if key in config_keys:
                    result[key] = value
                    i += 2
                else:
                    i += 1
            else:
                i += 1
        
        return SimpleNamespace(**result)
    
    # Usage
    config_keys = ["model", "dataset", "lr", "weight_decay", "label_smoothing"]
    name_str = "model_resnet18_dataset_flowers102_lr_0.001_weight_decay_0.0001"
    config = parse_config_string(name_str, config_keys)
    # Now use config.lr, config.weight_decay, etc.

## Analyzing Sweep Results

After your sweep completes, use `get_wandb_results.py` in the repository root to extract and analyze results from the WandB API.

### Basic Usage

```bash
python get_wandb_results.py "https://wandb.ai/entity/project/sweeps/SWEEP_ID"
```

This downloads raw architecture progression data into a CSV file named `entity_project_sweep_arch_scores.csv`.

### Output Modes

The script supports three modes via the `--mode` (or `-m`) argument:

#### 1. Download Mode (default)
Downloads all raw log entries from the sweep:

```bash
python get_wandb_results.py "URL"
```

**Output**: `entity_project_sweep_arch_scores.csv`

**Columns**: `run_id`, `run_name`, `state`, `step`, `timestamp`, `arch_param_count`, `arch_max_val`, `arch_dendrite_count`, `config_*` (hyperparameters)

**When to use**: Need raw data for custom analysis or verification

#### 2. By-Run Mode
Creates a pivot table for line graphs where each run is a separate line:

```bash
python get_wandb_results.py "URL" -m gen-by-run
```

**Output**: `entity_project_sweep_by_run.csv`

**Format**: Rows = Arch Param Count, Columns = Run names, Values = Arch Max Val

**When to use**: Compare different hyperparameter configurations as lines on the same graph

#### 3. By-Dendrite Mode
Creates scatter plot data grouped by dendrite count:

```bash
python get_wandb_results.py "URL" -m by-dendrite
```

**Output**: `entity_project_sweep_by_dendrite.csv`

**Format**: Columns = `run_name`, `param_count`, `dendrite_0_max_val`, `dendrite_1_max_val`, ...

**When to use**: Visualize how adding dendrites improves performance across different runs

### Important Options

#### Final Metrics (`--include-final`)
Add final metrics (logged at training completion) to the output:

```bash
python get_wandb_results.py "URL" --include-final
```

Adds columns: `final_param_count`, `final_max_val`, `final_dendrite_count`

**Use for**: Verifying final logged values match the last architecture values  
**Don't use for**: Graph generation (creates duplicate data points)

#### Dendrite Offset (`--dendrite-offset`)
Specify starting dendrite counts for models that begin with dendrites already present (e.g., pretrained models):

```bash
# Model with index 0 starts with 2 dendrites
python get_wandb_results.py "URL" --dendrite-offset "0:2"

# Multiple models with different starting counts
python get_wandb_results.py "URL" --dendrite-offset "0:2" "1:3"
```

**Format**: `"model_index:count"`

**Used for**:
- Suppressing diagnostic warnings about "missing" dendrites
- Correctly processing data in by-dendrite mode

### CSV Caching

When using `gen-by-run` or `by-dendrite` modes, the script automatically checks for an existing raw CSV file and uses it if found, avoiding unnecessary WandB API calls. To force a fresh download, delete the CSV or run download mode first.

### Common Workflows

**Workflow 1: Quick visualization**
```bash
# Download and generate by-run comparison
python get_wandb_results.py "URL" -m gen-by-run
# Import entity_project_sweep_by_run.csv into Excel/Google Sheets for line graph
```

**Workflow 2: Analyze dendrite progression**
```bash
# Generate dendrite-focused data
python get_wandb_results.py "URL" -m by-dendrite --dendrite-offset "0:2"
# Create scatter plot showing performance vs dendrite count
```

**Workflow 3: Full verification**
```bash
# Download with all metrics
python get_wandb_results.py "URL" --include-final
# Inspect CSV to verify final values match last arch values
```

### Understanding the Output

**Arch metrics** (`Arch Max Val`, `Arch Param Count`, `Arch Dendrite Count`): Logged each time the model restructures (adds dendrites). Shows the best validation score achieved for that architecture before adding more dendrites.

**Final metrics** (`Final Max Val`, `Final Max Train`, `Final Param Count`, `Final Dendrite Count`): Logged once at training completion. Represents the overall best performance across all architectures.

**Key insight**: For successful dendrite integration, `Final Max Val` should equal or exceed the best `Arch Max Val` values, and `Final Dendrite Count` should equal `num_dendrites_integrated` (successfully integrated dendrites, not just attempted).
