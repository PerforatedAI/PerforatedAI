# Getting started with Weights and Biases

Weights and biases includes a great tool for doing hyperparameter sweeps.  This is strongly recommended to use when getting started with dendrites because the dendritic hyperparameters play an important role and with each new project there could be a high variation in optimal parameters.

## Initialization
Make sure you have already created an account and then add the import at the top of your file.

    import wandb

Note: wandb.login() is typically handled automatically or via environment variables. Manual login in code is optional.

## Set Up the Sweep
We typically recommend using the random sweep method and maximizing the Final Max Val metric (the best validation score achieved across all architectures).  But other sweep methods are available and minimizing loss is also an appropriate goal.

    def get_sweep_config(dataset_name):
        """Get WandB sweep configuration for a specific dataset."""
        base_config = {
            "method": "random",  # Random sampling
            "metric": {"name": "Final Max Val", "goal": "maximize"},
        }
        
        # Example for a typical dataset
        base_config["parameters"] = {
            "dataset": {"value": dataset_name},
            "lr": {"values": [0.0001, 0.0003, 0.001, 0.003]},
            "weight_decay": {"values": [0.0, 1e-5, 1e-4, 1e-3]},
            "label_smoothing": {"values": [0.0, 0.05, 0.1]},
            # Dendritic hyperparameters
            "improvement_threshold": {"values": [[0.01, 0.001, 0.0001, 0], [0]]},
            "pai_forward_function": {"values": ["sigmoid", "relu", "tanh"]},
        }
        
        return base_config

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
        config_keys = ["model", "dataset", "lr", "weight_decay", "label_smoothing", 
                       "improvement_threshold", "pai_forward_function"]
        name_parts = [f"{k}_{config.get(k, 'default')}" for k in config_keys if k in config]
        if name_parts:
            wandb.run.name = "_".join(name_parts)
        
        # Your training code here, using config values
        # ...
    
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser(description="Train with WandB sweep support")
        parser.add_argument("--sweep-dataset", type=str, help="Initialize sweep for this dataset")
        parser.add_argument("--sweep-id", type=str, help="Join existing sweep by ID")
        parser.add_argument("--sweep-count", type=int, default=100, help="Number of runs")
        args = parser.parse_args()
        
        if args.sweep_dataset or args.sweep_id:
            if args.sweep_id:
                # Join existing sweep
                wandb.agent(args.sweep_id, function=train_with_wandb, 
                           count=args.sweep_count, project="your-project")
            elif args.sweep_dataset:
                # Initialize new sweep
                sweep_config = get_sweep_config(args.sweep_dataset)
                sweep_id = wandb.sweep(sweep_config, project=args.sweep_dataset)
                print(f"Sweep initialized: {sweep_id}")
                print(f"To join from other machines: --sweep-id {sweep_id}")
                
                # Start agent on this machine
                wandb.agent(sweep_id, function=train_with_wandb, 
                           count=args.sweep_count, project=args.sweep_dataset)
        else:
            # Single run mode
            # Your normal training code here


## Using the config values

Access sweep parameters via `wandb.config`. Apply them when configuring PAI settings:

    def train_with_wandb():
        wandb.init()
        config = wandb.config
        
        # Apply dendritic hyperparameters from config
        if hasattr(wandb, "run") and wandb.run is not None and hasattr(wandb, "config"):
            if "improvement_threshold" in config:
                GPA.pc.set_improvement_threshold(config.improvement_threshold)
            
            if "pai_forward_function" in config:
                pai_fwd = config.pai_forward_function
                if pai_fwd == "sigmoid":
                    GPA.pc.set_pai_forward_function(torch.sigmoid)
                elif pai_fwd == "relu":
                    GPA.pc.set_pai_forward_function(torch.relu)
                elif pai_fwd == "tanh":
                    GPA.pc.set_pai_forward_function(torch.tanh)
        
        # Use config values for standard hyperparameters
        lr = config.lr
        weight_decay = config.weight_decay
        label_smoothing = config.label_smoothing
        
        # ... rest of your training code

### Retaining Perforated AI Logs

When calling perforate_model, set save_name to organize dendrite test results. 
For sweeps, include config values in the save_name to keep each run's results separate:

    # Build save_name from wandb config
    if hasattr(wandb, "run") and wandb.run is not None and hasattr(wandb, "config"):
        config_keys = ["model", "dataset", "lr", "weight_decay", "label_smoothing",
                       "improvement_threshold", "pai_forward_function"]
        name_parts = [f"{k}_{wandb.config.get(k, 'default')}" 
                     for k in config_keys if k in wandb.config]
        save_name = "_".join(name_parts) if name_parts else "default_run"
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
            "Param Count": sum(p.numel() for p in model.parameters()),
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
        max_params = sum(p.numel() for p in model.parameters())
    
    # Track global best (only during neuron training)
    if current_mode == "n" and val_acc > global_max_val:
        global_max_val = val_acc
        global_max_train = train_acc
        global_max_params = sum(p.numel() for p in model.parameters())
    
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


## Running Sweeps

### Initialize a new sweep:
```bash
python your_script.py --sweep-dataset flowers102 --sweep-count 100
```

This will:
1. Create the sweep configuration for the dataset
2. Initialize the sweep on WandB
3. Print the sweep ID for joining from other machines
4. Start running sweep agents on this machine

### Join an existing sweep from another machine:
```bash
python your_script.py --sweep-id YOUR_SWEEP_ID --sweep-count 100
```

### View results:
Visit https://wandb.ai to see your sweep dashboard with all runs, metrics, and visualizations.

## Complete Example

See `Examples/imagenet_pretrained/train_from_hf_wandb_sweep.py` for a working implementation that includes:
- Dataset-specific sweep configurations
- Model index mapping for better WandB visualization
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
