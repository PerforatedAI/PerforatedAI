# Getting started with Weights and Biases

Weights and biases includes a great tool for doing hyperparameter sweeps.  This is strongly recommended to use when getting started with dendrites because the dendritic hyperparameters play an important role and with each new project there could be a high variation in optimal parameters.

## Initialization
Make sure you have already created an account and then add the import and login at the top of your file.

    import wandb
    wandb.login()

## Set Up the Sweep
We typically recommend using the random sweep method and maximizing the validation accuracy.  But other sweep methods are available and minimizing the validation loss is also an appropriate goal.

    sweep_config = {"method": "random"}
    metric = {"name": "ValAcc", "goal": "maximize"}
    sweep_config["metric"] = metric

## Picking Hyperparameters
In addition to normal hyperparameters you may want to use, these are the ones most important for dendrites.

    parameters_dict = {

    # Associated values for sweeping

    # Dropout can be especially important if your training is already higher than your validation
    "dropout": {"values": [0.1, 0.3, 0.5]},
    # Weight decay can have negative impact on dendritic models
    "weight_decay": {"values": [0, your original value]},

    # Used for all dendritic models:

    # Speed of improvement required to prevent switching
    # 0.1 means dendrites will switch after score stops improving by 10% over recent history
    # This extra-early early-stopping sometimes enables high final scores as well as
    # achieving top scores with fewer epochs
    "improvement_threshold": {"values": [[0.01, 0.001, 0.0001, 0], [0.001, 0.0001, 0]]},
    # Multiplier to initialize dendrite weights
    "candidate_weight_initialization_multiplier": {"values": [0.1, 0.01]},
    # Forward function for dendrites
    "pai_forward_function": {"values": ["sigmoid", "relu", "tanh"]},

    # Only used with Perforated Backpropagation add-on

    # A setting for dendritic connectivity
    "dendrite_graph_mode": {"values": [True, False]},
    # A setting for dendritic learning rule
    "dendrite_learn_mode": {"values": [True, False]},
    }

## Additional Parameter 
Another important parameter, which can be included in sweeps or kept separate, is cap_at_n.
If set to true this will cap the total number of dendrite training epochs to be the total number
of neuron epochs.  This is only used for Perforated Backpropagation training where they are separated.
This defaults to False which has the best results, however, setting to True can sometimes cut 
out a lot of epochs for only slightly worse performance.

    GPA.pc.set_cap_at_n(True)

## Getting the Run Setup
Next you have to modify your main function to actually run the sweep, this can be done very simply
by creating a wrapper function run() which calls your original main.  We recommend the following
try/except which drops into pdb on failure instead of exiting since often there will be problems
setting this up for the first time.  This setup will also allow you to run on multiple instances by setting args.sweep_id to "main" for the original and the sweep id for any additional instances

    def run():
        try:
            with wandb.init(config=sweep_config) as run:
                main(run)
        except Exception:
            import pdb
            pdb.post_mortem()
    if __name__ == "__main__":
        # Count is how many runs to perform.
        project="Dendritic Optimization"
        sweep_config["parameters"] = parameters_dict
        if args.sweep_id == "main":
            sweep_id = wandb.sweep(sweep_config, project=project)
            print("\nInitialized sweep. Use --sweep_id", sweep_id, "to join on other machines.\n")
            # Optionally run the agent on this machine as well
            wandb.agent(sweep_id, run, count=300)
        else:
            # Join the existing sweep as an agent
            wandb.agent(args.sweep_id, run, count=300, project=project)


## Using the config values
Inside your main you now have access to the current config.  This can be used as follows:

    def main(run):
        config = run.config
        GPA.pc.set_improvement_threshold(config.improvement_threshold)
        GPA.pc.set_candidate_weight_initialization_multiplier(
            config.candidate_weight_initialization_multiplier
        )
        if config.pai_forward_function == "sigmoid":
            pai_forward_function = torch.sigmoid
        elif config.pai_forward_function == "relu":
            pai_forward_function = torch.relu
        elif config.pai_forward_function == "tanh":
            pai_forward_function = torch.tanh
        GPA.pc.set_pai_forward_function(pai_forward_function)
        GPA.pc.set_dendrite_graph_mode(config.dendrite_graph_mode)
        GPA.pc.set_dendrite_update_mode(config.dendrite_update_mode)
        name_str = "_".join([f"{key}_{wandb.config[key]}" for key in wandb.config.keys()])
        run.name = name_str

### Retaining Perforated AI Logs

When calling initialize_pai a save name must be set to determine the folder to save dendrite tests.
If you want to retain all of your tests separately you can label your save_name with
the wandb sweep configuration values.

    # Build run name with priority ordering
    excluded = ['method', 'metric', 'parameters']
    priorities = ['dendrite_mode', 'model_arch']
    # Add priority keys first
    name_parts = [str(wandb.config[k]) for k in priorities if k in wandb.config]
    # Add remaining keys in default order
    remaining_keys = [k for k in parameters_dict.keys() if k not in excluded and k not in priorities]
    name_parts.extend(str(wandb.config[k]) for k in remaining_keys if k in wandb.config)
    name_str = "_".join(name_parts)
    run.name = name_str

    model = UPA.initialize_pai(model, save_name=run.name)

## Logging
Then inside your script you'll also need to add logging.  You can add as much or as little as you'd like
But we'd recommend at least logging training and validation scores

    run.log({"ValAcc": val_acc_val, "TrainAcc": train_acc_val, "TestAcc": test_acc_val, "Param Count": UPA.count_params(model), 'Dendrite Count': GPA.pai_tracker.member_vars["num_dendrites_added"]})
        
### Arch Logging
It can sometimes be valuable to also log the best scores at each architecture, and the final scores.  This can be done by adding the following:

    #outside of the loop
    dendrite_count = 0
    max_val = 0
    max_train = 0
    max_test = 0
    max_params = 0
    dendrite_count = 0

    global_max_val = 0
    global_max_train = 0
    global_max_test = 0
    global_max_params =

    #inside the loop after add_validation_score
    if(val_acc > max_val):
        max_val = val_acc
        max_test = test_acc
        max_train = train_acc
        max_params = UPA.count_params(model)
    if(val_acc > global_max_val):
        global_max_val = val_acc
        global_max_test = test_acc
        global_max_train = train_acc
        global_max_params = UPA.count_params(model)
    if(restructured):
        # if n mode and dendrites were just added
       if(GPA.pai_tracker.member_vars["mode"] == 'n' and (not dendrite_count == GPA.pai_tracker.member_vars["num_dendrites_added"])):
            dendrite_count = GPA.pai_tracker.member_vars["num_dendrites_added"]
            run.log({"Arch Max Val": max_val, "Arch Max Test": max_test, "Arch Max Train": max_train, "Arch Param Count": max_params, 'Arch Dendrite Count': GPA.pai_tracker.member_vars["num_dendrites_added"]-1})
    elif training_complete:
        if config.dendrite_mode == 0 or max_dendrites == GPA.pai_tracker.member_vars["num_dendrites_added"]:
            run.log({"Arch Max Val": max_val, "Arch Max Test": max_test, "Arch Max Train": max_train, "Arch Param Count": max_params, 'Arch Dendrite Count': GPA.pai_tracker.member_vars["num_dendrites_added"]})
        run.log({"Final Max Val": global_max_val, "Final Max Test": global_max_test, "Final Max Train": global_max_train, "Final Param Count": global_max_params, 'Final Dendrite Count': GPA.pai_tracker.member_vars["num_dendrites_added"]})


## Running
Now just run your program as usual and you will be able to see the sweep on the sweeps tab on your Weights and Biases homepage


## Optional Debugging Function

We sometimes aften a run want to re-run with specific settings where our best models came from.  By using the name that is defined by the parameters this string can also be used to recreate the config without editing the rest of your code.

    from types import SimpleNamespace

    def parse_config_string(name_str, parameters_dict):
        prefix = "Dendrites-"
        if not name_str.startswith(prefix):
            raise ValueError("name_str must start with 'Dendrites-'")
        rest = name_str[len(prefix):]
        if "_" in rest:
            dend_token, rest = rest.split("_", 1)
        else:
            dend_token, rest = rest, ""
        result = {"dendrite_mode": dend_token}
        tokens = rest.split("_") if rest else []
        excluded = ['method', 'metric', 'parameters', 'dendrite_mode']
        keys = [k for k in parameters_dict.keys() if k not in excluded]
        for key in keys:
            if not tokens:
                break
            result[key] = tokens.pop(0)
        if tokens:
            result["_extras"] = tokens
        return SimpleNamespace(**result)

Then in your main include, for example:

    name_str = "Dendrites-0_0.3_3_1_0.5_0.1_10_5_2_2_0.01_0_0_1_0"
    config = parse_config_string(name_str, parameters_dict=parameters_dict) 
    
Anw now this config can be used instead of the wandb config.  Just be sure the same parameters dict that you used to train is passed into this function.
