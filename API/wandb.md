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
    # Used for all dendritic models:

    # Speed of improvement required to prevent switching
    # 0.1 means dendrites wills witch after score stops improving by 10% over recent history
    # This extra-early early-stopping sometimes enables high final scores as well as
    # acheiving top scores with fewer epochs
    "improvement_threshold": {"values": [0.1, 0.01, 0.001]}
    # Multiplier to initialize dendrite weights
    "candidate_weight_initialization_multiplier": {"values": [0.1, 0.01]}
    # Forward function for dendrites
    "pb_forward_function": {"values": [torch.sigmoid, torch.relu, torch.tanh]}

    # Only used with Perforated Backpropagation add-on

    # A setting for dendritic connectivity
    "dendrite_graph_mode": {"values": [True, False]},
    # A setting for dendritic learning rule
    "dendrite_learn_mode": {"values": [True, False]},
    # A setting for dendritic weight adjustments
    "dendrite_update_mode": {"values": [True, False]}
}

## Additional Parameter 
Another important parameter, which can be included in sweeps or kept separate, is cap_at_n.
If set to true this will cap the total number of dendrite training epochs to be the total number
of neuron epochs.  This is only used for Perforated Backpropagation training where they are separated.
This defaults to False which has the best results, however, setting to True can sometimes cut 
out a lot of epochs for only slightly worse performance.

    GBP.cap_at_n = True

## Updating config and generating ID
Once this dict has been setup with options, the sweep config can be adjusted and the wandb sweep can be initialized.

    sweep_config["parameters"] = parameters_dict
    sweep_id = wandb.sweep(sweep_config, project="My First Dendrite Project")

## Getting the Run Setup
Next you have to modify your main function to actually run the sweep, this can be done very simply
by creating a wrapper function run() which calls your original main.

    def run():
        with wandb.init(config=sweep_config) as run:
            main(run)

    if __name__ == "__main__":
        # Count is how many runs to perform.
        wandb.agent(sweep_id, run, count=100)

## Using the config values
Inside your main you now have access to the current config.  This can be used as follows:

    def main(run):
        config = run.config
        PBG.improvement_threshold = config.improvement_threshold
        PBG.candidate_weight_initialization_multiplier = config.candidate_weight_initialization_multiplier
        PBG.pb_forward_function = config.pb_forward_function
        PBG.dendrite_graph_mode = config.dendrite_graph_mode
        PBG.dendrite_learn_mode = config.dendrite_learn_mode
        PBG.dendrite_update_mode = config.dendrite_update_mode
        name_str = "_".join([f"{key}_{wandb.config[key]}" for key in wandb.config.keys()])
        run.name = name_str

## Logging
Then inside your script you'll also need to add logging.  You can add as much or as little as you'd like
But we'd recommend at least logging training and validation scores

    run.log({"ValAcc": val_acc_val, "TrainAcc": train_acc_val, "TestAcc": test_acc_val})
        

## Running
Now just run your program as usual and you will be able to see the sweep on the sweeps tab on your weights and biases homepage