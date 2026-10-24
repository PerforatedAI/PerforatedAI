# Huggingface MNIST Instructions

## Transformers Fork

To work with hugginface some internal mechanisms of the trainer must be changed to do Perforated AI steps where they need to be done.  We have created a repo that is a fork which has everything in the correct place.  Get it and install requirements by running:

    pip install -r requirements.txt
    git clone https://github.com/PerforatedAI/transformers-perforated.git
    cd transformers-perforated
    pip install -e .
    cd ..
    pip install perforatedai

    
## Other Examples
The rest of this readme covers the mnist example in this folder.  But additional examples are as follows:
    [BERT Language Modeling](https://github.com/PerforatedAI/transformers-perforated/tree/main/examples/pytorch/language-modeling)
    
## Code Changes

### Imports and Constants

    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA

    # When to switch between Dendrite learning and neuron learning.
    GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    # Patience: epochs without improvement before switching modes. Keep this well below num_train_epochs.
    GPA.pc.set_n_epochs_to_switch(3)
    # The default shape of input tensors
    GPA.pc.set_output_dimensions([-1, 0, -1, -1])

    and just for testing
    GPA.pc.set_testing_dendrite_capacity(True)

### Validation Score

The Trainer runs evaluation itself, so tell PAI which key of the eval metrics to use.  Set these before `perforate_model`, and use the setters (assigning `GPA.metric` has no effect):

    GPA.pc.set_library_validation_score("eval_accuracy")  # "eval_" + the key returned by compute_metrics
    GPA.pc.set_library_extra_scores(["train_loss"])       # optional extra scores to log and graph

### Save Name

The save name is passed directly to `perforate_model`.  See the Converting Network section below.

### Trainer

You need to make sure that the trainer does evaluation after every epoch so PAI has a validation score to work with.  Without it PAI never gets a score and training never ends.  The patched fork handles training termination internally, so `num_train_epochs` no longer stops training; it sets the length of the learning rate schedule, which restarts after every PAI restructure.  Keep it at your normal value:

    training_args.eval_strategy = "epoch"
    training_args.num_train_epochs = 10

Then turn PAI on in the Trainer itself:

    trainer = MyTrainer(
        ...
        perforate_model=True,
    )

### Converting Network

    model = UPA.perforate_model(
        model,
        save_name='mnistHF',     # Change for different parameter runs
        maximizing_score=True,   # True for maximizing validation score (eval_accuracy), False for minimizing loss
        making_graphs=True)      # True to save output graphs
    
### TestingDendriteCapacity

While testing dendrite capacity you still want epoch-level evaluation.  Just ensure your dataset or epoch length is short enough that cycles complete quickly.
    
### Ignoring Shared Tensors

Some models within huggingface don't play niceley with safetensors, which is the method used to save Perforated AI models.  If you encounter the following error:
    
    Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again
    
This means the model is setup in a way that has tensors sharing memory.  If you created the model you should fix this problem.  But if you did not create the model, it can generally be ignored.  At the top of your file with the rest of your imports just overwrite the function within safetensors that finds shared tensors with a function that just returns empty to convince it there are no shared tensors.

    import safetensors
    from collections import defaultdict
    def _ignore_shared_tensors(state_dict):
        tensors = defaultdict(set)
        return tensors
    
    safetensors.torch._find_shared_tensors = _ignore_shared_tensors
    
## Running
    
Then just run as usual:

    CUDA_VISIBLE_DEVICES=0 python mnist_huggingface_perforatedai.py 

Validation scores of original and dendrite optimized networks:

!["Output Graph](OutputGraph.png "Output Graph")

Exact graph that gets generated within the output folder:

!["Example Output](ExampleOutput.png "Example Output")


## Example Output
This shows an example output which quit after 2 Dendrite Epochs.

![ExampleOutput](ExampleOutput.png "Example Output")
