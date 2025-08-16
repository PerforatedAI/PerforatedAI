# Perforated AI Outputs

# Graphs

!["MNIST example 1](mnist1.png "MNIST1")
!["MNIST example 2](mnist2.png "MNIST2")
Four graphs are automatically generated at every epoch.  The first is all of the scores you are saving.  The Validation scores and a running validation score are first to keep track of what is the deciding factor for early stopping and Dendrite learning switching.  Additionally each score that was added with addExtraScore and addTestScore will be included.  In each case a red vertical line represents a switch to dendrite training and a blue vertical line represents a switch back to neuron training.  For GD dendrites only a blue line will be shown because dendrite training happens along with neuron training.  Additionally, when switching to dendrite training values are retained only for display purposes for what happened before the switch since the switch triggers the weights from the best validation score to be loaded.

The second is the training times to compare runtime as Dendrites are added.

The third is the learning rate to view how the rate is adjusted over epochs.

The fourth graph shows the correlation scores of each set of Dendrite Nodes.  This graph is what can be used to determine if you should be wrapping your network differently.  For GD training this graph will be blank.

Graphs and model files are retained in the directory of the save name you specified, which is PB by default.

In addition csv files are output with the values created in the graphs.  An additional csv file is created which has the most important values in one place ending with bestTestScores.  This csv file will track the number of parameters of your system each time a new set of Dendrites has been added as well as the test scores affiliated with the best validation scores with each architecture.  If addTestScore has not been called it just tracks the best validation scores.

## What to Look for in the Graphs for PB Training

For PB training the PB scores of every module should not be significantly lower than 0.001, if they're lower than this it means correlation isn't being learned.  To debug this look into the options in the [customization](customization.md) README and play around with the architecture with what modules are wrapped and how you are defining your processors.
    
Additionally the validation/training scores should be flatlined during Dendrite learning.  There may be minor fluctuation if you are randomizing the order of your inputs but if there are any significant changes there is a problem.  The only times we have seen this happen is when modules were not properly wrapped.  When you first initialize the network there should be a warning for modules that exist in the model that were not wrapped.  If that is being displayed when you run your system try to clear the list so everything is wrapped.  Then most importantly, if things are working, After every blue vertical line, your scores should be getting better.

## Output Names

  - name_x_startSteps_y
    X specifieds how many dendrites have been created.  Y specifies how many times the learning rate was stepped before starting this cycle.
 - beforeSwitch_x
    State of the network before performing the switch out of cycle x
 - switch_x
    State of the network directly after performing the switch into cycle x
 - best_model_beforeSwitch_x 
    State of the best validation score before switching out of cycle x
 - best_model
    Best model achieved by validation score
 - final_clean_pai
    When experiment is complete this is the best model that was produced
 -latest
    Most recently created model.  This is what you should use if anything crashes and you want to pick up where it left off.
 - anything_pai
    The _pai versions of models are optimized for inference.  These models are smaller and also can be run with open source code from this API without requiring a license to use the training software.

## CSV Names
 - bestTestScores
    A final scores output that shows the best test scores at the epoch where the best validation scores were achieved.  This includes parameter counts for each version of the network as well.  This could be considered the main output results of an experiment.  Requires using addTestScore to work properly.
- Best_PB_Scores
    Best PB correlation scores at each epoch
 - Scores
    Best Validation scores and any extra scores that were added at each epoch
 - Times
    Timing information for each epoch
 - learningRate
    Learning Rate at each epoch
 - paramCounts
    Total parameter count of network during each cycle
 - switchEpochs
    List of which epochs cycles switched between neuron training and Dendrite training.
