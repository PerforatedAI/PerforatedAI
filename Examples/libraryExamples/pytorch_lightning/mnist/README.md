# PyTorch Lightning MNIST Example

## Setup

    pip install -r requirements.txt
    pip install perforatedai

## Code Changes

### Model Definition

There are a couple changes that are required within the model definition.  The first is to add the following to the init function. 

    self.epochs = 0

This will come into play later when we use it as the value for early stopping to convince ptl to use our training_complete value to stop running cycles

#### on_validation_epoch_end

This is the function that computes the validation score so we need to use addValidationScore here.  A few other changes are required as well.
    
Store the total number of epochs for logging
    
    goodEpochs = self.epochs
    
Wrap everything inside this if block when you are doing a initial validation step before any training

    if(self.epochs != 0):
    
Then is the main addValidationScore block.

    self.model, improved, restructured, training_complete = PBG.pai_tracker.addValidationScore(your score value, 
        self.model, # .module if its a dataParallel
        your save value)
    self.model.to('cuda')
    if(training_complete):
        #send the early stop signal by not increaseing the good epochs
        goodEpochs = 0
    elif(restructured): 
        # This call will reinitialize the optimizers to point to the new model
        self.trainer.strategy.setup(trainer)
    self.log("Good Epochs", goodEpochs)
    self.epochs += 1
    
However, there are a few changes of note. Inside training_complete block you now see goodEpochs has been reset to 0 this is so that when it is logged at the end of the block that log will continuously go up every single validation epoch until training_complete returns true.  Later we will define an early stopping function to stop as soon as that counter stops going up every time.

Next in restructured instead of redefining the optimizer and scheduler as usual, we call `self.trainer.strategy.setup(trainer)` which calls configure_optimizers inside.

Finnaly epochs is incremented so that the goodEpochs counter can continue to go up every time until training_complete is true.

#### configure_optimizers

This funtion should just be changed in the same was as usual for optimizers and schedulers.

    def configure_optimizers(self):
        PBG.pai_tracker.setOptimizer(torch.optim.Adam)
        PBG.pai_tracker.setScheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        optimArgs = {'params':self.model.parameters(),'lr':1e-3}
        schedArgs = {'mode':'min', 'patience': 5}
        self.optimizer, self.scheduler = PBG.pai_tracker.setupOptimizer(model, optimArgs, schedArgs)
        #don't actually return the scheduler since PB handles scheduler interanlly
        return [self.optimizer]

#### Trainer Setup
The reason to add the Good Epochs log is to early stop when we say there was an epoch that was not good.  That can be acheived with the following:

early_stopping = EarlyStopping('Good Epochs', mode='max', patience=0)

trainer = pl.Trainer(
                        Your original settings,
                        callbacks=[early_stopping]
                    )
    
## Running

Then just run as usual:

    CUDA_VISIBLE_DEVICES=0 python pl_mnist_perforatedai.py 

## Example Output
This shows an example output which quit after 2 Dendrite Epochs.

![ExampleOutput](ExampleOutput.png "Example Output")
