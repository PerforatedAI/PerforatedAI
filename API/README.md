# PAI-API

This README provides a walkthrough for how to add dendrites to your code.  When starting a new project first just add the sections from this README. Once they have been added you can run your code and it will give you errors and warnings about if any "customization" coding is required for your architecture.  The ways to fix these are in [customization.md](customization.md).  Additionally the customization README begins by describing alternative options to the recommended settings here.  After running your pipeline you can view the graphs in the PB that show the correlation values and experiment with the other settings in customization.md that may help get better results. [output.md](output.md) describes what you're seeing in the graphs.

## 1 - Main Script

First install perforatedai from the main folder with:

    pip install -e .

### 1.1 - Imports
These are all the imports you will need at the top of your main training file.  They will be needed in all of your files that call these functions if some of the below ends up being put into other files.

    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    
## 2 - Network Initialization
A large benefit PAI provides is automatic conversion of networks to work with dendrite nodes through the initializePB function.
    
    
### 2.1 - Network Conversion
The call to initializePB should be done directly after the model is initialized, before cuda and parallel calls.
    
    model = yourModel()
    model = UPA.initialize_pai(model)

## 3 - Setup Optimizer

When calling intitializePB a pb_neuron_layer_tracker called pai_tracker will be created.  This keeps track of all neuron modules and important values as well as performing the actions behind the scenes to add dendrite modules where they need to go.  It also must have a pointer to the optimizer being used. To get started quickly, or if the optimizer is hidden by a training framework, the following can be used:

    GPA.pai_tracker.set_optimizer_instance(optimizer)

However, we reccomend your optimizer and scheduler should be set this way instead. This method will automatically sweep over multiple learning rate options when adding dendrites, where often a lower learning rate is better for when after dendrites have been added. If you do use this method, the scheduler will get stepped inside our code so get rid of your scheduler.step() if you have one.  We recommend using ReduceLROnPlateau but any scheduler and optimizer should work.

    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimArgs = {'params':model.parameters(),'lr':learning_rate}
    schedArgs = {'mode':'max', 'patience': 5} #Make sure this is lower than epochs to switch
    optimizer, PAIscheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    
    Get rid of scheduler.step if there is one. If your scheduler is operating in a way
    that it is doing things in other functions other than just a scheduler.step this
    can cause problems and you should just not add the scheduler to our system.
    We leave this uncommented inside the code block so it is not forgotten.
    
    Another note - It seems that weight decay can sometimes cause problems with dendrite learning.  If you currently have weight decay and are not happy with the results, try without it.
    
## 4 - Scores

### 4.1 Validation Requirements
At the end of your validation loop the following must be called so the tracker knows when to switch between dendrite learning and normal learning

    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(score, 
    model) # .module if its a dataParallel
    Then following line should be replaced with whatever is being used to set up the gpu settings, including DataParallel
    model.to(device)
    if(training_complete):
        Break the loop or do whatever you need to do once training is over
    elif(restructured): if it does get restructured, reset the optimizer with the same block of code you use initially. 
        optimArgs = your args from above
        schedArgs = your args from above
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        if you are doing set_optimizer_instance you instead need to do the full reinitialization here
    
Description of variables:

    model - This is the model to continue using.  It may be the same model or a restructured model
    restructured - This is True if the model has been restructured, either by adding new 
    Dendrites, or by incorporating trained dendrites into the model.
    training_complete - This is True if the training is completed and further training will
    not be performed.  
    score - This is the validation score you are using to determine if the model is improving.
    It can be an actual score like accuracy or the loss value.  If you are using a loss value
    be sure when you called initialize() you set maximizing_score to False.
    
#### 4.1.1 Separate Validation Functions
If this is called from within a test/validation function you'll need to add the following where the validation step is called

    return model, optimizer, scheduler
      
And then set them all when it is called like this
      
    model, optimizer, scheduler = validate(model, otherArgs)
        
Additionally make sure all three are being passed into the function because otherwise they won't be defined if the network is not restructured

### 4.2 Extra Scores
Adding the validation score like above is required since it is the function that actually updates your model.  However, if is often helpful to also keep track of other scores such as train or test scores.  By calling the function below these scores will also be added to your graph and the best_arch_scores csv file to track the corresponding scores at the epoch where the best validation score was calculated for each dendrite count.

    GPA.pai_tracker.add_extra_score(training_score, 'Train')

Additionally, if you would like to track the extra score for the csv file, but not graph it you can call the function below.  This is often best if you would like to keep track of both an accuracy score and a loss score, or generally when you are tracking metrics where it would not make sense to graph them both on the same axis.

    GPA.pai_tracker.add_extra_score_without_graphing(test_score, 'Test Accuracy')
    
### 4.3 Additional Scores

In additional to the above which will be added to the graph you may want to save scores thare are not the same format.  A common reason for this is when a project calculates training loss, validation loss, and validation accuracy, but not training accuracy.  You may want the graph to reflect the training and validation loss to confirm experiments are working and both losses are improving, but what is the most important at the end is the validation accuracy.  In cases like this just use the following to add scores to the csv files but not to the graphs.

    GPA.pai_tracker.add_extra_score_without_graphing(extraScore, 'Test Accuracy')

## 5 - Training Loop Modification

The pai_tracker will tell you when the program should be stopped by returning training_complete as true.  This occurs when a set of dendrites has been added which does not improve the validation score.  At this time the previous best network is loaded and returned.  Because this happens automatically you should change your training loop to be a while(True) loop or set epochs to be a very high number.  Be careful if this has impact on your learning rate etc.  E.g.:

    for epoch in range(1, args.epochs + 1):
    ->
    epoch = -1
    while(True):
        epoch += 1

## That's all that's Required!
With this short README you are now set up to try your first experiment.  When your first experiment runs it will have a default setting called `GPA.pc.set_testing_dendrite_capacity(True)`.  This will test your system with adding three sets of dendrites to ensure all setup parameters are correct and the GPU can handle the size of the larger network.  Once it has been confirmed your script will output a message telling you the test has compelted.  After this message has been received, set this variable to be False to run a real experiment.

While you are testing the dendrite capacity, Warnings will automatically be generated when problems occur which can be debugged using the [customization](customization.md) README.  This README also contains some suggestions at the end for optimization which can help make results even better. If there are any actual problems that are occuring that are not being caught and shown to you we have also created a [debugging](debugging.md) README with some of the errors we have seen and our suggestions about how to track them down.  Please check the debugging README first but then feel free to contact us if you'd like help. Additionally, to understand the output files take a look at the [output](output.md) README.
