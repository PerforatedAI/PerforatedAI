# Debugging

If you are crashing in ways that aren't caught by the other md files, search for them here before looking up online.

## New Best Scores are not Being Tiggered
Check GPA.pc.get_improvement_threshold().  If the improvement is very small it may not be beating the previous best by a high enough margin.  You can also set GPA.pc.set_verbose(True) to check if this is the case.

Also ensure that maximizing_score is set properly in initialize_pai.  If you are maximizing an accuracy score this should be true, if you are minimizing a loss score this should be false.  

## Errors Where Warnings are Printed

    "The following layer has not properly set this_output_dimensions"

Check out the suggestions that are printed and section 4 in customization

    Didn't get any non zero scores or a score is nan or inf.
    
This means that the Dendrites learned a correlation that was either nan or infinite.  We have only seen this happen with training pipelines where the neurons are also learning weights that are getting close to triggering an inf overflow error themselves.  See if you can add normalization layers to keep your weights within more usual ranges.

    An entire layer got exactly 0 Correlation
    
Same as above but for zero.

    Trying to call backwards but module X wasn't PAIified
    
This means something went wrong with the conversion.  The module is getting triggered for PAI modifications but also wasn't converted in a way that allowed it to initialize properly.  Look into how you set up that layer.

    Need exactly one 0 in the input dimensions
    
This means you set your input dimensions but it wasn't the proper -1s and a single zero as it should be.

## Input Dimensions

    'pbValueTracker' object has no attribute 'out_channels'

Look at section 3 from customization.  This explains how to set input dimensions.

## Broadcast Shape

    Values[0].normalPassAverageD += (val.sum(mathTuple) * 0.01) / fullMult
    RuntimeError: output with shape [X] doesn't match the broadcast shape [Y]
    
If you get this error it means input dimensions were not properly set.  Run again with pdb and when this error comes up print Values[0].layerName to see which layer the problem is with.  You can also print the shape of val to see what the dimensions are supposed to be.  This should be caught automatically so In our experience when this happens it means you have a layer that can accept tensors which have multiple dimensionalities without having problems.  This is not accounted for with our software currently so just wrap that layer in a module as required so you don't need to do that. 

## Errors in Forward
These usually mean the processors were not set up correctly. Look at 1.2 from customization.

    AttributeError: 'tuple' object has no attribute 'requires_grad'

This specifically is saying that you are returning a tuple of tensors rather than a single tensor.  Your processor needs to tell you how to handle this so the Dendrite only collaborates on one tensor with the neuron.

Make sure that you put the GPA.pc.get_modules_with_processing() setup before the call to convertNetwork
    
## Errors in filterBackward
There are a couple errors that can happen in the filterBackward function

    AttributeError: 'NoneType' object has no attribute 'detach'

This also usually means the processors were not set up correctly. Look at 1.2 from customization. This means you are not using the tensor that is being passed to the dendrites.  For example if you are using the default LSTM processor but using hidden_state rather than output from "output, (hidden_state, cell_state)"

    AttributeError: 'pbValueTracker' object has no attribute 'normalPassAverageD'

This means that the pbValueTracker was not properly initialized.  This can happen for two reasons. The first is if you are running on multiple GPUs.  With a single GPU pbValueTracker are setup automatically but when running on multiple GPUs this has to be setup by hand using the saveTrackerSettings and initializeTrackerSettings functions.  Look at the DataParallel section from the customization readme.

The other reason this can happen is if you initialize a Dendrite layer, but then you don't actually use it.  I.e, it is not being called in the forward and backwards pass of the network.  In these cases look into your forward functions and try to track down why the layer is not properly being used. This same effect can take place if you try to add a set of dendrites before performing any training, with our system you should not run initial validation epochs before starting training, or if you do, make sure to not add Dendrites during those cycles.
    
## dype Errors

This is anything like the following:

    (was torch.cuda.HalfTensor got torch.cuda.FloatTensor)
    Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same

If you are not working with float data change GPA.pc.get_d_type() to whatever you are using. eg:

    GPA.pc.set_d_type(torch.double
)

## Device Errors

    RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

Similar as above there is a setting that defaults to using what is available.  If cuda is available but you still don't want to use it call:

    GPA.pc.set_device('cpu')
    
Additionally, if you are running on mac or generally using any device that will not be properly set with the call of `device = torch.device("cuda" if use_cuda else "cpu")` you should set your device as `GPA.pc.set_device(your device type)`
    
## Attribute Error:
    
    AttributeError: 'pb_neuron_layer' object has no attribute 'SOMEVARIABLE'

This is the error you will get if you need to access an attribute of a module that is now wrapped as a pb_neuron_layer.  All you have to do in this case is the following change.

    #model.yourModule.SOMEVARIABLE
    model.yourModule.mainModule.SOMEVARIABLE
    
## Initialize Error

    AttributeError: 'list' object has no attribute 'set_optimizer_instance'

This means the pai_tracker was not initialized.  This generally only happens if zero layers are converted.  Make sure that at least one layer has been converted correctly.

# Path Error During AddValidationScore

    TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType
    
This means you entered None for the saveName, likely args.saveName did not have a default value.


## setupOptimizer Error
    
    TypeError: 'list' object is not callable
    
If you get this error in setupOptimizer it means you called setupOptimizer but you did not call setOptimizer.  Be sure to call that first.


## Size Mismatch

    File "perforatedai/pb_layer.py", line X, in perforatedai.pb_layer.pb_neuron_layer.forward
    RuntimeError: The size of tensor a (X) must match the size of tensor b (X) at non-singleton dimension 
    
If you get this error it means your neurons are not correctly matched in setoutput_dimensions.  If your 0 is in the wrong index the tensors that get used for tracking the Dendrite to Neuron weights will be the wrong size.

## Initialization Errors

    File "perforatedai/pb_layer.py" ... perforatedai.pb_layer.pb_neuron_layer.__init__
    IndexError: list index out of range
    
This means you did something wrong with the processing classes.  We have seen this before when moduleNamesWithProcessing and moduleByNameProcessingClasses don't line up.  Remember they need to be added in order in both arrays, and if the module is "by name" the processor also has to be added to the "by name" array.



## Index Out of Range in saveGraphs

    perforatedai.pb_neuron_layer_tracker.pb_neuron_layer_tracker.saveGraphs
    IndexError: list index out of range

This error likely means that you added the validation score before the test score.  Test scores must be added before the validation score since graphs are generated when the validation score is added and the tracker must have access to the test scores at that time.


## Things not Getting Converted
The conversion script runs by going through all member variables and determining all member variables that inherit from nn.Module.  If you have any lists or non nn.Module variables that then have nn.Modules in them it will miss them.  If you have a list just put that list into a nn.ModuleList and it will then find everything.  If you do this, make sure you replace the original variable name because that is what will be used. If you use the "add_module" function this is a sign you might cause this sort of problem.  Do not currently have a workaround for non-module objects that contain module objects, just let us know if that is a situation you are in and there is a reason the top object can't also be a module.

## DeepCopy Error

    RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001     

This error has been seen when the processor doesn't properly clear the values it has saved.  Make sure you define a clear_processor function for any processors you create.  If you believe you did and are still getting this error, reach out to us.

This can also happen when forward is called to accumulate gradients but then backwards is not called to clear those gradients.  get GPA.pc.set_extra_verbose(True) to print when gradient tensors are added and removed.  If you have them being added but not removed this is the cause.  Check to make sure optimizer.step() is being called properly.  Some programs have methods that do not call optimizer.step() under certain situations.  Our code is also setup such that optimizer.zero_grad will correct this which can be called as an alternative if the optimizer should actually not be stepped.

If this does not seem to be the problem try going up in the debugger and calling deep copy on individual modules and submodules to track down which module is causing a problem.

## Different Devices

- If you are getting different device problems check if you have a Parameter being set to a device inside the init function.  This seems to cause a problem with calling to() on the main model.

- A second thing to check is if you are calling to() on a variable inside of the forward() function.  Don't do this, just put it on the right device before passing it in.

## Memory Leak

A memory leak is happening if you run out of memory in the middle of a training epoch, i.e. it had enough memory for the first batch but a later batch crashes with OOM error.  These are always a pain to debug but here's some we have caught.

- Check if one of your layers is not being cleared during backwards.  This can build up
    if you are forwarding a module but not calling backwards even though this won't cause a
    leak without PAI in the same model.  We have seen a handful of models which calculate
    values but then never actually use them for anything that goes towards calculating loss,
    so make sure to avoid that.  To check for this you can use:
        GPA.pc.set_debugging_memory_leak(True)
- If this is happening in the validation/test loop after safely completing the train loop make sure you are in eval() mode which does not have a backwards pass.
- Check for your training loop if there are any tensors being tracked during the loop which
    would not be cleared every time.  One we have seen often is a cumulative loss being tracked.
    Without PAI this gets cleared appropriately, but with PAI it does not.  This can be fixed
    by adding detach() before the loss is added to the cumulative variable.
- Try removing various blocks of your model or components of your full training process to try to track down exactly which component is causing the problem.  If you can track it down to exactly which line causes a leak with and without it present, we can help you debug why that line is causing problems if it is on our side.

### Slow Memory Leak Debugging
If the above does not work the following can be used to try to debug problems.  Put this around in your code to try to find where you would expect the count to not be increasing to figure out where it is going up when it shouldn't and then review the above section to see if you may be doing something wrong on that line.  Sometimes the count will be fluctuating for other reasons so try to find the places where the eventual upticks are happening consistently

    import gc
    # Arrays to store history of GPU stats
    gpu_objects_count = []
    def count_objects_on_gpu():
        # Force garbage collection to update counts
        gc.collect()
        # Count number of Python objects on GPU (tensors on cuda device)
        count = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    count += 1
            except:
                pass
        # Append to array
        gpu_objects_count.append(count)
        # Clears array to just retain the most recent 2 for easier viewing
        if(len(gpu_objects_count) < 3):
            print("GPU Objects Count History:", gpu_objects_count)
            return
        del gpu_objects_count[0]
        # Print array
        print("GPU Objects Count History:", gpu_objects_count)



<!-- 
If these don't quickly solve it, the best thing to do would be just move on to another test.  The best method we have to track it down is to try to remove various components from your model until you can identify which one stops the leak when it is gone.

There is another method which involves tracking exactly where cuda tensors are being allocated but it's extremely difficult to track this down with it. -->

## Memory Issues Inside Docker but not Outside
If you are running with a docker container and you were not using docker before it is likely an issue with the shared memory inside the docker container.  Just run with the additional flag of shm-size like so:

    docker run --gpus all -i --shm-size=8g -v .:/pai -w /pai -t pai /bin/bash


## Optimizer Initialization Error

    -optimizer = self.member_vars['optimizer'](**optArgs)
    -TypeError: __init__() got an unexpected keyword argument 'momentum'

This can happen if you are using more than one optimizer in your program.  If you are, be sure to call GPA.pai_tracker.setOptimizer() again when you switch to the second optimizer and also call it as the first line in the if(restructured) block for adding validation scores.


## Debugging Docker Installation

    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
    >>> solved with:
    sudo apt-get install libgl1-mesa-glx

## Saving PAI 

### Parameterized Modules Error

    RuntimeError: Serialization of parametrized modules is only supported through state_dict()

If this is happening it means you have a parameterized module.  You can track down what it is by running in pdb and then calling torch.save on each module of your model recursively until you get to the smallest module which flags the error.  Whatever that one is will have to be changed so you can call torch.save.  We have seen this happen before in a model that used to work because of an updated version of pytorch so downgrading to a pre 2.0 torch version may fix it.  Using safeTensors should resolve this as this error only seems to come up when using_safe_tensors is equal to False.

### Pickle Errors
PAI saves the entire system rather than just your model.  If you run into issues with saving such as 

    `Can't pickle local object 'train.<locals>.tmp_func'`

This likely means the optimizer or scheduler are using lambda functions.  just replace the lambda function with a defined function eg: 

    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  
    #converted to a global function
    def tmp_func(x):
        return (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    lf = tmp_func #where it was originally defined

### Autograd Errors

    Trying to backward through the graph a second time

This is caused by something in your graph containing the same tensor twice.  If you run into this you can try to track it down with the following code block.  Set this up and then call `from perforatedai import globals_perforatedai as GPA; GPA.get_param_name(t_outputs)` within the error block. If this does not work try filling in `GPA.param_name_by_id` with additional tensors

    def get_param_name(tensor):
        return GPA.param_name_by_id.get(id(tensor), None)
    # Create mapping from tensor id to name
    GPA.param_name_by_id = {id(param): name for name, param in model.named_parameters()}
    GPA.get_param_name = get_param_name

It can also sometimes help to use the torchviz package to try to show the entire graph of the tensor.  Go up in debugger to where the problem first occurs in your code then call:

    from torchviz import make_dot; dot = make_dot(TENSOR); dot.render('graph', format='pdf') 

### Safetensors Errors

    Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again:
    
    Then a really long list of pairs
    
    A potential way to correctly save your model is to use `save_model`.
    More information at https://huggingface.co/docs/safetensors/torch_shared_tensors

A lot of modern models have this tendency to save a pointer to a copy of themselves which will cause an error with the Perforated AI save function. This can be remedied in two ways.  First is by using torch.load rather than the safetensors method.  Be aware there is risk of loading pretrained models file from outside your group, and this should only be used when working with models you trust or models which are training from scratch.  To accept this risk and use torch.load set the following: 

    GPA.pc.set_using_safe_tensors(False)
    
However, this will sometimes cause the Parameterized Modules Error above.  In these cases another alternative method is to choose which of the modules is causing the error and add it to GPA.pc.get_module_names_to_not_save().  It will likely not be either of the exact names in the pair list and you will have to find the PAI name for it.  This is often just removing the first "model" string before the first "." but including that ".".This will set the save function to ignore the copy.  We already have the following included by default:

    GPA.pc.append_module_names_to_not_save(['.base_model'])

To remove this default value if you are using a base_model module which is not a duplicate you must clear this array.

#### Weight Tying
In some cases this is done intentionally with weight tying. Which is not just a duplicate pointer, but also a known issue where multiple modules actually are using the same weight tensor in their forward.  We have a workaround for this, but it is only experimental for now so your results may vary.

    GPA.pc.set_using_safe_tensors(True)
    GPA.pc.set_weight_tying_experimental(True)

### Other loading Errors

    KeyError: 'moduleName.mainModule.numCycles'

This error can be caused by a few different reasons:
1 - Calling intializePB before loadPAIModel.  This function should be called on a baseline model not a PAIModel.
2 - Your model definition or modules_to_convert and moduleNamesToConvert lists are different between your training script and your inference script.

## Errors that are currently not fixable

### Loss scaling

    Functions such as ApexScaler or NativeScaler from timm.utils can cause the following error:

        pytorch RuntimeError "has changed the type of value"
    
    These functions are applied after the computation graph is created from the forward pass and types within both PB and the original model are set.  Then when they make adjustments to tensors within the model they do not make the equivilent changes to the tensors in the PB version of the model.  At the moment there is not a workaround for this, so if you encounter this error you just have to turn off loss scaling.

## Errors that are currently not fixable

### Centered RMSprop cusing nan

    We are aware that with RMSprop centered = True can cause correlations to be calculated as nan.  For now, just set the setting to not be centered or pick an alternative optimizer.


## Extra Debugging

If you are unable to debug things feel free to contact us!  We will be happy to help you work through issues and get you running with Perforated AI.
