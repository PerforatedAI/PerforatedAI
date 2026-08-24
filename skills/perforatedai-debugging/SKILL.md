# Perforated AI — Debugging Reference

Full error-by-error reference. Find your error in the index in `../SKILL.md`, then
jump to the matching section below. Every section follows: **symptom / error string →
cause → fix**.

Cross-references to "customization" (e.g. "section 3 from customization", "1.2 from
customization", "the DataParallel section") point to the separate Perforated AI
customization readme. If you don't have it loaded, get it before applying fixes that
depend on it.

## Contents
- [New best scores are not being triggered](#new-best-scores-are-not-being-triggered)
- [Errors where warnings are printed](#errors-where-warnings-are-printed)
- [Input dimensions](#input-dimensions)
- [Broadcast shape](#broadcast-shape)
- [Errors in forward](#errors-in-forward)
- [Exponential Moving Average (EMA)](#exponential-moving-average-ema)
- [Errors in filterBackward](#errors-in-filterbackward)
- [DDP gradient errors](#ddp-gradient-errors)
- [dtype errors](#dtype-errors)
- [Device errors](#device-errors)
- [AttributeError on PAINeuronLayer](#attributeerror-on-paineuronlayer)
- [Initialize error](#initialize-error)
- [Path error during addValidationScore](#path-error-during-addvalidationscore)
- [setupOptimizer error](#setupoptimizer-error)
- [learning_rate error](#learning_rate-error)
- [Parameter not wrapped](#parameter-not-wrapped)
- [Unpacking a non-existing scheduler](#unpacking-a-non-existing-scheduler)
- [Optimizer loop error](#optimizer-loop-error)
- [Size mismatch](#size-mismatch)
- [Initialization errors (pb_layer __init__)](#initialization-errors-pb_layer-__init__)
- [Index out of range in saveGraphs](#index-out-of-range-in-savegraphs)
- [Things not getting converted](#things-not-getting-converted)
- [DeepCopy error](#deepcopy-error)
- [Different devices](#different-devices)
- [Memory leak](#memory-leak)
- [Memory issues inside Docker but not outside](#memory-issues-inside-docker-but-not-outside)
- [Optimizer initialization error](#optimizer-initialization-error)
- [Debugging Docker installation](#debugging-docker-installation)
- [Saving PAI](#saving-pai)
- [Errors that are currently not fixable](#errors-that-are-currently-not-fixable)
- [Extra debugging](#extra-debugging)

---

## New best scores are not being triggered

No error string — new best scores just never register.

- Check `GPA.pc.get_improvement_threshold()`. If the improvement is very small it may
  not be beating the previous best by a high enough margin. Set
  `GPA.pc.set_verbose(True)` to check whether this is the case.
- Ensure `maximizing_score` is set properly in `perforate_model`. If you are maximizing
  an accuracy score it should be `True`; if you are minimizing a loss score it should be
  `False`.

## Errors where warnings are printed

    "The following layer has not properly set this_output_dimensions"

Check the suggestions that are printed and section 4 in customization.

    Didn't get any non zero scores or a score is nan or inf.

The Dendrites learned a correlation that was either nan or infinite. We have only seen
this happen with training pipelines where the neurons are also learning weights that are
getting close to triggering an inf overflow error themselves. See if you can add
normalization layers to keep your weights within more usual ranges.

    An entire layer got exactly 0 Correlation

Same as above, but for zero.

    Trying to call backwards but module X wasn't PAIified

Something went wrong with the conversion. The module is getting triggered for PAI
modifications but was not converted in a way that allowed it to initialize properly. Look
into how you set up that layer.

    Need exactly one 0 in the input dimensions

You set your input dimensions but it wasn't the proper -1s and a single zero as it should
be.

## Input dimensions

    'pbValueTracker' object has no attribute 'out_channels'

Look at section 3 from customization. This explains how to set input dimensions.

## Broadcast shape

    Values[0].normalPassAverageD += (val.sum(mathTuple) * 0.01) / fullMult
    RuntimeError: output with shape [X] doesn't match the broadcast shape [Y]

Input dimensions were not properly set. Run again with pdb and when this error comes up
print `Values[0].layerName` to see which layer the problem is with. You can also print the
shape of `val` to see what the dimensions are supposed to be. This should be caught
automatically, so in our experience when this happens it means you have a layer that can
accept tensors which have multiple dimensionalities without having problems. This is not
accounted for with our software currently, so wrap that layer in a module as required so
you don't need to do that.

## Errors in forward

These usually mean the processors were not set up correctly. Look at 1.2 from
customization.

    AttributeError: 'tuple' object has no attribute 'requires_grad'

This specifically means you are returning a tuple of tensors rather than a single tensor.
Your processor needs to tell you how to handle this so the Dendrite only collaborates on
one tensor with the neuron.

Make sure you put the `GPA.pc.get_modules_with_processing()` setup before the call to
`convertNetwork`.

## Exponential Moving Average (EMA)

If you have any problems with EMA it is because EMA keeps a shadow copy of the model you
are working with. This has to be created after `perforate_model` has been called. It must
also be reinitialized after each time `restructured` returns `True`.

## Errors in filterBackward

There are a couple of errors that can happen in the `filterBackward` function.

    AttributeError: 'NoneType' object has no attribute 'detach'

This also usually means the processors were not set up correctly. Look at 1.2 from
customization. It means you are not using the tensor that is being passed to the dendrites.
For example, if you are using the default LSTM processor but using `hidden_state` rather
than `output` from `output, (hidden_state, cell_state)`.

    AttributeError: 'pbValueTracker' object has no attribute 'normalPassAverageD'

The `pbValueTracker` was not properly initialized. This can happen for two reasons:

1. You are running on multiple GPUs. With a single GPU, `pbValueTracker`s are set up
   automatically, but when running on multiple GPUs this has to be set up by hand using
   the `saveTrackerSettings` and `initializeTrackerSettings` functions. Look at the
   DataParallel section from the customization readme.
2. You initialize a Dendrite layer but then don't actually use it — i.e., it is not being
   called in the forward and backward pass of the network. In these cases, look into your
   forward functions and track down why the layer is not properly being used. This same
   effect can take place if you try to add a set of dendrites before performing any
   training. With our system you should not run initial validation epochs before starting
   training, or if you do, make sure not to add Dendrites during those cycles.

## DDP gradient errors

    RuntimeError: Encountered gradient which is undefined, but still allreduced by DDP reducer.

This occurs when using DistributedDataParallel (DDP) with PerforatedAI. It happens because
PerforatedAI's selective training (Cascade Correlation) means some parameters don't receive
gradients during `backward()`, but DDP expects all parameters to have gradients for
allreduce.

**Fix:** Pre-initialize all parameter gradients to zeros AFTER `optimizer.zero_grad()` and
BEFORE `loss.backward()`:

```python
optimizer.zero_grad()

# Pre-initialize gradients for DDP compatibility
if args.distributed:
    for param in model.parameters():
        if param.requires_grad and param.grad is None:
            param.grad = torch.zeros_like(param)

loss.backward()
optimizer.step()
```

This ensures DDP's allreduce doesn't encounter None gradients while still allowing
PerforatedAI's selective training to work correctly.

## dtype errors

Anything like the following:

    (was torch.cuda.HalfTensor got torch.cuda.FloatTensor)
    Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same

If you are not working with float data, change `GPA.pc.get_d_type()` to whatever you are
using, e.g.:

    GPA.pc.set_d_type(torch.double)

## Device errors

    RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!

There is a setting that defaults to using what is available. If cuda is available but you
still don't want to use it, call:

    GPA.pc.set_device('cpu')

Additionally, if you are running on mac or generally using any device that will not be
properly set with the call of `device = torch.device("cuda" if use_cuda else "cpu")`, set
your device as `GPA.pc.set_device(your device type)`.

    AttributeError: 'DendriteValueTracker' object has no attribute 'device'

This can be caused when, after restructuring, there is an `optimizer.step()` before a
`forward()` and `backward()`. We have seen this automatically happen in the skorch
framework. It can be solved in that situation by passing the first datapoint through:

    # Do a dummy forward+backward pass to populate gradients
    iterator = net.get_iterator(dataset_train, training=True)
    Xi, yi = next(iter(iterator))
    net.optimizer_.zero_grad()
    y_pred = net.infer(Xi)
    loss = net.get_loss(y_pred, yi, X=Xi, training=True)
    loss.backward()
    print("PAI: Dummy forward/backward completed")
    net.optimizer_.zero_grad()
    print("PAI: Model restructured and optimizer reinitialized")

We have also seen this happen if gradient checkpointing is enabled. Turn this off for
Perforated AI usage.

## AttributeError on PAINeuronLayer

    AttributeError: 'PAINeuronLayer' object has no attribute 'SOMEVARIABLE'

This is the error you get if you need to access an attribute of a module that is now
wrapped as a `PAINeuronLayer`. Make the following change:

    #model.yourModule.SOMEVARIABLE
    model.yourModule.mainModule.SOMEVARIABLE

This is done automatically for most variables. But for special variables and functions that
start with a `_`, python does not do it for you. In these cases, if the above is not easily
accessible because it is in the backend of a library you are working with, you can also do
the following after `perforate_model`:

    UPA.apply_method_delegation_to_model(model, "method_name", module_type)

## Initialize error

    AttributeError: 'list' object has no attribute 'set_optimizer_instance'

The `pai_tracker` was not initialized. This generally only happens if zero layers are
converted. Make sure at least one layer has been converted correctly.

## Path error during addValidationScore

    TypeError: stat: path should be string, bytes, os.PathLike or integer, not NoneType

You entered `None` for the `saveName`, likely because `args.saveName` did not have a default
value.

## setupOptimizer error

    TypeError: 'list' object is not callable

If you get this in `setupOptimizer`, it means you called `setupOptimizer` but did not call
`setOptimizer`. Call that first.

## learning_rate error

    UnboundLocalError: cannot access local variable 'learning_rate' where it is not associated with a value

This happens if your optimizer does not have any params it is optimizing for. This can be
the case when you are in 'p' mode during perforated learning but none of the modules are
actually being perforated. Similarly, if you are using multiple optimizers and you passed an
optimizer to the `pai_tracker` which is not the one actually pointing to the perforated
module, this error will come up.

## Parameter not wrapped

    WARNING: Parameter does not have parameter_type attribute in n mode
    You can find this param by going up in the stack and calling:
    UPA.find_param_name_by_id(model,124630993409104)

There is a module that was not properly handled during the call to `perforate_model`. It is
generally one of two things:

1. Confirm that all of your modules are either tracked or perforated.
2. Confirm that you are not changing anything about the model definition after
   `perforate_model`. For example, if you are doing transfer learning and replacing your fc
   layer, make sure to replace it before the call to `perforate_model`.

In some cases this happens if you have a module that can't be perforated or tracked because
one of its submodules will be. If this is from a different submodule, just perforate or
track that one. However, if a module has both submodules and also raw `torch.Parameter`s,
you can add those by id with a call to `GPA.pc.append_parameter_ids_to_track` with the id
that gets printed after you go up in the pdb trace that flagged this warning and call
`find_param_name_by_id` as instructed.

Another common cause with external training libraries is that everything is properly tagged
right after `perforate_model`, but then the library recreates or reloads parameters before
training begins (for example during Trainer setup, wrapping, or optimizer preparation). In
this case `parameter_type` attributes can be lost and this warning appears at the first
training step. If this happens, move `perforate_model` so it is called immediately before
the function that starts training (or immediately before the library function that finalizes
the trainable model state). In short: avoid any model mutation, weight loading, wrapping, or
replacement steps between `perforate_model` and training start.

If running on XLA, this can also be caused if the model is perforated before moving to XLA.
When moving to cuda the same parameter variable is used, but on XLA it is a new variable that
copies the values, but not our values. To fix, move the model to the XLA device before
calling `perforate_model()`.

## Unpacking a non-existing scheduler

    optimizer, _ = GPA.pai_tracker.setup_optimizer(model, optimArgs, None)
        ^^^^^^^^^^^^
    TypeError: cannot unpack non-iterable _____ object

Coding assistants will sometimes call `setup_optimizer` like this when there is no scheduler.
When there is no scheduler, only `optimizer` is returned. You are not ignoring a second
return variable — there isn't one. Delete that `_` or python will try to unpack your
optimizer into two values.

## Optimizer loop error

    RuntimeError: For non-complex input tensors, argument alpha must not be a complex number.

This can come up for some schedulers when the loop actually goes longer than the scheduler's
settings for the max number of epochs. If you want to keep the current scheduler, you must
use the mode to switch on fixed epoch counts rather than waiting for a plateau. E.g.:

    # Use fixed-epoch switch mode: add dendrites every 80 epochs
    GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
    GPA.pc.set_fixed_switch_num(original count)
    GPA.pc.set_first_fixed_switch_num(original count)

## Size mismatch

    File "perforatedai/pb_layer.py", line X, in perforatedai.pb_layer.pb_neuron_layer.forward
    RuntimeError: The size of tensor a (X) must match the size of tensor b (X) at non-singleton dimension

Your neurons are not correctly matched in `setoutput_dimensions`. If your 0 is in the wrong
index, the tensors used for tracking the Dendrite-to-Neuron weights will be the wrong size.

## Initialization errors (pb_layer __init__)

    File "perforatedai/pb_layer.py" ... perforatedai.pb_layer.pb_neuron_layer.__init__
    IndexError: list index out of range

You did something wrong with the processing classes. We have seen this before when
`moduleNamesWithProcessing` and `moduleByNameProcessingClasses` don't line up. They need to
be added in order in both arrays, and if the module is "by name" the processor also has to be
added to the "by name" array.

## Index out of range in saveGraphs

    perforatedai.pb_neuron_layer_tracker.pb_neuron_layer_tracker.saveGraphs
    IndexError: list index out of range

You likely added the validation score before the test score. Test scores must be added before
the validation score, since graphs are generated when the validation score is added and the
tracker must have access to the test scores at that time.

    [rank0]: IndexError: list index out of range

Similarly, especially within a DDP system, this can be caused when a switch has just happened
but 'latest' is loaded instead of `switch_x`. 'latest' currently has a bug where it does not
have correct records for that single epoch.

## Things not getting converted

The conversion script runs by going through all member variables and determining all that
inherit from `nn.Module`. If you have any lists or non-`nn.Module` variables that then have
`nn.Module`s in them, it will miss them. If you have a list, put that list into an
`nn.ModuleList` and it will then find everything. If you do this, make sure you replace the
original variable name, because that is what will be used. If you use the `add_module`
function, this is a sign you might cause this sort of problem. We do not currently have a
workaround for non-module objects that contain module objects — let us know if that is a
situation you are in and there is a reason the top object can't also be a module.

## DeepCopy error

    RuntimeError: Only Tensors created explicitly by the user (graph leaves) support the deepcopy protocol at the moment.  If you were attempting to deepcopy a module, this may be because of a torch.nn.utils.weight_norm usage, see https://github.com/pytorch/pytorch/pull/103001

This has been seen when the processor doesn't properly clear the values it has saved. Make
sure you define a `clear_processor` function for any processors you create. If you believe
you did and are still getting this error, reach out to us.

This can also happen when `forward` is called to accumulate gradients but then `backward` is
not called to clear those gradients. Set `GPA.pc.set_extra_verbose(True)` to print when
gradient tensors are added and removed. If they are being added but not removed, this is the
cause. Check that `optimizer.step()` is being called properly. Some programs have methods
that do not call `optimizer.step()` under certain situations. Our code is also set up such
that `optimizer.zero_grad` will correct this, which can be called before `add_validation_score`
as an alternative if the optimizer should actually not be stepped.

If this does not seem to be the problem, go up in the debugger and call deepcopy on individual
modules and submodules to track down which module is causing the problem.

## Different devices

- Check whether you have a Parameter being set to a device inside the `init` function. This
  seems to cause a problem with calling `to()` on the main model.
- Check whether you are calling `to()` on a variable inside the `forward()` function. Don't
  do this — put it on the right device before passing it in.

## Memory leak

A memory leak is happening if you run out of memory in the middle of a training epoch (it had
enough memory for the first batch but a later batch crashes with an OOM error). These are
always a pain to debug, but here are some we have caught:

- Check whether one of your layers is not being cleared during backwards. This can build up if
  you are forwarding a module but not calling backwards, even though this won't cause a leak
  without PAI in the same model. We have seen a handful of models which calculate values but
  then never actually use them for anything that goes towards calculating loss, so avoid that.
  To check for this you can use: `GPA.pc.set_debugging_memory_leak(True)`
- Check whether you are using `model.zero_grad` rather than `optimizer.zero_grad`. The current
  system requires optimizer.
- If this is happening in the validation/test loop after safely completing the train loop,
  make sure you are in `eval()` mode, which does not have a backwards pass.
- Check your training loop for any tensors being tracked during the loop which would not be
  cleared every time. One we have seen often is a cumulative loss being tracked. Without PAI
  this gets cleared appropriately, but with PAI it does not. Fix by adding `.detach()` or
  `.item()` before the loss is added to the cumulative variable. This can also sometimes be
  indirect, such as using a `MulticlassJaccardIndex` in PyTorch Lightning which tracks stats
  over multiple batches.
- Try removing various blocks of your model or components of your full training process to
  track down exactly which component is causing the problem. If you can narrow it to exactly
  which line causes a leak with and without it present, we can help debug why that line is
  causing problems if it is on our side.

### Slow memory leak debugging

If the above does not work, use the following to try to find where the count goes up when it
shouldn't, then review the section above to see if you may be doing something wrong on that
line. Sometimes the count fluctuates for other reasons, so find the places where the eventual
upticks happen consistently:

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

## Memory issues inside Docker but not outside

If you are running with a docker container and you were not using docker before, it is likely
an issue with the shared memory inside the docker container. Run with the additional
`--shm-size` flag:

    docker run --gpus all -i --shm-size=8g -v .:/pai -w /pai -t pai /bin/bash

## Optimizer initialization error

    -optimizer = self.member_vars['optimizer'](**optArgs)
    -TypeError: __init__() got an unexpected keyword argument 'momentum'

This can happen if you are using more than one optimizer in your program. If you are, call
`GPA.pai_tracker.setOptimizer()` again when you switch to the second optimizer, and also call
it as the first line in the `if(restructured)` block for adding validation scores.

## Debugging Docker installation

    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
    >>> solved with:
    sudo apt-get install libgl1-mesa-glx

## Saving PAI

PAI saves the entire system rather than just your model.

### Parameterized modules error

    RuntimeError: Serialization of parametrized modules is only supported through state_dict()

You have a parameterized module. Track it down by running in pdb and calling `torch.save` on
each module of your model recursively until you get to the smallest module that flags the
error. Whatever that one is will have to be changed so you can call `torch.save`. We have seen
this happen in a model that used to work because of an updated version of pytorch, so
downgrading to a pre-2.0 torch version may fix it. Using safeTensors should resolve this, as
this error only seems to come up when `using_safe_tensors` is `False`.

### Pickle errors

    Can't pickle local object 'train.<locals>.tmp_func'

The optimizer or scheduler are likely using lambda functions. Replace the lambda with a
defined function, e.g.:

    lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    #converted to a global function
    def tmp_func(x):
        return (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']
    lf = tmp_func #where it was originally defined

### Autograd errors

#### Second backwards

    Trying to backward through the graph a second time

Caused by something in your graph containing the same tensor twice. Try to track it down with
the following. Set this up and then call
`from perforatedai import globals_perforatedai as GPA; GPA.get_param_name(t_outputs)` within
the error block. If this does not work, try filling in `GPA.param_name_by_id` with additional
tensors:

    def get_param_name(tensor):
        return GPA.param_name_by_id.get(id(tensor), None)
    # Create mapping from tensor id to name
    GPA.param_name_by_id = {id(param): name for name, param in model.named_parameters()}
    GPA.get_param_name = get_param_name

It can also help to use the torchviz package to show the entire graph of the tensor. Go up in
the debugger to where the problem first occurs in your code, then call:

    from torchviz import make_dot; dot = make_dot(TENSOR); dot.render('graph', format='pdf')

#### Inplace operations

    RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [128, 1280]], which is output 0 of ReluBackward0, is at version 1; expected version 0 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient, with torch.autograd.set_detect_anomaly(True).

This can happen any time the forward is using `+=` type functions. Use `var = var + var2`
instead. In some modules like dropout this is a setting: `nn.Dropout(p=dropout, inplace=False)`.

### Safetensors errors

    Some tensors share memory, this will lead to duplicate memory on disk and potential differences when loading them again:

    Then a really long list of pairs

    A potential way to correctly save your model is to use `save_model`.
    More information at https://huggingface.co/docs/safetensors/torch_shared_tensors

A lot of modern models tend to save a pointer to a copy of themselves, which causes an error
with the Perforated AI save function. This can be remedied in two ways.

First, use `torch.load` rather than the safetensors method. Be aware there is risk in loading
pretrained model files from outside your group; this should only be used with models you trust
or models training from scratch. To accept this risk and use `torch.load`:

    GPA.pc.set_using_safe_tensors(False)

However, this will sometimes cause the Parameterized Modules Error above. In these cases,
another alternative is to choose which of the modules is causing the error and add it to
`GPA.pc.get_module_names_to_not_save()`. It will likely not be either of the exact names in
the pair list, and you will have to find the PAI name for it. This is often just removing the
first "model" string before the first "." but including that ".". This sets the save function
to ignore the copy. We already include the following by default:

    GPA.pc.append_module_names_to_not_save(['.base_model'])

To remove this default value, if you are using a base_model module which is not a duplicate,
you must clear this array.

In newer versions of safetensors the following should also work:

    import safetensors
    from collections import defaultdict
    def _ignore_shared_tensors(state_dict):
        tensors = defaultdict(set)
        return tensors
    safetensors.torch._find_shared_tensors = _ignore_shared_tensors

#### Weight tying

In some cases this is done intentionally with weight tying, which is not just a duplicate
pointer but also a known issue where multiple modules actually use the same weight tensor in
their forward. We have a workaround, but it is only experimental for now, so your results may
vary:

    GPA.pc.set_using_safe_tensors(True)
    GPA.pc.set_weight_tying_experimental(True)

### Other loading errors

    KeyError: 'moduleName.mainModule.numCycles'

This can be caused by a few different reasons:
1. Calling `intializePB` before `loadPAIModel`. This function should be called on a baseline
   model, not a PAIModel.
2. Your model definition, or `modules_to_perforate` and `moduleNamesToConvert` lists, are
   different between your training script and your inference script.

<br>

    Getting a warning with modules not tracked or wrapped with main_module in the list

You are trying to perforate a model that has already been perforated. This should never be
done.

## Errors that are currently not fixable

### Loss scaling

Functions such as `ApexScaler` or `NativeScaler` from `timm.utils` can cause:

    pytorch RuntimeError "has changed the type of value"

These functions are applied after the computation graph is created from the forward pass and
types within both PB and the original model are set. When they make adjustments to tensors
within the model, they do not make the equivalent changes to the tensors in the PB version of
the model. At the moment there is no workaround, so if you encounter this error you have to
turn off loss scaling.

### AMP — no inf checks were recorded

    AssertionError: No inf checks were recorded for this optimizer.

This can come up when amp is used with perforatedbp. The problem is that
`set_optimizer_instance` is doing things during `optimizer.step()` that `scalar.step(optimizer)`
doesn't activate. The workaround for now is to not use AMP during 'p' learning mode, such as
with the block below:

    if scaler is not None and GPA.pai_tracker.member_vars['mode'] != 'p':
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

Or if you are using a library like pytorch_lightning:

    # Set AMP scaler based on PAI mode before strategy setup
    if hasattr(self.trainer.strategy, 'precision_plugin') and hasattr(self.trainer.strategy.precision_plugin, 'scaler'):
        if GPA.pai_tracker.member_vars['mode'] == 'p':
            # Disable scaler for perforated mode to avoid issues with restructured model
            self.trainer.strategy.precision_plugin.scaler = torch.cuda.amp.GradScaler(enabled=False)
        else:  # mode == 'n'
            # Re-enable scaler for normal mode
            self.trainer.strategy.precision_plugin.scaler = torch.cuda.amp.GradScaler(enabled=True)

### Centered RMSprop causing nan

We are aware that with RMSprop, `centered = True` can cause correlations to be calculated as
nan. For now, set the setting to not be centered or pick an alternative optimizer.

## Extra debugging

If you are unable to debug things, feel free to contact us. We are happy to help you work
through issues and get running with Perforated AI.
