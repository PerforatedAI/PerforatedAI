# Copyright (c) 2025 Perforated AI

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import sys
import numpy as np
import pdb
import os
import time
import warnings
from perforatedai import pb_globals as PBG
from perforatedai import pb_layer as PB
from perforatedai import pb_models as PBM
from perforatedai import pb_neuron_layer_tracker as PBT

import copy

from safetensors.torch import load_file
from safetensors.torch import save_file

# Main function to initialize the network to add dendrites
def initializePB(model, doingPB=True, saveName='PB', makingGraphs=True, maximizingScore=True, num_classes=10000000000, values_per_train_epoch=-1, values_per_val_epoch=-1, zoomingGraph=True):
    PBG.pbTracker = PBT.pb_neuron_layer_tracker(doingPB=doingPB,saveName=saveName)
    PBG.saveName = saveName
    model = PBG.pbTracker.initialize(model, doingPB=doingPB, saveName=saveName, makingGraphs=makingGraphs, maximizingScore=maximizingScore, num_classes=num_classes, values_per_train_epoch=-values_per_train_epoch, values_per_val_epoch=values_per_val_epoch, zoomingGraph=zoomingGraph)
    return model

# Get a list of all neuron_layers
def getPBModules(net, depth):
    allMembers = net.__dir__()
    thisList = []
    if issubclass(type(net),nn.Sequential) or issubclass(type(net),nn.ModuleList):
        for submoduleID, layer in net.named_children():
            # If there is a self pointer ignore it
            if net.get_submodule(submoduleID) is net:
                continue
            if type(net.get_submodule(submoduleID)) is PB.pb_neuron_layer:
                thisList = thisList + [net.get_submodule(submoduleID)]
            else:
                thisList = thisList + getPBModules(net.get_submodule(submoduleID), depth + 1)            
    else:
        for member in allMembers:
            if getattr(net,member,None) is net:
                continue
            if type(getattr(net,member,None)) is PB.pb_neuron_layer:
                thisList = thisList + [getattr(net,member)]
            elif issubclass(type(getattr(net,member,None)),nn.Module):
                thisList = thisList + getPBModules(getattr(net,member), depth+1)
    return thisList

# Get a list of all tracked_layers
def getTrackedModules(net, depth):
    allMembers = net.__dir__()
    thisList = []
    if issubclass(type(net),nn.Sequential) or issubclass(type(net),nn.ModuleList):
        for submoduleID, layer in net.named_children():
            if net.get_submodule(submoduleID) is net:
                continue
            if type(net.get_submodule(submoduleID)) is PB.tracked_neuron_layer:
                thisList = thisList + [net.get_submodule(submoduleID)]
            else:
                thisList = thisList + getTrackedModules(net.get_submodule(submoduleID), depth + 1)            
    else:
        for member in allMembers:        
            if getattr(net,member,None) is net:
                continue
            if type(getattr(net,member,None)) is PB.tracked_neuron_layer:
                thisList = thisList + [getattr(net,member)]
            elif issubclass(type(getattr(net,member,None)),nn.Module):
                thisList = thisList + getTrackedModules(getattr(net,member), depth+1)
    return thisList 

# Get all parameters from neuron_layers
def getPBModuleParams(net, depth):
    allMembers = net.__dir__()
    thisList = []
    if issubclass(type(net),nn.Sequential) or issubclass(type(net),nn.ModuleList):
        for submoduleID, layer in net.named_children():
            if type(net.get_submodule(submoduleID)) is PB.pb_neuron_layer:
                for param in net.get_submodule(submoduleID).parameters():
                    if(param.requires_grad):
                        thisList = thisList + [param]
            else:
                thisList = thisList + getPBModuleParams(net.get_submodule(submoduleID), depth + 1)            
    else:
        for member in allMembers:
            if(getattr(net,member,None) == net):
                continue  
            if type(getattr(net,member,None)) is PB.pb_neuron_layer:
                for param in getattr(net,member).parameters():
                    if(param.requires_grad):
                        thisList = thisList + [param]
            elif issubclass(type(getattr(net,member,None)),nn.Module):
                thisList = thisList + getPBModuleParams(getattr(net,member), depth+1)
    return thisList
def getPBNetworkParams(net):
    paramList = getPBModuleParams(net, 0)
    return paramList

# Replace a module with the module from globals list
def replacePredefinedModules(startModule):
    index = PBG.modulesToReplace.index(type(startModule))
    return PBG.replacementModules[index](startModule)

# Recursive function to do all conversion of modules to wrappers of modules
def convertModule(net, depth, nameSoFar, convertedList, convertedNamesList):
    if(PBG.verbose):
        print('calling convert on %s depth %d' % (net, depth))
        print('calling convert on %s: %s, depth %d' % (nameSoFar, type(net).__name__, depth))
    if((type(net) is PB.pb_neuron_layer)
       or type(net) is PB.tracked_neuron_layer):
        if(PBG.verbose):
            print('This is only being called because something in your model is pointed to twice by two different variables.  Highest thing on the list is one of the duplicates')
        return net
    allMembers = net.__dir__()
    if issubclass(type(net),nn.Sequential) or issubclass(type(net),nn.ModuleList):
        for submoduleID, layer in net.named_children():
            subName = nameSoFar + '.' + str(submoduleID)
            if(subName in PBG.moduleIDsToTrack):
                if(PBG.verbose):
                    print('Seq sub is in track IDs: %s' % subName)
                setattr(net,submoduleID,PB.tracked_neuron_layer(net.get_submodule(submoduleID), subName))
                continue
            if type(net.get_submodule(submoduleID)) in PBG.modulesToReplace:
                if(PBG.verbose):
                    print('Seq sub is in replacement module so replaceing: %s' % subName)
                setattr(net,submoduleID,replacePredefinedModules(net.get_submodule(submoduleID)))
            if (type(net.get_submodule(submoduleID)) in PBG.modulesToConvert
                or
                type(net.get_submodule(submoduleID)).__name__ in PBG.moduleNamesToConvert):
                if(PBG.verbose):
                    print('Seq sub is in conversion list so initing PB for: %s' % subName)
                if(issubclass(type(net.get_submodule(submoduleID)), torch.nn.modules.batchnorm._BatchNorm) or issubclass(type(net.get_submodule(submoduleID)), torch.nn.modules.instancenorm._InstanceNorm) or
                issubclass(type(net.get_submodule(submoduleID)), torch.nn.modules.normalization.LayerNorm)):
                    print('You have an unwrapped normalizaiton layer, this is not reccomended: ' + nameSoFar)
                    pdb.set_trace()
                setattr(net,submoduleID,PB.pb_neuron_layer(net.get_submodule(submoduleID), subName))
            else:
                if(net != net.get_submodule(submoduleID)):
                    convertedList += [id(net.get_submodule(submoduleID))]
                    convertedNamesList += [subName]
                    setattr(net,submoduleID,convertModule(net.get_submodule(submoduleID), depth + 1, subName, convertedList, convertedNamesList))
    else:
        for member in allMembers:
            subName = nameSoFar + '.' + member
            if(subName in PBG.moduleIDsToTrack):
                if(PBG.verbose):
                    print('Seq sub is in track IDs: %s' % subName)
                setattr(net,member,PB.tracked_neuron_layer(getattr(net,member),subName))
                continue
            if(id(getattr(net,member,None)) == id(net)):
                if(PBG.verbose):
                    print('member sub is a self pointer: %s' % subName)
                continue
            if(subName in PBG.moduleNamesToNotSave):
                if(PBG.verbose):
                    print('Skipping %s during convert' % subName)
                else:
                    if(subName == '.base_model'):
                        print('By default skipping base_model.  See \"Safetensors Errors\" section of customization.md to include it.')
                continue
            if(id(getattr(net,member,None)) in convertedList):
                print('The following module has a duplicate pointer within your model: %s' % subName)
                print('It is shared with: %s' % convertedNamesList[convertedList.index(id(getattr(net,member,None)))])
                print('One of these must be added to PBG.moduleNamesToNotSave (with the .)')
                sys.exit(0)
            try:
                getattr(net,member,None)
            except:
                continue
            if type(getattr(net,member,None)) in PBG.modulesToReplace:
                if(PBG.verbose):
                    print('sub is in replacement module so replaceing: %s' % subName)
                setattr(net,member,replacePredefinedModules(getattr(net,member,None)))
            if (type(getattr(net,member,None)) in PBG.modulesToConvert
                or
                type(getattr(net,member,None)).__name__ in PBG.moduleNamesToConvert):
                if(PBG.verbose):
                    print('sub is in conversion list so initing PB for: %s' % subName)
                setattr(net,member,PB.pb_neuron_layer(getattr(net,member),subName))
            elif (type(getattr(net,member,None)) in PBG.modulesToTrack
                or
                type(getattr(net,member,None)).__name__ in PBG.moduleNamesToTrack):
                if(PBG.verbose):
                    print('sub is in tracking list so initing tracked for: %s' % subName)
                setattr(net,member,PB.tracked_neuron_layer(getattr(net,member),subName))
            elif issubclass(type(getattr(net,member,None)),nn.Module):
                if(net != getattr(net,member)):
                    convertedList += [id(getattr(net,member))]
                    convertedNamesList += [subName]
                    setattr(net,member,convertModule(getattr(net,member), depth+1, subName, convertedList, convertedNamesList))
            if (issubclass(type(getattr(net,member,None)), torch.nn.modules.batchnorm._BatchNorm) or issubclass(type(getattr(net,member,None)), torch.nn.modules.instancenorm._InstanceNorm) or
                 issubclass(type(getattr(net,member,None)), torch.nn.modules.normalization.LayerNorm)):
                if(not PBG.unwrappedModulesConfirmed):
                    print('potentially found a norm Layer that wont be convereted, this is not reccomended: %s' % (subName))
                    print('Set PBG.unwrappedModulesConfirmed to True to skip this next time')
                    print('Type \'net\' + enter to inspect your network and see what the module type containing this layer is.')
                    print('Then do one of the following:')
                    print(' - Add the module type to PBG.moduleNamesToConvert to wrap it entirely')
                    print(' - If the norm layer is part of a sequential wrap it and the previous layer in a PBSequential')
                    print(' - If you do not want to add dendrites to this module add tye type to PBG.moduleNamesToTrack')
                    pdb.set_trace()
            else:
                if(PBG.verbose):
                    if(member[0] != '_' or PBG.extraVerbose == True):
                        print('not calling convert on %s depth %d' % (member, depth))            
    if(PBG.verbose):
        print('returning from call to: %s' % (nameSoFar)) 
    return net

# Function that calls the above and checks results
def convertNetwork(net, layerName=''):
    if type(net) in PBG.modulesToReplace:
        net = replacePredefinedModules(net)
    if((type(net) in PBG.modulesToConvert) or
        (type(net).__name__ in PBG.moduleNamesToConvert)):
        if(layerName == ''):
            print('converting a single layer without a name, add a layerName param to the call')
            sys.exit(-1)
        net = PB.pb_neuron_layer(net, layerName)
    else:
        net = convertModule(net, 0, '', [], [])
    missedOnes = []
    trackedOnes = []
    for name, param in net.named_parameters():
        wrapped = 'wrapped' in param.__dir__()
        if(wrapped):
            if(PBG.verbose):
                print('param %s is now wrapped' % (name))
        else:
            tracked = 'tracked' in param.__dir__()
            if(tracked):
                trackedOnes.append(name)
            else:
                missedOnes.append(name)
    if((len(missedOnes) != 0 or len(trackedOnes) != 0) 
       and PBG.unwrappedModulesConfirmed == False):
        print('\n------------------------------------------------------------------')
        print('The following params are not wrapped.\n------------------------------------------------------------------')
        for name in trackedOnes:
            print(name)
        print('\n------------------------------------------------------------------')
        print('The following params are not tracked or wrapped.\n------------------------------------------------------------------')
        for name in missedOnes:
            print(name)
        print('\n------------------------------------------------------------------')
        print('Modules that are not wrapped will not have Dendrites to optimize them')
        print('Modules modules that are not tracked can cause errors and is NOT reccomended')
        print('Any modules in the second list should be added to moduleNamesToTrack')
        print('------------------------------------------------------------------\nType \'c\' + enter to continue the run to confirm you do not want them to be refined')
        print('Set PBG.unwrappedModulesConfirmed to True to skip this next time')
        print('Type \'net\' + enter to inspect your network and see what the module types of these values are to add them to PGB.moduleNamesToConvert')
        import pdb; pdb.set_trace()
        print('confirmed')
    net.register_buffer('trackerString', torch.tensor([]))
    return net

# Helper function to convert a layer_tracker into a string and back to comply with safetensors saving
def stringToTensor(string):
    ords = list(map(ord, string))
    ords = torch.tensor(ords)
    return ords
def stringFromTensor(stringTensor):
    # Convert tensor to python list.
    ords = stringTensor.tolist()
    toReturn = ''
    # Doing block proceessing like this helps with memory errors
    while(len(ords) != 0):
        remainingOrds = ords[100000:]
        ords = ords[:100000]
        toAppend = ''.join(map(chr, ords))
        toReturn = toReturn + toAppend
        ords = remainingOrds
    return toReturn

def saveSystem(net, folder, name):
    if(PBG.verbose):
        print('saving system %s' % name)
    temp = stringToTensor(PBG.pbTracker.toString())
    if hasattr(net, 'trackerString'):
        net.trackerString = stringToTensor(PBG.pbTracker.toString()).to(next(net.parameters()).device)
    else:
        net.register_buffer('trackerString', stringToTensor(PBG.pbTracker.toString()).to(next(net.parameters()).device))
    # Before saving the tracker must be cleared to not contain pointers to the models modules
    oldList = PBG.pbTracker.PBNeuronLayerVector
    PBG.pbTracker.PBNeuronLayerVector = []

    saveNet(net, folder, name)
    
    PBG.pbTracker.PBNeuronLayerVector = oldList
    paiSaveSystem(net, folder, name)

def loadSystem(net, folder, name, loadFromRestart = False, switchCall=False):
    if(PBG.verbose):
        print('loading system %s' % name)
    net = loadNet(net, folder,name)
    PBG.pbTracker.resetLayerVector(net,loadFromRestart)

    PBG.pbTracker.fromString(stringFromTensor(net.trackerString))
    PBG.pbTracker.savedTime = time.time()    
    PBG.pbTracker.loaded=True
    PBG.pbTracker.memberVars['currentBestValidationScore'] = 0
    PBG.pbTracker.memberVars['epochLastImproved'] = PBG.pbTracker.memberVars['numEpochsRun']
    if(PBG.verbose):
        print('after loading epoch last improved is %d mode is %c' % (PBG.pbTracker.memberVars['epochLastImproved'], PBG.pbTracker.memberVars['mode']))
    # Saves always take place before the call to startEpoch so call it here when loading to correct off by 1 problems
    if not switchCall:
        PBG.pbTracker.startEpoch(internalCall=True)
    return net

def saveNet(net, folder, name):
    # If running a DDP only save with first thread
    if('RANK' in os.environ):
        if(int(os.environ["RANK"]) != 0):
            return
    if not os.path.isdir(folder):
        os.makedirs(folder)
    save_point = folder + '/'
    if not os.path.isdir(save_point):
        os.mkdir(save_point)
    for param in net.parameters(): param.data = param.data.contiguous()
    if(PBG.usingSafeTensors):
        save_file(net.state_dict(), save_point + name + '.pt')
    else:
        torch.save(net, save_point + name + '.pt')

def loadNet(net, folder, name):
    save_point = folder + '/'
    if(PBG.usingSafeTensors):
        stateDict = load_file(save_point + name + '.pt')
    else:
        #Different versions of torch require this change
        try:
            stateDict = torch.load(save_point + name + '.pt', map_location=torch.device('cpu'), weights_only=False).state_dict()
        except:
            stateDict = torch.load(save_point + name + '.pt', map_location=torch.device('cpu')).state_dict()
    return loadNetFromDict(net, stateDict)
    
def loadNetFromDict(net, stateDict):
    pbModules = getPBModules(net,0)
    if(pbModules == []):
        print('PAI loadNet and loadSystem uses a state_dict so it must be called with a net after initializePB has been called')
        sys.exit()
    for module in pbModules:
        # Set up name to be what will be saved in the state dict
        moduleName = module.name
        # This should always be true
        if moduleName[0] == '.':
            #strip "."
            moduleName = moduleName[1:]
        # If it was a dataparallel it will also have a module at the start so strip that for loading
        if moduleName[:6] == 'module':
            moduleName = moduleName[7:]
        module.clearDendrites()
        for tracker in module.pb.pbValues:
            try:
                tracker.setupArrays(len(stateDict[moduleName + '.pb.pbValues.0.shape']))
            except Exception as e:
                print(e)
                print('When missing this value it typically means you converted a module but didn\'t actually use it in your forward and backward pass')
                print('module was: %s' % moduleName)
                print('check your model definition and forward function and ensure this module is being used properly')
                print('or add it to PBG.moduleNamesToSkip to leave it out of conversion')
                print('This can also happen if you adjusted your model definition after calling intitializePB')
                print('for example with torch.compile.  If the module name printed above does not contain all modules leading to the main definition')
                print('this is likely the case for your problem. Fix by calling initializePB after all other model initialization steps')
                import pdb; pdb.set_trace()
                
        # Perform as many cycles as the state dict has
        numCycles = int(stateDict[moduleName + '.pb.numCycles'].item())
        if(numCycles > 0):
            simulateCycles(module, numCycles, doingPB = True)    
    if hasattr(net, 'trackerString'):
        net.trackerString = stateDict['trackerString']
    else:
        net.register_buffer('trackerString', stateDict['trackerString'])
    net.load_state_dict(stateDict)
    net.to(PBG.device)
    return net

def paiSaveSystem(net, folder, name):
    net.memberVars = {}
    for memberVar in PBG.pbTracker.memberVars:
        if memberVar == 'schedulerInstance' or memberVar == 'optimizerInstance':
            continue
        net.memberVars[memberVar] = PBG.pbTracker.memberVars[memberVar]
    paiSaveNet(net, folder, name)

def deepCopyPAI(net):
    PBG.pbTracker.clearAllProcessors()
    return copy.deepcopy(net)

def countParams(net):
    return sum(p.numel() for p in net.parameters())

# For open source implimentation just use regular saving for now
# This function removes extra scaffolding that open source version already has minimal values for
def paiSaveNet(net, folder, name):
    return

# Simulate the back and forth processes of adding dendrites to build a pretrained dendrite model before loading weights
def simulateCycles(module, numCycles, doingPB):
    checkSkipped = PBG.checkedSkippedLayers
    if(doingPB == False):
        return
    PBG.checkedSkippedLayers = True
    mode = 'n'
    for i in range(numCycles):
        if(mode == 'n'):
            module.setMode('p')
            module.addPBLayer()
            mode = 'p'
        else:
            module.setMode('n')
            mode = 'n'
    PBG.checkedSkippedLayers = checkSkipped

'''
High level steps for entire system to switch back and forth between neuron learning and dendrite learning
'''
def changeLearningModes(net, folder, name, doingPB):    
    # If not doing PB this just allows training to continue longer with flags every time early stopping should be occuring
    if(doingPB == False):
        PBG.pbTracker.memberVars['switchEpochs'].append(PBG.pbTracker.memberVars['numEpochsRun'])
        PBG.pbTracker.memberVars['lastSwitch'] = PBG.pbTracker.memberVars['switchEpochs'][-1]
        PBG.pbTracker.resetValsForScoreReset()
        return net
    if(PBG.pbTracker.memberVars['mode'] == 'n'):
        currentEpoch = PBG.pbTracker.memberVars['numEpochsRun']
        overWrittenEpochs = PBG.pbTracker.memberVars['overWrittenEpochs']
        overWrittenExtra = PBG.pbTracker.memberVars['extraScores']
        if(PBG.drawingPB):
            overWrittenVal = PBG.pbTracker.memberVars['accuracies']
        else:
            overWrittenVal = PBG.pbTracker.memberVars['nAccuracies']
        '''
        The only reason that retainAllPB should ever be used is to test GPU memory and 
        configuration.  So if true don't load the best system because it will delete dendrites if 
        the previous best was better than the current best
        '''
        if(not PBG.retainAllPB):
            if(not PBG.silent):
                print('Importing best Model for switch to PB...')
            net = loadSystem(net, folder, name, switchCall=True)
        else:
            if(not PBG.silent):
                print('Not importing new model since retaining all PB')
        PBG.pbTracker.setPBTraining()        
        PBG.pbTracker.memberVars['overWrittenEpochs'] = overWrittenEpochs
        PBG.pbTracker.memberVars['overWrittenEpochs'] += currentEpoch - PBG.pbTracker.memberVars['numEpochsRun']
        PBG.pbTracker.memberVars['totalEpochsRun'] = PBG.pbTracker.memberVars['numEpochsRun'] + PBG.pbTracker.memberVars['overWrittenEpochs']
        
        if(PBG.saveOldGraphScores):
            PBG.pbTracker.memberVars['overWrittenExtras'].append(overWrittenExtra)
            PBG.pbTracker.memberVars['overWrittenVals'].append(overWrittenVal)
        else:
            PBG.pbTracker.memberVars['overWrittenExtras'] = [overWrittenExtra]
            PBG.pbTracker.memberVars['overWrittenVals'] = [overWrittenVal]
        if(PBG.drawingPB):
            PBG.pbTracker.memberVars['nswitchEpochs'].append(PBG.pbTracker.memberVars['numEpochsRun'])
        else:
            if(len(PBG.pbTracker.memberVars['switchEpochs']) == 0):
                PBG.pbTracker.memberVars['nswitchEpochs'].append(PBG.pbTracker.memberVars['numEpochsRun'])
            else:
                PBG.pbTracker.memberVars['nswitchEpochs'].append(PBG.pbTracker.memberVars['nswitchEpochs'][-1] + ((PBG.pbTracker.memberVars['numEpochsRun'])-(PBG.pbTracker.memberVars['switchEpochs'][-1])))
            
        PBG.pbTracker.memberVars['switchEpochs'].append(PBG.pbTracker.memberVars['numEpochsRun'])
        PBG.pbTracker.memberVars['lastSwitch'] = PBG.pbTracker.memberVars['switchEpochs'][-1]

        # Because open source version is only doing neuron training for gradient descent dendrites, switch back to n mode right away
        if(PBG.noExtraNModes):
            net = changeLearningModes(net, folder, name, doingPB)
    else:
        if(not PBG.silent):
            print('Switching back to N...')
        setBest = PBG.pbTracker.memberVars['currentNSetGlobalBest']
        PBG.pbTracker.setNormalTraining()
        if(len(PBG.pbTracker.memberVars['pswitchEpochs']) == 0):
            PBG.pbTracker.memberVars['pswitchEpochs'].append(((PBG.pbTracker.memberVars['numEpochsRun']-1)-(PBG.pbTracker.memberVars['switchEpochs'][-1])))
        else:
            PBG.pbTracker.memberVars['pswitchEpochs'].append(PBG.pbTracker.memberVars['pswitchEpochs'][-1] + ((PBG.pbTracker.memberVars['numEpochsRun'])-(PBG.pbTracker.memberVars['switchEpochs'][-1])))
        PBG.pbTracker.memberVars['switchEpochs'].append(PBG.pbTracker.memberVars['numEpochsRun'])
        PBG.pbTracker.memberVars['lastSwitch'] = PBG.pbTracker.memberVars['switchEpochs'][-1]
        # Will be false for open source implimentation
        if(PBG.retainAllPB or (PBG.learnPBLive and setBest)):
            if(not PBG.silent):
                print('Saving model before starting normal training to retain PBNodes regardless of next N Phase results')
            saveSystem(net, folder, name)
            
    # Track parameter counts for each architecture
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    PBG.pbTracker.memberVars['paramCounts'].append(pytorch_total_params)
    return net

