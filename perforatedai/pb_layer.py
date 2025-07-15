# Copyright (c) 2025 Perforated AI

import torch
import torch.nn as nn
import math
import sys
import numpy as np
import pdb
import os 

import time

from datetime import datetime
from perforatedai import pb_globals as PBG
from perforatedai import pb_models as PBM
from perforatedai import pb_neuron_layer_tracker as PBT
from perforatedai import pb_utils as PBU
import copy

# Values for Dendrite training, minimally used in open source version
dendriteTensorValues = ['shape'] # Shape is simply a tensor of the same shape as the total neurons in the layer
dendriteSingleValues = []

dendriteInitValues = ['initialized',
                      'currentDInit']

# Values for reinitializing and saving dendrite scaffolding
dendriteReinitValues = dendriteTensorValues + dendriteSingleValues
dendriteSaveValues = dendriteTensorValues + dendriteSingleValues + dendriteInitValues

valueTrackerArrays = ['pbOuts']
    
def filterBackward(grad_out, Values, candidateNonlinearOuts):
    
    if(PBG.extraVerbose):
        print('%s calling backward' % Values[0].layerName)

    with torch.no_grad():
        val = grad_out.detach()
        # If the input dimentions are not initialized
        if(not Values[0].currentDInit.item()):    
            # If the input dimentions and gradient don't have the same shape trigger an error and quit        
            if(len(Values[0].thisInputDimensions) != len(grad_out.shape)):
                print('The following layer has not properly set thisInputDimensions')
                print(Values[0].layerName)
                print('it is expecting:')
                print(Values[0].thisInputDimensions)
                print('but recieved')
                print(grad_out.shape)
                print('to check these all at once set PBG.debuggingInputDimensions = 1')
                print('Call setThisInputDimensions on this layer after initializePB')
                if(not PBG.debuggingInputDimensions):
                    sys.exit(0)
                else:
                    PBG.debuggingInputDimensions = 2
                    return
            # Make sure that the input dimentions are correct
            for i in range(len(Values[0].thisInputDimensions)):
                if(Values[0].thisInputDimensions[i] == 0):
                    continue
                # Make sure all input dimensions are either -1 (new format) or exact values (old format)
                if(not (grad_out.shape[i] == Values[0].thisInputDimensions[i])
                    and not Values[0].thisInputDimensions[i] == -1):
                    print('The following layer has not properly set thisInputDimensions with this incorrect shape')
                    print(Values[0].layerName)
                    print('it is expecting:')
                    print(Values[0].thisInputDimensions)
                    print('but recieved')
                    print(grad_out.shape)
                    print('to check these all at once set PBG.debuggingInputDimensions = 1')
                    if(not PBG.debuggingInputDimensions):
                        sys.exit(0)
                    else:
                        PBG.debuggingInputDimensions = 2
                        return
            # Setup the arrays with the now known shape
            with(torch.no_grad)():
                if(PBG.verbose):
                    print('setting d shape for')
                    print(Values[0].layerName)
                    print(val.size())
                
                Values[0].setOutChannels(val.size())
                Values[0].setupArrays(Values[0].out_channels)
            # Flag that it has been setup
            Values[0].currentDInit[0] = 1

# Functions to flag that parameters have been either wrapped with dendrites or intentionaly not wrapped.
def setWrapped_params(model):
    for p in model.parameters():
        p.wrapped = True

def setTracked_params(model):
    for p in model.parameters():
        p.tracked = True

'''
Wrapper to set a module as one that will have dendritic copies
'''
class pb_neuron_layer(nn.Module):
    def __init__(self, startModule, name):
        super(pb_neuron_layer, self).__init__()

        self.mainModule = startModule
        self.name = name
            
        setWrapped_params(self.mainModule)
        if(PBG.verbose):
            print('initing a layer %s with main type %s' % (self.name, type(self.mainModule)))
            print(startModule)
        
        # If this mainModule is one that requires processing set the processor
        if(type(self.mainModule) in PBG.modulesWithProcessing):
            moduleIndex = PBG.modulesWithProcessing.index(type(self.mainModule))
            self.processor = PBG.moduleProcessingClasses[moduleIndex]()
            if(PBG.verbose):
                print('with processor')
                print(self.processor)
        elif(type(self.mainModule).__name__ in PBG.moduleNamesWithProcessing):
            moduleIndex = PBG.moduleNamesWithProcessing.index(type(self.mainModule).__name__)
            self.processor = PBG.moduleByNameProcessingClasses[moduleIndex]()
            if(PBG.verbose):
                print('with processor')
                print(self.processor)
        else:
            self.processor = None
            
        # Field that can be filled in if your activation function requires a parameter
        self.activationFunctionValue = -1
        self.type = 'neuronLayer'
        
        self.register_buffer('thisInputDimensions', (torch.tensor(PBG.inputDimensions)))
        if((self.thisInputDimensions == 0).sum() != 1):
            print('5 Need exactly one 0 in the input dimensions: %s' % self.name)
            print(self.thisInputDimensions)
            sys.exit(-1)
        self.register_buffer('thisNodeIndex', torch.tensor(PBG.inputDimensions.index(0)))
        self.pbLayersAdded = 0

        # Values for dendrite to neuron weights
        self.PBtoTop = nn.ParameterList()
        self.register_parameter('newestPBtoTop', None)
        self.CandidatetoTop = nn.ParameterList()
        self.register_parameter('currentCandidatetoTop', None)
        # Create the dendrite layer
        self.pb = pb_dendrite_layer(self.mainModule,
                                    activationFunctionValue = self.activationFunctionValue,
                                    name = self.name,
                                    inputDimensions = self.thisInputDimensions)
        # If it is linear and default has convolutional dimensions, automatically set to just be batch size and neuron indexes
        if ((issubclass(type(startModule),nn.Linear) or 
            (issubclass(type(startModule),PBG.PBSequential) and issubclass(type(startModule.model[0]),nn.Linear))) 
            and (np.array(self.thisInputDimensions)[2:] == -1).all()): # Everything past 2 is a negative 1
            self.setThisInputDimensions(self.thisInputDimensions[0:2])        
        PBG.pbTracker.addPBNeuronLayer(self)        
        
    # This function allows you to get member variables from the main module
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mainModule, name)
            
    # If processors save values they must be cleared in order to call DeepCopy and save
    def clearProcessors(self):
        if not self.processor:
            return
        else:
            self.processor.clear_processor()
            self.pb.clearProcessors()

    # Before loading from a state dict Dendrites should be cleared and reset.
    # This may not be the most effecient way to do things, but clearing and then
    # simulating cycles is the easiest way to ensure the state dict and the
    # current network have the same number of dendrites
    def clearDendrites(self):
        self.pbLayersAdded = 0
        self.PBtoTop = nn.ParameterList()
        self.CandidatetoTop = nn.ParameterList()
        self.pb = pb_dendrite_layer(self.mainModule,
            activationFunctionValue = self.activationFunctionValue,
            name = self.name,
            inputDimensions = self.thisInputDimensions)

    def __str__(self):
        # If verbose print the whole module otherwise just print the module type as a PAILayer
        if(PBG.verbose):
            totalString = self.mainModule.__str__()
            totalString = 'PAILayer(' + totalString + ')'
            return totalString + self.pb.__str__()
        else:
            totalString = self.mainModule.__str__()
            totalString = 'PAILayer(' + totalString + ')'
            return totalString
    def __repr__(self):
        return self.__str__()
    
    # Set the input dimensions for the neuron and dendrite blocks
    def setThisInputDimensions(self, newInputDimensions):
        if type(newInputDimensions) is list:
            newInputDimensions = torch.tensor(newInputDimensions)
        delattr(self, 'thisInputDimensions')
        self.register_buffer('thisInputDimensions', newInputDimensions.detach().clone())
        if (newInputDimensions == 0).sum() != 1:
            print('6 need exactly one 0 in the input dimensions: %s' % self.name)
            print(newInputDimensions)
        self.thisNodeIndex.copy_((newInputDimensions == 0).nonzero(as_tuple=True)[0][0])
        self.pb.setThisInputDimensions(newInputDimensions)

    # Switch between neuron training and dendrite training
    def setMode(self, mode):
        if(PBG.verbose):
            print('%s calling set mode %c' % (self.name, mode))
        # If returning to neuron training
        if(mode == 'n'):
            self.pb.setMode(mode)
            # Initialize the dendrite to neuron connections
            if(self.pbLayersAdded > 0):
                if(PBG.learnPBLive):
                    values = torch.cat((self.PBtoTop[self.pbLayersAdded-1],nn.Parameter(self.CandidatetoTop.detach().clone())),0)
                else:
                    values = torch.cat((self.PBtoTop[self.pbLayersAdded-1],nn.Parameter(torch.zeros((1,self.out_channels), device=self.PBtoTop[self.pbLayersAdded-1].device, dtype=PBG.dType))),0)
                self.PBtoTop.append(nn.Parameter(values.detach().clone().to(PBG.device), requires_grad=True))
            else:
                if(PBG.learnPBLive):
                    self.PBtoTop.append(nn.Parameter(self.CandidatetoTop.detach().clone(), requires_grad=True))
                else:
                    self.PBtoTop.append(nn.Parameter(torch.zeros((1,self.out_channels), device=PBG.device, dtype=PBG.dType).detach().clone(), requires_grad=True))
            self.pbLayersAdded += 1
        # If starting dendrite training
        else:
            try:
                # Save the values that were calculated in filterBackward
                self.out_channels = self.pb.pbValues[0].out_channels
                self.pb.out_channels = self.pb.pbValues[0].out_channels
            except Exception as e:
                print(e)
                print('this occured in layer: %s' % self.pb.pbValues[0].layerName)
                print('Module should be added to moduleNamesToTrack so it doesn\'t have dendrites added')
                print('If you are getting here but out_channels has not been set')
                print('A common reason is that this layer never had gradients flow through it.')
                print('I have seen this happen because:')
                print('-The weights were frozen (requires_grad = False)')
                print('-A model is added but not used so it was convereted but never PB initialized')
                print('-A module was converted that doesn\'t have weights that get modified so backward doesnt flow through it')
                print('If this is normal behavior set PBG.checkedSkippedLayers = True in the main to ignore')
                print('You can also set right now in this pdb terminal to have this not happen more after checking all layers this cycle.')
                if(not PBG.checkedSkippedLayers):
                    import pdb; pdb.set_trace()
                return False
            # Only change mode if it makes it past the above exception
            self.pb.setMode(mode)
        return True
        
    def addPBLayer(self):
        self.pb.addPBLayer()
            
    def forward(self, *args, **kwargs):
        # If debugging all input dimensions, quit program on first forward call
        if(PBG.debuggingInputDimensions == 2):
            print('all input dim problems now printed')
            sys.exit(0)
        if(PBG.extraVerbose):
            print('%s calling forward' % self.name)
        # Call the main modules forward
        out = self.mainModule(*args, **kwargs)
        # Filter with the processor if required
        if not self.processor is None:
            out = self.processor.post_n1(out)
        # Call the forwards for all of the Dendrites
        pbOuts = self.pb(*args, **kwargs)

        # If there are dendrites add all of their outputs to the neurons output
        if(self.pbLayersAdded > 0):
            for i in range(0,self.pbLayersAdded):
                toTop = self.PBtoTop[self.pbLayersAdded-1][i,:]
                for dim in range(len(pbOuts[i].shape)):
                    if(dim == self.thisNodeIndex):
                        continue
                    toTop = toTop.unsqueeze(dim)
                if(PBG.confirmCorrectSizes):
                    toTop = toTop.expand(list(pbOuts[i].size())[0:self.thisNodeIndex] + [self.out_channels] + list(pbOuts[i].size())[self.thisNodeIndex+1:])
                out = ( out + (pbOuts[i].to(out.device) * toTop.to(out.device)))
        
        # Catch if processors are required
        if(type(out) is tuple):
            print(self)
            print('The output of the above module %s is a tuple when it must be a single tensor')
            print('Look in the API customization.md at section 2.2 regarding processors to fix this.')
            import pdb; pdb.set_trace()

        # Call filter backward to ensure the neuron index is setup correctly
        if(out.requires_grad):
            out.register_hook(lambda grad: filterBackward(grad, self.pb.pbValues, {}))
        
        # If there is a processor apply the second neuron stage
        if not self.processor is None:
            out = self.processor.post_n2(out)
        return out
        
'''
This class exists to wrap the modules you dont want to add dendrites to.
Ensures all modules are accounted for
'''
class tracked_neuron_layer(nn.Module):
    def __init__(self, startModule, name):
        super(tracked_neuron_layer, self).__init__()

        self.mainModule = startModule
        self.name = name
            
        self.type = 'trackedLayer'
        setTracked_params(self.mainModule)
        if(PBG.verbose):
            print('tracking a layer %s with main type %s' % (self.name, type(self.mainModule)))
            print(startModule)
        PBG.pbTracker.addTrackedNeuronLayer(self)        
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.mainModule, name)
        
    def setMode(self, mode):
        if(PBG.verbose):
            print('%s calling set mode %c' % (self.name, mode))
        return True
    
    def forward(self, *args, **kwargs):
        return self.mainModule(*args, **kwargs)

# Randomize weights after duplicating the main module for the next set of dendrites
def init_params(model):
    for p in model.parameters():
        p.data=torch.randn(p.size(), dtype=p.dtype)*PBG.candidateWeightInitializationMultiplier

'''
Module containing all dendrites modules added to the neuron module
'''
class pb_dendrite_layer(nn.Module):
    def __init__(self, initialModule,
                 activationFunctionValue=0.3,
                 name='noNameGiven',
                 inputDimensions = []):
        super(pb_dendrite_layer, self).__init__()
        
        self.layers = nn.ModuleList([])
        self.processors = []
        self.candidateProcessors = []
        self.numPBLayers = 0
        # Number of dendrite cycles performed
        self.register_buffer('numCycles', torch.zeros(1, device=PBG.device, dtype=PBG.dType))
        self.mode = 'n'
        self.name=name
        # Create a copy of the parent module so you dont have a pointer to the real one which causes save errors
        self.parentModule = PBU.deepCopyPAI(initialModule)
        # Setup the input dimensions and node index for combining dendrite outputs
        if(inputDimensions == []):
            self.register_buffer('thisInputDimensions', torch.tensor(PBG.inputDimensions))
        else:
            self.register_buffer('thisInputDimensions', inputDimensions.detach().clone())
        if((self.thisInputDimensions == 0).sum() != 1):
            print('1 need exactly one 0 in the input dimensions: %s' % self.name)
            print(self.thisInputDimensions)
            sys.exit(-1)
        self.register_buffer('thisNodeIndex', torch.tensor(PBG.inputDimensions.index(0)))

        # Initialize dendrite to dendrite connections
        self.PBtoCandidates = nn.ParameterList()
        self.PBtoPB = nn.ParameterList()

        # Store an activation function value if required
        self.activationFunctionValue = activationFunctionValue
        self.pbValues = nn.ModuleList([])
        for j in range(0, PBG.globalCandidates):
            if(PBG.verbose):
                print('creating pb Values for %s' % (self.name))
            self.pbValues.append(pbValueTracker(False, self.activationFunctionValue, self.name, self.thisInputDimensions))

    def setThisInputDimensions(self, newInputDimensions):
        if type(newInputDimensions) is list:
            newInputDimensions = torch.tensor(newInputDimensions)
        delattr(self, 'thisInputDimensions')
        self.register_buffer('thisInputDimensions', newInputDimensions.detach().clone())
        if (newInputDimensions == 0).sum() != 1:
            print('2 Need exactly one 0 in the input dimensions: %s' % self.name)
            print(newInputDimensions)
            sys.exit(-1)
        self.thisNodeIndex.copy_((newInputDimensions == 0).nonzero(as_tuple=True)[0][0])
        for j in range(0, PBG.globalCandidates):
            self.pbValues[j].setThisInputDimensions(newInputDimensions)

    '''
    Function to add a new set of dendrites
    They are initially added as candidates and then added as layers with mode switch to p then back to n
    Open source implimention does both consecutively
    '''
    def addPBLayer(self):
        # Candidate layer
        self.candidateLayer = nn.ModuleList([])
        # Copy that is unused for open source version
        self.candidateBestLayer = nn.ModuleList([])
        if(PBG.verbose):
            print(self.name)
            print('Setting candidate processors')
        self.candidateProcessors = []
        with torch.no_grad():
            for i in range(0, PBG.globalCandidates):
                
                newModule = PBU.deepCopyPAI(self.parentModule)
                init_params(newModule)
                self.candidateLayer.append(newModule)
                self.candidateBestLayer.append(PBU.deepCopyPAI(newModule))
                if(type(self.parentModule) in PBG.modulesWithProcessing):
                    moduleIndex = PBG.modulesWithProcessing.index(type(self.parentModule))
                    self.candidateProcessors.append(PBG.moduleProcessingClasses[moduleIndex]())
                elif(type(self.parentModule).__name__ in PBG.moduleNamesWithProcessing):
                    moduleIndex = PBG.moduleNamesWithProcessing.index(type(self.parentModule).__name__)
                    self.candidateProcessors.append(PBG.moduleByNameProcessingClasses[moduleIndex]())

        for i in range(0, PBG.globalCandidates):
            self.candidateLayer[i].to(PBG.device)
            self.candidateBestLayer[i].to(PBG.device)
        
        # Reset the pbValues objects
        for j in range(0, PBG.globalCandidates):
            self.pbValues[j].reinitializeForPB(0)
        
        # If there are already dendrites initialize the dendrite to dendrite connections
        if(self.numPBLayers > 0):
            self.PBtoCandidates = nn.ParameterList()
            for j in range(0,PBG.globalCandidates): #Loopy Loops
                self.PBtoCandidates.append(nn.Parameter(torch.zeros((self.numPBLayers, self.out_channels), device=PBG.device, dtype=PBG.dType), requires_grad=True))
                self.PBtoCandidates[j].data.pbWrapped = True

    def clearProcessors(self):
        for processor in self.processors:
            if not processor:
                continue
            else:
                processor.clear_processor()
        for processor in self.candidateProcessors:
            if not processor:
                continue
            else:
                processor.clear_processor()

    '''
    Perform actions when switching between neuron and dendrite training
    For open source this will do both in a row to maintain n mode during all training epochs
    '''
    def setMode(self, mode):
        self.mode = mode
        self.numCycles += 1
        if(PBG.verbose):
            print('pb calling set mode %c : %d' % (mode, self.numCycles))
        '''
        When switching back to neuron training mode convert candidates layers into accepted layers
        '''
        if(mode == 'n'):
            if(PBG.verbose):
                print('So calling all the things to add to layers')                
            # Copy weights/bias from correct candidates
            if(self.numPBLayers == 1):
                self.PBtoPB = nn.ParameterList()
                self.PBtoPB.append(torch.tensor([]))
            if(self.numPBLayers >= 1):
                self.PBtoPB.append(torch.nn.Parameter(torch.zeros([self.numPBLayers,self.out_channels], device=PBG.device, dtype=PBG.dType), requires_grad=True))#NEW
            with torch.no_grad():
                if(PBG.globalCandidates > 1):
                    print('This was a flag that will be needed if using multiple candidates.  It\'s not set up yet but nice work finding it.')
                    pdb.set_trace()
                planeMaxIndex = 0
                self.layers.append(PBU.deepCopyPAI(self.candidateBestLayer[planeMaxIndex]))
                self.layers[self.numPBLayers].to(PBG.device)
                if(self.numPBLayers > 0):
                    self.PBtoPB[self.numPBLayers].copy_(self.PBtoCandidates[planeMaxIndex])
                if(type(self.parentModule) in PBG.modulesWithProcessing):
                    self.processors.append(self.candidateProcessors[planeMaxIndex])
                if(type(self.parentModule).__name__ in PBG.moduleNamesWithProcessing):
                    self.processors.append(self.candidateProcessors[planeMaxIndex])

            del self.candidateLayer, self.candidateBestLayer

            self.numPBLayers += 1
        

        
    def forward(self, *args, **kwargs):
        outs = {}
            
        '''
        For all layers apply processors, call the layers, then apply post processors
        '''
        for c in range(0,self.numPBLayers):
            if(self.processors != []):
                args, kwargs = self.processors[c].pre_d(*args, **kwargs)
            outValues = self.layers[c](*args, **kwargs)
            if(self.processors != []):
                outs[c] = self.processors[c].post_d(outValues)
            else:
                outs[c] = outValues

        '''
        Create dendrite outputs
        Each dendrite has input from previously created dendrites
        So activation is added before the nonlinearity is called
        '''
        for outIndex in range(0,self.numPBLayers):
            currentOut = outs[outIndex]
            viewTuple = []
            for dim in range(len(currentOut.shape)):
                if dim == self.thisNodeIndex:
                    viewTuple.append(-1)
                    continue
                viewTuple.append(1)

            for inIndex in range(0,outIndex):
                if(viewTuple == [1]): #This is only the case when passing a single datapoint rather than a batch
                    currentOut = currentOut + self.PBtoPB[outIndex][inIndex,:].to(currentOut.device) * outs[inIndex]            
                else:
                    currentOut = currentOut + self.PBtoPB[outIndex][inIndex,:].view(viewTuple).to(currentOut.device) * outs[inIndex]            
            currentOut = PBG.PBForwardFunction(currentOut)
        # Return a dict which has all dendritic outputs after the activation functions were called
        return outs
        
'''
A tracker object that maintains certain values for each set of dendrites
This object allows for easier communication of data and saving
'''
class pbValueTracker(nn.Module):
    def __init__(self, initialized, activationFunctionValue, name, inputDimensions, out_channels=-1):
        super(pbValueTracker, self).__init__()
        
        self.layerName = name
        for valName in dendriteInitValues:
            self.register_buffer(valName, torch.zeros(1, device=PBG.device, dtype=PBG.dType))
        self.initialized[0] = initialized
        self.activationFunctionValue = activationFunctionValue
        self.register_buffer('thisInputDimensions', inputDimensions.clone().detach())
        if((self.thisInputDimensions == 0).sum() != 1):
            print('3 need exactly one 0 in the input dimensions: %s' % self.layerName)
            print(self.thisInputDimensions)
            sys.exit(-1)
        self.register_buffer('thisNodeIndex', (inputDimensions == 0).nonzero(as_tuple=True)[0])
        if(out_channels != -1):
            self.setupArrays(out_channels)   
        else:
            self.out_channels = -1

    def print(self):
        totalString = 'Value Tracker:'
        for valName in dendriteInitValues:
            totalString += '\t%s:\n\t\t' % valName
            totalString += getattr(self,valName).__repr__()
            totalString += '\n'
        for valName in dendriteTensorValues:
            if(not getattr(self,valName,None) is None):
                totalString += '\t%s:\n\t\t' % valName
                totalString += getattr(self,valName).__repr__()
                totalString += '\n'
        print(totalString)
    
    def setThisInputDimensions(self, newInputDimensions):
        if type(newInputDimensions) is list:
            newInputDimensions = torch.tensor(newInputDimensions)
        delattr(self, 'thisInputDimensions')
        self.register_buffer('thisInputDimensions', newInputDimensions.detach().clone()) 
        if (newInputDimensions == 0).sum() != 1:
            print('4 need exactly one 0 in the input dimensions: %s' % self.layerName)
            print(newInputDimensions)
            sys.exit(-1)
        self.thisNodeIndex.copy_((newInputDimensions == 0).nonzero(as_tuple=True)[0][0])

    def setOutChannels(self, shapeValues):
        if(type(shapeValues) == torch.Size):
            self.out_channels = int(shapeValues[self.thisNodeIndex])
        else:
            self.out_channels = int(shapeValues[self.thisNodeIndex].item())

    def setupArrays(self, out_channels):
        self.out_channels = out_channels
        for valName in dendriteTensorValues:
            self.register_buffer(valName, torch.zeros(out_channels, device=PBG.device, dtype=PBG.dType))
 
        for name in valueTrackerArrays:
            setattr(self,name,{})
            count = 1
            if torch.cuda.device_count() > count:
                count = torch.cuda.device_count()
            for i in range(count):
                getattr(self,name)[i] = []
        for valName in dendriteSingleValues:
            self.register_buffer(valName, torch.zeros(1, device=PBG.device, dtype=PBG.dType))            
        
    def reinitializeForPB(self, initialized):
        if(self.out_channels == -1):
            print('You have a converted module that was never initialized')
            print('This likely means it not being added to the autograd graph')
            print('Check your forward function that it is actually being used')
            print('If its not you should really delete it, but you can also add')
            print('the name below to PBG.moduleNamesToSkip to not convert it')
            print(self.layerName)
            print('This can also happen while testingDendriteCapactity if you')
            print('run a validation cycle and try to add Dendrites before doing any training.\n')
            
        self.initialized[0] = initialized
        for valName in dendriteReinitValues:
            setattr(self,valName,getattr(self,valName) * 0)
