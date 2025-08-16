# Copyright (c) 2025 Perforated AI

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math
import sys
import numpy as np
import pdb
import io
import shutil

import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
import pandas as pd
import copy
import os
from pydoc import locate

from perforatedai import pb_globals as PBG
from perforatedai import pb_layer as PB
from perforatedai import pb_utils as PBU

'''
This is a manager class that tracks all the neuron layers and dendrite layers,
controls when new dendrites are added, and communicates those signals to the modules
'''

class pb_neuron_layer_tracker():
    
    def __init__(self, doingPB, saveName, makingGraphs=True, paramValsSetting=-1, values_per_train_epoch=-1, values_per_val_epoch=-1):
        # Dict of member vars and what types they are for saving
        self.memberVars = {}
        self.memberVarTypes = {}

        # Whether or not PB will be running
        self.memberVars['doingPB'] = doingPB
        self.memberVarTypes['doingPB'] = 'bool'

        # How many Dendrite Nodes have been added
        self.memberVars['numPBNeuronLayers'] = 0
        self.memberVarTypes['numPBNeuronLayers'] = 'int'

        # How many cycles have been run, *2 or *2+1 of the above
        self.memberVars['numCycles'] = 0
        self.memberVarTypes['numCycles'] = 'int'

        # Pointers to all neuron wrapped modules
        self.PBNeuronLayerVector = []

        # Pointers to all non neuron modules for tracking
        self.trackedNeuronLayerVector = []

        # Neuron training or dendrite training mode
        self.memberVars['mode'] = 'n'
        self.memberVarTypes['mode'] = 'string'

        # Number of epochs run excluding overwritten epochs
        self.memberVars['numEpochsRun'] = -1
        self.memberVarTypes['numEpochsRun'] = 'int'

        # Number including overwritten epochs
        self.memberVars['totalEpochsRun'] = -1
        self.memberVarTypes['totalEpochsRun'] = 'int'

        # Last epoch that the validation score or correlation score was improved
        self.memberVars['epochLastImproved'] = 0
        self.memberVarTypes['epochLastImproved'] = 'int'

        # Running validation accuracy
        self.memberVars['runningAccuracy'] = 0
        self.memberVarTypes['runningAccuracy'] = 'float'

        # True if maxing validation, False if minimizing Loss
        self.memberVars['maximizingScore'] = True
        self.memberVarTypes['maximizingScore'] = 'bool'

        # Mode for switching back and forth between learning modes
        self.memberVars['switchMode'] = PBG.switchMode
        self.memberVarTypes['switchMode'] = 'int'

        # Epoch of the last switch
        self.memberVars['lastSwitch'] = 0
        self.memberVarTypes['lastSwitch'] = 'int'

        # Highest validation score from current cycle
        self.memberVars['currentBestValidationScore'] = 0
        self.memberVarTypes['currentBestValidationScore'] = 'float'

        # Last epoch where the learning rate was updated
        self.memberVars['initialLRTestEpochCount'] = -1
        self.memberVarTypes['initialLRTestEpochCount'] = 'int'

        # Highest validation score of full run
        self.memberVars['globalBestValidationScore'] = 0
        self.memberVarTypes['globalBestValidationScore'] = 'float'
        
        # List of switch epochs
        self.memberVars['switchEpochs'] = []
        self.memberVarTypes['switchEpochs'] = 'int array'
        
        # Paramter counts at each network structure
        self.memberVars['paramCounts'] = []
        self.memberVarTypes['paramCounts'] = 'int array'

        # List of epochs where switch was made to neuron training
        self.memberVars['nswitchEpochs'] = []
        self.memberVarTypes['nswitchEpochs'] = 'int array'

        # List of epochs where switch was made to dendrite training
        self.memberVars['pswitchEpochs'] = []
        self.memberVarTypes['pswitchEpochs'] = 'int array'

        # List of validation accuracies
        self.memberVars['accuracies'] = []
        self.memberVarTypes['accuracies'] = 'float array'

        # List of epochs where score was improved for use when updating scheduler
        self.memberVars['lastImprovedAccuracies'] = []
        self.memberVarTypes['lastImprovedAccuracies'] = 'int array'

        # List of test accuracy scores registered
        self.memberVars['testAccuracies'] = []
        self.memberVarTypes['testAccuracies'] = 'float array'

        # List of accuracies registered during neuron training
        self.memberVars['nAccuracies'] = []
        self.memberVarTypes['nAccuracies'] = 'float array'

        # List of accuracies registered during dendrite training
        self.memberVars['pAccuracies'] = []
        self.memberVarTypes['pAccuracies'] = 'float array'

        # Running average accuracies from recent epochs
        self.memberVars['runningAccuracies'] = []
        self.memberVarTypes['runningAccuracies'] = 'float array'

        # List of additional score recorded
        self.memberVars['extraScores'] = {}
        self.memberVarTypes['extraScores'] = 'float array dictionary'

        # Extra scores that are not set to be graphed
        # Can be used to track loss when graph is tracking accuracies
        self.memberVars['extraScoresWithoutGraphing'] = {}
        self.memberVarTypes['extraScoresWithoutGraphing'] = 'float array dictionary'

        # List of test scores
        self.memberVars['testScores'] = []
        self.memberVarTypes['testScores'] = 'float array'

        # List of extra scores calculated during neuron training
        self.memberVars['nExtraScores'] = {}
        self.memberVarTypes['nExtraScores'] = 'float array dictionary'

        # List of training losses calculated
        self.memberVars['trainingLoss'] = []
        self.memberVarTypes['trainingLoss'] = 'float array'

        # List of learning rates at each epoch
        self.memberVars['trainingLearningRates'] = []
        self.memberVarTypes['trainingLearningRates'] = 'float array'

        # Best dendrite scores
        self.memberVars['bestScores'] = []
        self.memberVarTypes['bestScores'] = 'float array array'

        # Current dendrite scores
        self.memberVars['currentScores'] = []
        self.memberVarTypes['currentScores'] = 'float array array'

        # Times for neuron training epochs
        self.memberVars['nEpochTimes'] = []
        self.memberVarTypes['nEpochTimes'] = 'float array'

        # Timing values
        self.memberVars['pEpochTimes'] = []
        self.memberVarTypes['pEpochTimes'] = 'float array'
        self.memberVars['nTrainTimes'] = []
        self.memberVarTypes['nTrainTimes'] = 'float array'
        self.memberVars['pTrainTimes'] = []
        self.memberVarTypes['pTrainTimes'] = 'float array'
        self.memberVars['nValTimes'] = []
        self.memberVarTypes['nValTimes'] = 'float array'
        self.memberVars['pValTimes'] = []
        self.memberVarTypes['pValTimes'] = 'float array'
        # Setting involved with how to track timing
        self.memberVars['manualTrainSwitch'] = False
        self.memberVarTypes['manualTrainSwitch'] = 'bool'

        # Tracking of additional scores that get overwritten when reloading best model before adding dendrites
        self.memberVars['overWrittenExtras'] = []
        self.memberVarTypes['overWrittenExtras'] = 'float array dictionary array'
        self.memberVars['overWrittenVals'] = []
        self.memberVarTypes['overWrittenVals'] = 'float array array'
        self.memberVars['overWrittenEpochs'] = 0
        self.memberVarTypes['overWrittenEpochs'] = 'int'

        # Setting for how to determine some of above scores
        self.memberVars['paramValsSetting'] = PBG.paramValsSetting
        self.memberVarTypes['paramValsSetting'] = 'int'

        # Pointer to optimizer and scheduler types, and instantiated instances
        self.memberVars['optimizer'] = None
        self.memberVarTypes['optimizer'] = 'type'
        self.memberVars['scheduler'] = None
        self.memberVarTypes['scheduler'] = 'type'
        self.memberVars['optimizerInstance'] = None
        self.memberVarTypes['optimizerInstance'] = 'empty array'
        self.memberVars['schedulerInstance'] = None
        self.memberVarTypes['schedulerInstance'] = 'empty array'

        # Flag for if the tracker was loaded
        self.loaded = False

        # Settings for tracking learning rates to find start rate when adding dendrites
        self.memberVars['currentNLearningRateInitialSkipSteps'] = 0
        self.memberVarTypes['currentNLearningRateInitialSkipSteps'] = 'int'
        self.memberVars['lastMaxLearningRateSteps'] = 0
        self.memberVarTypes['lastMaxLearningRateSteps'] = 'int'
        self.memberVars['lastMaxLearningRateValue'] = -1
        self.memberVarTypes['lastMaxLearningRateValue'] = 'float'
        self.memberVars['currentCycleLRMaxScores'] = []
        self.memberVarTypes['currentCycleLRMaxScores'] = 'float array'
        self.memberVars['currentStepCount'] = 0
        self.memberVarTypes['currentStepCount'] = 'int'
        self.memberVars['committedToInitialRate'] = True
        self.memberVarTypes['committedToInitialRate'] = 'bool'
        self.memberVars['currentNSetGlobalBest'] = True
        
        # Flag for if the current dendrite achieved the highest global score
        self.memberVarTypes['currentNSetGlobalBest'] = 'bool'

        # Number of tries of adding this dendrite count
        self.memberVars['numDendriteTries'] = 0
        self.memberVarTypes['numDendriteTries'] = 'int'

        # count of batches per epoch
        self.values_per_train_epoch=values_per_train_epoch
        self.values_per_val_epoch=values_per_val_epoch

        self.saveName = saveName
        self.makingGraphs = makingGraphs

        self.startTime = time.time()
        self.savedTime = 0
        self.startEpoch(internalCall=True)

        if(PBG.verbose):
            print('initing with switchMode%s' % (self.memberVars['switchMode']))
            
    # Converts the values of the tracker to a string for saving with safetensors
    def toString(self):
        fullString = ''
        for var in self.memberVars:
            fullString += (var+',')
            if(self.memberVars[var] == None):
                fullString += ('None')
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'bool':
                fullString += (str(self.memberVars[var]))
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'int':
                fullString += (str(self.memberVars[var]))
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'float':
                fullString += (str(self.memberVars[var]))
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'string':
                fullString += (str(self.memberVars[var]))
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'type':
                name = self.memberVars[var].__module__ + '.' + self.memberVars[var].__name__
                fullString += (str(self.memberVars[var]))
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'empty array':
                fullString += ('[]')
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'int array' or self.memberVarTypes[var] == 'float array':
                fullString += ('\n')
                string = ''
                for val in self.memberVars[var]:
                    string += str(val) + ','
                #remove the last ,
                string = string[:-1]
                fullString += (string)
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'float array dictionary array':
                fullString += ('\n')
                for array in self.memberVars[var]:
                    for key in array:
                        string = ''
                        string += key + ','
                        for val in array[key]:
                            string += str(val) + ','
                        #remove the last ,
                        string = string[:-1]
                        fullString += (string)
                        fullString += ('\n')
                    fullString += ('endkey')
                    fullString += ('\n')
                fullString += ('endarray')
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'float array dictionary':
                fullString += ('\n')
                for key in self.memberVars[var]:
                    string = ''
                    string += key + ','
                    for val in self.memberVars[var][key]:
                        string += str(val) + ','
                    #remove the last ,
                    string = string[:-1]
                    fullString += (string)
                    fullString += ('\n')
                fullString += ('end')
                fullString += ('\n')
            elif self.memberVarTypes[var] == 'float array array':
                fullString += ('\n')
                for array in self.memberVars[var]:
                    string = ''
                    for val in array:
                        string += str(val) + ','
                    #remove the last ,
                    string = string[:-1]
                    fullString += (string)
                    fullString += ('\n')
                fullString += ('end')
                fullString += ('\n')
            else:
                print('Didnt find a member variable')
                import pdb; pdb.set_trace()
        return fullString
    #Same for loading
    def fromString(self, string):
        f = io.StringIO(string)
        while(True):
            line = f.readline()
            if(not line):
                break
            vals = line.split(',')
            var = vals[0]
            if self.memberVarTypes[var] == 'bool':
                val = vals[1][:-1]
                if(val == 'True'):
                    self.memberVars[var] = True
                elif(val == 'False'):
                    self.memberVars[var] = False
                elif(val == '1'):
                    self.memberVars[var] = 1
                elif(val == '0'):
                    self.memberVars[var] = 0
                else:
                    print('Something went wrong with loading')
                    import pdb; pdb.set_trace()
            elif self.memberVarTypes[var] == 'int':
                val = vals[1]
                self.memberVars[var] = int(val)
            elif self.memberVarTypes[var] == 'float':
                val = vals[1]
                self.memberVars[var] = float(val)
            elif self.memberVarTypes[var] == 'string':
                val = vals[1][:-1]
                self.memberVars[var] = val
            elif self.memberVarTypes[var] == 'type':
                #ignore loading types, the pbTracker should already have them setup and overwriting will cause problems
                continue
            elif self.memberVarTypes[var] == 'empty array':
                val = vals[1]
                self.memberVars[var] = []
            elif self.memberVarTypes[var] == 'int array':
                vals = f.readline()[:-1].split(',')
                self.memberVars[var] = []
                if vals[0] == '':
                    continue
                for val in vals:
                    self.memberVars[var].append(int(val))
            elif self.memberVarTypes[var] == 'float array':
                vals = f.readline()[:-1].split(',')
                self.memberVars[var] = []
                if vals[0] == '':
                    continue
                for val in vals:
                    self.memberVars[var].append(float(val))
            elif self.memberVarTypes[var] == 'float array dictionary array':
                self.memberVars[var] = []
                line2 = f.readline()[:-1]
                while line2 != 'endarray':
                    temp = {}
                    while line2 != 'endkey':
                        vals = line2.split(',')
                        name = vals[0]
                        temp[name] = []
                        vals = vals[1:]
                        for val in vals:
                            temp[name].append(float(val))
                        line2 = f.readline()[:-1]
                    self.memberVars[var].append(temp)
                    line2 = f.readline()[:-1]
            elif self.memberVarTypes[var] == 'float array dictionary':
                self.memberVars[var] = {}
                line2 = f.readline()[:-1]
                while line2 != 'end':
                    vals = line2.split(',')
                    name = vals[0]
                    self.memberVars[var][name] = []
                    vals = vals[1:]
                    for val in vals:
                        self.memberVars[var][name].append(float(val))
                    line2 = f.readline()[:-1]
            elif self.memberVarTypes[var] == 'float array array':
                self.memberVars[var] = []
                line2 = f.readline()[:-1]
                while line2 != 'end':
                    vals = line2.split(',')
                    self.memberVars[var].append([])
                    if not line2 == '':
                        for val in vals:
                            self.memberVars[var][-1].append(float(val))
                    line2 = f.readline()[:-1]
            else:
                print('didnt find a member variable')
                import pdb; pdb.set_trace()
    
    # The following two functions are used for the current method of using DistributedDataParallel.
    # Instructions for use in API customization.md
    # We are looking into better options
    def saveTrackerSettings(self):
        if not os.path.isdir(self.saveName):
            os.makedirs(self.saveName)
        f = open(self.saveName + '/arrayDims.csv', 'w')
        for layer in self.PBNeuronLayerVector:
            f.write('%s,%d\n' % (layer.name, layer.pb.pbValues[0].out_channels))
        f.close()
        if(not PBG.silent):
            print('Tracker settings saved.')
            print('You may now delete saveTrackerSettings')
    def initializeTrackerSettings(self):
        channels = {}
        if not os.path.exists(self.saveName + '/arrayDims.csv'):
            print('You must call saveTrackerSettings before initializeTrackerSettings')
            print('Follow instructions in customization.md')
            pdb.set_trace()
        f = open(self.saveName + '/arrayDims.csv', 'r')
        for line in f:
            channels[line.split(',')[0]] = int(line.split(',')[1])
        for layer in self.PBNeuronLayerVector:
            layer.pb.pbValues[0].setupArrays(channels[layer.name])
        
    # This is the case for if you just want to use original optimizer and not track it here
    def setOptimizerInstance(self, optimizerInstance):
        try:
            if(optimizerInstance.param_groups[0]['weight_decay'] > 0 and PBG.weightDecayAccepted == False):
                print('For PAI training it is reccomended to not use weight decay in your optimizer')
                print('Set PBG.weightDecayAccepted = True to ignore this warning or c to continue')
                PBG.weightDecayAccepted = True
                import pdb; pdb.set_trace()
        except:
            pass
        self.memberVars['optimizerInstance'] = optimizerInstance

    #Set optimizer and scheduler types
    def setOptimizer(self, optimizer):
        self.memberVars['optimizer'] = optimizer
    def setScheduler(self, scheduler):
        if(not scheduler is torch.optim.lr_scheduler.ReduceLROnPlateau):
            if(PBG.verbose):
                print('Not using reduce on plateou, this is not reccomended')        
        self.memberVars['scheduler'] = scheduler
        
    # Increment the scheduler a set number of times.
    # Used for finding best initial learning rate when adding dendrites
    def incrementScheduler(self, numTicks, mode):
        currentSteps = 0
        currentTicker = 0
        for param_group in PBG.pbTracker.memberVars['optimizerInstance'].param_groups:
            learningRate1 = param_group['lr']
        if(PBG.verbose):
            print('using scheduler:')
            print(type(self.memberVars['schedulerInstance']))
        while currentTicker < numTicks:
            if(PBG.verbose):
                print('lower start rate initial %f stepping %d times' % (learningRate1, PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))
            if type(self.memberVars['schedulerInstance']) is torch.optim.lr_scheduler.ReduceLROnPlateau:
                if(mode == 'stepLearningRate'):
                    #step with the counter as last improved accuracy from the initial value before this switch.  This is used to initially start with a lower rate
                    self.memberVars['schedulerInstance'].step(metrics=self.memberVars['lastImprovedAccuracies'][PBG.pbTracker.stepsAfterSwitch()-1])
                elif(mode == 'incrementEpochCount'):
                    #step with the the improved epoch counts up to current location, this is used when loading.
                    self.memberVars['schedulerInstance'].step(metrics=self.memberVars['lastImprovedAccuracies'][-((numTicks-1)-currentTicker)-1])
            else:
                    self.memberVars['schedulerInstance'].step()
            for param_group in PBG.pbTracker.memberVars['optimizerInstance'].param_groups:
                learningRate2 = param_group['lr']
            if(learningRate2 != learningRate1):
                currentSteps += 1
                learningRate1 = learningRate2
                if(mode == 'stepLearningRate'):
                    currentTicker += 1
                if(PBG.verbose):
                    print('1 step %d to %f' % (currentSteps, learningRate2))
            if(mode == 'incrementEpochCount'):
                currentTicker += 1
        return currentSteps, learningRate1
    
    # Initialize the optimizer, and scheduler when added
    def setupOptimizer(self, net, optArgs, schedArgs = None):
        if('weight_decay' in optArgs and not(PBG.weightDecayAccepted)):
            print('For PAI training it is reccomended to not use weight decay in your optimizer')
            print('Set PBG.weightDecayAccepted = True to ignore this warning or c to continue')
            PBG.weightDecayAccepted = True
            import pdb; pdb.set_trace()
        if(not 'model' in optArgs.keys()):
            if(self.memberVars['mode'] == 'n'):
                optArgs['params'] = filter(lambda p: p.requires_grad, net.parameters())
            else:
                optArgs['params'] = PBU.getPBNetworkParams(net)
        optimizer = self.memberVars['optimizer'](**optArgs)
        self.memberVars['optimizerInstance'] = optimizer
        if(self.memberVars['scheduler'] != None):
            self.memberVars['schedulerInstance'] = self.memberVars['scheduler'](optimizer, **schedArgs)
            currentSteps = 0
            for param_group in PBG.pbTracker.memberVars['optimizerInstance'].param_groups:
                learningRate1 = param_group['lr']
            if(PBG.verbose):
                print('resetting scheduler with %d steps and %d initial ticks to skip' % (PBG.pbTracker.stepsAfterSwitch(), PBG.initialHistoryAfterSwitches))
            # This block finds the setting of the previously used learning rate before adding dendrites
            if(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'] != 0):
                additionalSteps, learningRate1 = self.incrementScheduler(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'], 'stepLearningRate')
                currentSteps += additionalSteps
            if(self.memberVars['mode'] == 'n' or PBG.learnPBLive):
                initial = PBG.initialHistoryAfterSwitches
            else:
                initial = 0
            if(PBG.pbTracker.stepsAfterSwitch() > initial):
                # Minus an extra 1 becuase this will be getting called after start epoch has been called at the end of add validation score, 
                # which means steps after switch will actually be off by 1
                additionalSteps, learningRate1 = self.incrementScheduler((PBG.pbTracker.stepsAfterSwitch() - initial)-1, 'incrementEpochCount')
                currentSteps += additionalSteps
            if(PBG.verbose):
                print('scheduler update loop with %d ended with %f' % (currentSteps, learningRate1))
                print('scheduler ended with %d steps and lr of %f' % (currentSteps, learningRate1))
            self.memberVars['currentStepCount'] = currentSteps
            return optimizer, self.memberVars['schedulerInstance']
        else:
            return optimizer

    # Clears the instances for saving
    def clearOptimizerAndScheduler(self):
        self.memberVars['optimizerInstance'] = None
        self.memberVars['schedulerInstance'] = None

    # Based on settings and scores determines if it is time to switch between neuron and dendrite training
    def switchTime(self):
        switchPhrase = 'No mode, this should never be the case.'
        if(self.memberVars['switchMode'] == PBG.doingSwitchEveryTime):
           switchPhrase = 'doingSwitchEveryTime'
        elif(self.memberVars['switchMode'] == PBG.doingHistory):
           switchPhrase = 'doingHistory'
        elif(self.memberVars['switchMode'] == PBG.doingFixedSwitch):
           switchPhrase = 'doingFixedSwitch'
        elif(self.memberVars['switchMode'] == PBG.doingNoSwitch):
           switchPhrase = 'doingNoSwitch'
        if(not PBG.silent):
            print('Checking PB switch with mode %c, switch mode %s, epoch %d, last improved epoch %d, total Epochs %d, n: %d, numCycles: %d' % 
            (self.memberVars['mode'], switchPhrase, self.memberVars['numEpochsRun'], self.memberVars['epochLastImproved'], 
            self.memberVars['totalEpochsRun'], PBG.nEpochsToSwitch, self.memberVars['numCycles']))
        if(self.memberVars['switchMode'] == PBG.doingNoSwitch):
            if(not PBG.silent):
                print('Returning False - doing no switch mode')
            return False
        if(self.memberVars['switchMode'] == PBG.doingSwitchEveryTime):
            if(not PBG.silent):
                print('Returning True - switching every time')
            return True
        if(((self.memberVars['mode'] == 'n') or PBG.learnPBLive) and (self.memberVars['switchMode'] == PBG.doingHistory) and 
           (PBG.pbTracker.memberVars['committedToInitialRate'] == False) and 
           (PBG.dontGiveUpUnlessLearningRateLowered)
           and (self.memberVars['currentNLearningRateInitialSkipSteps'] < self.memberVars['lastMaxLearningRateSteps']) and self.memberVars['scheduler'] != None):
            if(not PBG.silent):
                print('Returning False since no first step yet and comparing initial %d to last max %d' %(self.memberVars['currentNLearningRateInitialSkipSteps'], self.memberVars['lastMaxLearningRateSteps']))
            return False
        capSwitch = False
        if(len(self.memberVars['switchEpochs']) == 0):
            thisCount = (self.memberVars['numEpochsRun'])
        else:
            thisCount = (self.memberVars['numEpochsRun'] - self.memberVars['switchEpochs'][-1])
        if(self.memberVars['switchMode'] == PBG.doingHistory and 
            (
                ((self.memberVars['mode'] == 'n') and (self.memberVars['numEpochsRun'] - self.memberVars['epochLastImproved'] >= PBG.nEpochsToSwitch) and thisCount >= PBG.initialHistoryAfterSwitches + PBG.nEpochsToSwitch)
             or capSwitch)):
            if(not PBG.silent):
                print('Returning True - History and last improved is hit')
            return True
        if(self.memberVars['switchMode'] == PBG.doingFixedSwitch and ((self.memberVars['totalEpochsRun']%PBG.fixedSwitchNum == 0) and self.memberVars['numEpochsRun'] >= PBG.firstFixedSwitchNum)):
            if(not PBG.silent):
                print('Returning True - Fixed switch number is hit')
            return True
        if(not PBG.silent):
            print('Returning False - no triggers to switch have been hit')
        return False
    
    # Based on settings returns a value for how many steps its been since a switch
    def stepsAfterSwitch(self):
        if(self.memberVars['paramValsSetting'] == PBG.paramValsByTotalEpoch):
            return self.memberVars['numEpochsRun']
        elif(self.memberVars['paramValsSetting'] == PBG.paramValsByUpdateEpoch):
            return self.memberVars['numEpochsRun'] - self.memberVars['lastSwitch']
        elif(self.memberVars['paramValsSetting'] == PBG.paramValsByNormalEpochStart):
            if(self.memberVars['mode'] == 'p'):
                return self.memberVars['numEpochsRun'] - self.memberVars['lastSwitch']
            else:
                return self.memberVars['numEpochsRun']
        else:
            print('%d is not a valid param vals option' % self.memberVars['paramValsSetting'])
            pdb.set_trace()
    
    # Adds neuron layers and tracked layers to internal vectors
    def addPBNeuronLayer(self, newLayer, initialAdd=True):
        # If its a duplicate just ignore the second addition
        if(newLayer in self.PBNeuronLayerVector):
            return
        self.PBNeuronLayerVector.append(newLayer)
        if(self.memberVars['doingPB']):
            PB.setWrapped_params(newLayer)
        if(initialAdd):
            self.memberVars['bestScores'].append([])
            self.memberVars['currentScores'].append([])
    def addTrackedNeuronLayer(self, newLayer, initialAdd=True):
        #if its a duplicate just ignore the second addition
        if(newLayer in self.trackedNeuronLayerVector):
            return
        self.trackedNeuronLayerVector.append(newLayer)
        if(self.memberVars['doingPB']):
            PB.setTracked_params(newLayer)        
        
    # Clears internal vectors
    def resetLayerVector(self, net,loadFromRestart):
        self.PBNeuronLayerVector = []
        self.trackedNeuronLayerVector = []
        thisList = PBU.getPBModules(net, 0)
        for module in thisList:
            self.addPBNeuronLayer(module, initialAdd=loadFromRestart)
        thisList = PBU.getTrackedModules(net, 0)
        for module in thisList:
            self.addTrackedNeuronLayer(module, initialAdd=loadFromRestart)
                    
    # Reset values if resetting scores
    def resetValsForScoreReset(self):
        if(PBG.findBestLR):
            self.memberVars['committedToInitialRate'] = False        
        self.memberVars['currentNSetGlobalBest'] = False
        # Dont rest the global best, but do reset the current best,
        # this is needed when doing learning rate picking to not retain old best
        self.memberVars['currentBestValidationScore'] = 0
        self.memberVars['initialLRTestEpochCount'] = -1

    # Signal all layers to start dendrite training   
    def setPBTraining(self):
        if(PBG.verbose):
            print('calling set PBTraining')

        for layer in self.PBNeuronLayerVector[:]:
                worked = layer.setMode('p')
                '''
                worked is False when a layer was added to the PB vector
                but then its never actually be used.  This can happen when you have set a layer
                to have requires_grad = False or when you have a module as a member variable but
                its not actually part of the network.  Should be moved to be a tracked layer rather
                than a neuron layer
                '''
                if not worked:
                    self.PBNeuronLayerVector.remove(layer)
        for layer in self.trackedNeuronLayerVector[:]:
                worked = layer.setMode('p')
        self.addPBLayer()
        self.memberVars['mode'] = 'p'
        self.memberVars['currentNLearningRateInitialSkipSteps'] = 0
        if(PBG.learnPBLive):
            self.resetValsForScoreReset()

        self.memberVars['lastMaxLearningRateSteps'] = self.memberVars['currentStepCount']

        PBG.pbTracker.memberVars['currentCycleLRMaxScores'] = []
        PBG.pbTracker.memberVars['numCycles'] += 1

    # Signal all layers to start neuron training   
    def setNormalTraining(self):
        for layer in self.PBNeuronLayerVector:
            layer.setMode('n')
        for layer in self.trackedNeuronLayerVector[:]:
            layer.setMode('n')
        self.memberVars['mode'] = 'n'
        self.memberVars['numPBNeuronLayers'] += 1
        self.memberVars['currentNLearningRateInitialSkipSteps'] = 0
        self.resetValsForScoreReset()

        self.memberVars['currentCycleLRMaxScores'] = []        
        if(PBG.learnPBLive):
            self.memberVars['lastMaxLearningRateSteps'] = self.memberVars['currentStepCount']
        PBG.pbTracker.memberVars['numCycles'] += 1
        if(PBG.resetBestScoreOnSwitch):
            PBG.pbTracker.memberVars['currentBestValidationScore'] = 0
            PBG.pbTracker.memberVars['runningAccuracy'] = 0

    # Perform steps for when a new training epoch is about to begin
    def startEpoch(self, internalCall=False):
        if(self.memberVars['manualTrainSwitch'] and internalCall==True):
            return
        if(internalCall==False and self.memberVars['manualTrainSwitch'] == False):
            self.memberVars['manualTrainSwitch'] = True
            self.savedTime = 0
            self.memberVars['numEpochsRun'] = -1
            self.memberVars['totalEpochsRun'] = -1
        end = time.time()
        if(self.memberVars['manualTrainSwitch']):
            if(self.savedTime != 0):
                if(self.memberVars['mode'] == 'p'):
                    self.memberVars['pValTimes'].append(end - self.savedTime)
                else:
                    self.memberVars['nValTimes'].append(end - self.savedTime)
        if(self.memberVars['mode'] == 'p'):
            for layer in self.PBNeuronLayerVector:
                for m in range(0, PBG.globalCandidates):
                    with torch.no_grad():
                        if(PBG.verbose):
                            print('resetting score for %s' % layer.name)
                        layer.pb.pbValues[m].bestScoreImprovedThisEpoch = layer.pb.pbValues[m].bestScoreImprovedThisEpoch * 0
                        layer.pb.pbValues[m].nodesBestImprovedThisEpoch = layer.pb.pbValues[m].nodesBestImprovedThisEpoch * 0

        self.memberVars['numEpochsRun'] += 1
        self.memberVars['totalEpochsRun'] = self.memberVars['numEpochsRun'] + self.memberVars['overWrittenEpochs']
        self.savedTime = end

    # Perform steps when a training epoch has completed
    def stopEpoch(self, internalCall=False):
        end = time.time()
        if(self.memberVars['manualTrainSwitch'] and internalCall==True):
            return
        if(self.memberVars['manualTrainSwitch']):
            if(self.memberVars['mode'] == 'p'):
                self.memberVars['pTrainTimes'].append(end - self.savedTime)
            else:
                self.memberVars['nTrainTimes'].append(end - self.savedTime)
        else:
            if(self.memberVars['mode'] == 'p'):
                self.memberVars['pEpochTimes'].append(end - self.savedTime)
            else:
                self.memberVars['nEpochTimes'].append(end - self.savedTime)            
        self.savedTime = end

    # Setup the tracker with initial settings
    def initialize(self, 
                   model, 
                   doingPB=True, 
                   saveName='PB', 
                   makingGraphs=True, 
                   maximizingScore=True, 
                   num_classes=10000, 
                   values_per_train_epoch=-1, 
                   values_per_val_epoch=-1, 
                   zoomingGraph=True):
        model = PBU.convertNetwork(model)
        self.memberVars['doingPB'] = doingPB
        self.memberVars['maximizingScore'] = maximizingScore
        self.saveName = saveName
        self.zoomingGraph = zoomingGraph
        self.makingGraphs = makingGraphs
        if(self.loaded == False):
            self.memberVars['runningAccuracy'] = (1.0/num_classes) * 100
        self.values_per_train_epoch=values_per_train_epoch
        self.values_per_val_epoch=values_per_val_epoch
        
        if(PBG.testingDendriteCapacity):
            if(not PBG.silent):
                print('Running a test of Dendrite Capacity.')
            PBG.switchMode = PBG.doingSwitchEveryTime
            self.memberVars['switchMode'] = PBG.switchMode
            PBG.retainAllPB = True
            PBG.maxDendriteTries = 1000
            PBG.maxDendrites = 1000
        else:
            if(not PBG.silent):
                print('Running PB experiment')
        return model
        
    '''
    Function to save graphs for all the values
    TODO: clean this up, add comments, and separate it into more functions
    '''
    def saveGraphs(self, extraString=''):
        if(self.makingGraphs == False):
            return
        
        saveFolder = './' + self.saveName + '/'
        
        plt.ioff()
        fig = plt.figure(figsize=(28,14))

        #Plot with accuracy scores
        ax = plt.subplot(221)
        df1 = None
        
        for listID in range(len(self.memberVars['overWrittenExtras'])):
            for extraID in self.memberVars['overWrittenExtras'][listID]:
                ax.plot(np.arange(len(self.memberVars['overWrittenExtras'][listID][extraID])), self.memberVars['overWrittenExtras'][listID][extraID], 'r')
            ax.plot(np.arange(len(self.memberVars['overWrittenVals'][listID])), self.memberVars['overWrittenVals'][listID], 'b')
        
        if(PBG.drawingPB):
            accuracies = self.memberVars['accuracies']
            extraScores = self.memberVars['extraScores']
        else:
            accuracies = self.memberVars['nAccuracies']
            extraScores = self.memberVars['extraScores']
        
        ax.plot(np.arange(len(accuracies)), accuracies, label='Validation Scores')
        ax.plot(np.arange(len(self.memberVars['runningAccuracies'])), self.memberVars['runningAccuracies'], label='Validation Running Scores')
        for extraScore in extraScores:
            ax.plot(np.arange(len(extraScores[extraScore])), extraScores[extraScore], label=extraScore)
        plt.title(saveFolder + '/' + self.saveName + "Scores")
        plt.xlabel('Epochs')
        plt.ylabel('Score')
        
        # This will add a point at epoch last improved so while watching can tell when a switch is coming
        lastImproved = self.memberVars['epochLastImproved']
        if(PBG.drawingPB):
            ax.plot(lastImproved, self.memberVars['globalBestValidationScore'], 'bo', label='Global best (y)')
            ax.plot(lastImproved, accuracies[lastImproved], 'go', label='Epoch Last Improved \nmight be wrong in\nfirst after switch')
        else:
            if(self.memberVars['mode'] == 'n'):
                missedTime = self.memberVars['numEpochsRun'] - lastImproved
                ax.plot((len(self.memberVars['nAccuracies'])-1) - missedTime, self.memberVars['nAccuracies'][-(missedTime+1)], 'go', label='Epoch Last Improved')
            
        
        pd1 = pd.DataFrame({'Epochs': np.arange(len(accuracies)), 'Validation Scores': accuracies})
        pd2 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['runningAccuracies'])), 'Validation Running Scores': self.memberVars['runningAccuracies']})
        pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)        
        for extraScore in extraScores:
            pd2 = pd.DataFrame({'Epochs': np.arange(len(extraScores[extraScore])), extraScore: extraScores[extraScore]})
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)  
        extraScoresWithoutGraphing = self.memberVars['extraScoresWithoutGraphing']
        for extraScore in extraScoresWithoutGraphing:
            pd2 = pd.DataFrame({'Epochs': np.arange(len(extraScoresWithoutGraphing[extraScore])), extraScore: extraScoresWithoutGraphing[extraScore]})
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)        

        pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'Scores.csv', index=False)
        pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
        del pd1, pd2
        
        # If it has done a switch set the y min and max to zoom in on the more important part of the axis
        if(len(self.memberVars['switchEpochs']) > 0 and self.memberVars['switchEpochs'][0] > 0 and self.zoomingGraph):
            if(PBG.pbTracker.memberVars['maximizingScore']):
                minVal = np.array(accuracies[0:self.memberVars['switchEpochs'][0]]).mean()
                for extraScore in extraScores:
                    minPot = np.array(extraScores[extraScore][0:self.memberVars['switchEpochs'][0]]).mean()
                    if minPot < minVal:
                        minVal = minPot
                ax.set_ylim(ymin=minVal)
            else:
                maxVal = np.array(accuracies[0:self.memberVars['switchEpochs'][0]]).mean()
                for extraScore in extraScores:
                    maxPot = np.array(extraScores[extraScore][0:self.memberVars['switchEpochs'][0]]).mean()
                    if maxPot > maxVal:
                        maxVal = maxPot
                ax.set_ylim(ymax=maxVal)
                
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if(PBG.drawingPB and self.memberVars['doingPB']):
            color = 'r'
            for switcher in self.memberVars['switchEpochs']:
                plt.axvline(x=switcher, ymin=0, ymax=1,color=color)
                if(color == 'r'):
                    color = 'b'
                else:
                    color ='r'
        else:
            for switcher in self.memberVars['nswitchEpochs']:
                plt.axvline(x=switcher, ymin=0, ymax=1,color='b')
        
        # Plot the times for each training epoch
        ax = plt.subplot(222)        
        if(self.memberVars['manualTrainSwitch']):
            ax.plot(np.arange(len(self.memberVars['nTrainTimes'])), self.memberVars['nTrainTimes'], label='Normal Epoch Train Times')
            ax.plot(np.arange(len(self.memberVars['pTrainTimes'])), self.memberVars['pTrainTimes'], label='PB Epoch Train Times')
            ax.plot(np.arange(len(self.memberVars['nValTimes'])), self.memberVars['nValTimes'], label='Normal Epoch Val Times')
            ax.plot(np.arange(len(self.memberVars['pValTimes'])), self.memberVars['pValTimes'], label='PB Epoch Val Times')
            plt.title(saveFolder + '/' + self.saveName + "times (by train() and eval())")
            plt.xlabel('Iteration')
            plt.ylabel('Epoch Time in Seconds ')
            ax.set_ylim(ymin=0)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            
            pd1 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['nTrainTimes'])), 'Normal Epoch Train Times': self.memberVars['nTrainTimes']})
            pd2 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['pTrainTimes'])), 'PB Epoch Train Times': self.memberVars['pTrainTimes']})
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)        
            pd2 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['nValTimes'])), 'Normal Epoch Val Times': self.memberVars['nValTimes']})
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)        
            pd2 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['pValTimes'])), 'PB Epoch Val Times': self.memberVars['pValTimes']})
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)        
            pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'Times.csv', index=False)
            pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
            del pd1, pd2
        else:
            ax.plot(np.arange(len(self.memberVars['nEpochTimes'])), self.memberVars['nEpochTimes'], label='Normal Epoch Times')
            ax.plot(np.arange(len(self.memberVars['pEpochTimes'])), self.memberVars['pEpochTimes'], label='PB Epoch Times')
            plt.title(saveFolder + '/' + self.saveName + "times (by train() and eval())")
            plt.xlabel('Iteration')
            plt.ylabel('Epoch Time in Seconds ')
            ax.set_ylim(ymin=0)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            pd1 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['nEpochTimes'])), 'Normal Epoch Times': self.memberVars['nEpochTimes']})
            pd2 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['pEpochTimes'])), 'PB Epoch Times': self.memberVars['pEpochTimes']})
            pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)
            pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'Times.csv', index=False)
            pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
            del pd1, pd2
        
        
        
        if(self.values_per_train_epoch != -1 and self.values_per_val_epoch != -1):
            ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
            ax2.set_ylabel('Single Datapoint Time in Seconds')  # we already handled the x-label with ax1
            ax2.plot(np.arange(len(self.memberVars['nTrainTimes'])), np.array(self.memberVars['nTrainTimes'])/self.values_per_train_epoch, linestyle='dashed', label='Normal Train Item Times')
            ax2.plot(np.arange(len(self.memberVars['pTrainTimes'])), np.array(self.memberVars['pTrainTimes'])/self.values_per_train_epoch, linestyle='dashed', label='PB Train Item Times')
            ax2.plot(np.arange(len(self.memberVars['nValTimes'])), np.array(self.memberVars['nValTimes'])/self.values_per_val_epoch, linestyle='dashed', label='Normal Val Item Times')
            ax2.plot(np.arange(len(self.memberVars['pValTimes'])), np.array(self.memberVars['pValTimes'])/self.values_per_val_epoch, linestyle='dashed', label='PB Val Item Times')
            ax2.tick_params(axis='y')
            ax2.set_ylim(ymin=0)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        
        
        
        # Plot the learning rates for each training epoch
        ax = plt.subplot(223)        
                
        ax.plot(np.arange(len(self.memberVars['trainingLearningRates'])), self.memberVars['trainingLearningRates'], label='learningRate')
        plt.title(saveFolder + '/' + self.saveName + "learningRate")
        plt.xlabel('Epochs')
        plt.ylabel('learningRate')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        pd1 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['trainingLearningRates'])), 'learningRate': self.memberVars['trainingLearningRates']})
        pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'learningRate.csv', index=False)
        pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
        del pd1


        pd1 = pd.DataFrame({'Switch Number': np.arange(len(self.memberVars['switchEpochs'])), 'Switch Epoch': self.memberVars['switchEpochs']})
        pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'switchEpochs.csv', index=False)
        pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
        del pd1


        pd1 = pd.DataFrame({'Switch Number': np.arange(len(self.memberVars['paramCounts'])), 'Param Count': self.memberVars['paramCounts']})
        pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'paramCounts.csv', index=False)
        pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
        del pd1
        
        #This block creates the testTestScores.csv file
        testScores = self.memberVars['testScores']
        #if not tracking test scores just do validation scores again.
        if(len(self.memberVars['testScores']) == 0):
            testScores = self.memberVars['accuracies']
        if(len(testScores) != len(self.memberVars['accuracies'])):
            print('Your test scores are not the same length as your validation scores')
            print('addTestScore should only be included once, use addExtraScore for other variables')
        switchCounts = len(self.memberVars['switchEpochs']) 
        bestTest = []
        bestValid = []
        assosciatedParams = []
        for switch in range(0,switchCounts,2):
            startIndex = 0
            if(switch != 0):
                startIndex = self.memberVars['switchEpochs'][switch-1] + 1
            endIndex = self.memberVars['switchEpochs'][switch]+1
            if(PBG.pbTracker.memberVars['maximizingScore']):
                bestValidIndex = startIndex + np.argmax(self.memberVars['accuracies'][startIndex:endIndex])
            else:
                bestValidIndex = startIndex + np.argmin(self.memberVars['accuracies'][startIndex:endIndex])
            bestValidScore = self.memberVars['accuracies'][bestValidIndex]
            bestTestScore = testScores[bestValidIndex]
            bestValid.append(bestValidScore)
            bestTest.append(bestTestScore)
            assosciatedParams.append(self.memberVars['paramCounts'][switch])
        # If its neuron training mode but not the very first epoch, 
        # which means the last accuracy was the last one of p mode
        if(self.memberVars['mode'] == 'n' and 
            (
            ((len(self.memberVars['switchEpochs']) == 0) or
                (self.memberVars['switchEpochs'][-1] + 1 != len(self.memberVars['accuracies']))
                ))):
            startIndex = 0
            if(len(self.memberVars['switchEpochs']) != 0):
                startIndex = self.memberVars['switchEpochs'][-1] + 1
            if(PBG.pbTracker.memberVars['maximizingScore']):
                bestValidIndex = startIndex + np.argmax(self.memberVars['accuracies'][startIndex:])
            else:
                bestValidIndex = startIndex + np.argmin(self.memberVars['accuracies'][startIndex:])
            bestValidScore = self.memberVars['accuracies'][bestValidIndex]
            bestTestScore = testScores[bestValidIndex]
            bestValid.append(bestValidScore)
            bestTest.append(bestTestScore)
            assosciatedParams.append(self.memberVars['paramCounts'][-1])
        pd1 = pd.DataFrame({'Param Counts': assosciatedParams, 'Max Valid Scores':bestValid, 'Max Test Scores':bestTest})
        pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'bestTestScores.csv', index=False)
        pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
        del pd1

        # Plot the dendrite learning scores, not used in open source version
        ax = plt.subplot(224)
        if(self.memberVars['doingPB']):
            pd1 = None
            pd2 = None
            NUM_COLORS = len(self.PBNeuronLayerVector)
            if( len(self.PBNeuronLayerVector) > 0 and len(self.memberVars['currentScores'][0]) != 0):
                NUM_COLORS *= 2
            cm = plt.get_cmap('gist_rainbow')
            ax.set_prop_cycle('color', [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)])
            for layerID in range(len(self.PBNeuronLayerVector)):
                ax.plot(np.arange(len(self.memberVars['bestScores'][layerID])), self.memberVars['bestScores'][layerID], label=self.PBNeuronLayerVector[layerID].name)
                pd2 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['bestScores'][layerID])), 'Best ever for all nodes Layer ' + self.PBNeuronLayerVector[layerID].name: self.memberVars['bestScores'][layerID]})
                if(pd1 is None):
                    pd1 = pd2
                else:
                    pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)
                if(len(self.memberVars['currentScores'][layerID]) != 0):
                    ax.plot(np.arange(len(self.memberVars['currentScores'][layerID])), self.memberVars['currentScores'][layerID], label='Best current for all Nodes Layer ' +  self.PBNeuronLayerVector[layerID].name)
                pd2 = pd.DataFrame({'Epochs': np.arange(len(self.memberVars['currentScores'][layerID])), 'Best current for all nodes Layer ' + self.PBNeuronLayerVector[layerID].name: self.memberVars['currentScores'][layerID]})
                pd1 = pd.concat([pd1, pd.DataFrame(pd2)], ignore_index=True)
            plt.title(saveFolder + '/' + self.saveName + " Best PBScores")
            plt.xlabel('Epochs')
            plt.ylabel('Best PBScore')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=math.ceil(len(self.PBNeuronLayerVector)/30))
            for switcher in self.memberVars['pswitchEpochs']:
                plt.axvline(x=switcher, ymin=0, ymax=1,color='r')
            
            if(self.memberVars['mode'] == 'p'):
                missedTime = self.memberVars['numEpochsRun'] - lastImproved
                #T
                plt.axvline(x=(len(self.memberVars['bestScores'][0])-(missedTime+1)), ymin=0, ymax=1,color='g')
                

            
            pd1.to_csv(saveFolder + '/' + self.saveName + extraString + 'Best PBScores.csv', index=False)
            pd1.to_csv('pd.csv', float_format='%.2f', na_rep="NAN!")
            del pd1, pd2

        
        
        fig.tight_layout()
        plt.savefig(saveFolder + '/' + self.saveName+extraString+'.png')
        
        plt.close('all')      

    # Function block to add scores to vectors
    def addLoss(self, loss):
        if (type(loss) is float) == False and (type(loss) is int) == False:
            loss = loss.item()
        self.memberVars['trainingLoss'].append(loss)
    def addLearningRate(self, learningRate):
        if (type(learningRate) is float) == False and (type(learningRate) is int) == False:
               learningRate = learningRate.item()
        self.memberVars['trainingLearningRates'].append(learningRate)
    def addExtraScore(self, score, extraScoreName):
        if (type(score) is float) == False and (type(score) is int) == False:
            try:
                score = score.item()
            except:
                print('Scores added for Perforated Backpropagation should be float, int, or tensor, yours is a:')
                print(type(score))
                print('in addExtraScore')
                pdb.set_trace()
        if(PBG.verbose):
            print('adding extra score %s of %f' % (extraScoreName, float(score)))
        if((extraScoreName in self.memberVars['extraScores']) == False):
                self.memberVars['extraScores'][extraScoreName] = []
        self.memberVars['extraScores'][extraScoreName].append(score)
        if(self.memberVars['mode'] == 'n'):
            if((extraScoreName in self.memberVars['nExtraScores']) == False):
                    self.memberVars['nExtraScores'][extraScoreName] = []
            self.memberVars['nExtraScores'][extraScoreName].append(score)
    def addExtraScoreWithoutGraphing(self, score, extraScoreName):
        if (type(score) is float) == False and (type(score) is int) == False:
            try:
                score = score.item()
            except:
                print('Scores added for Perforated Backpropagation should be float, int, or tensor, yours is a:')
                print(type(score))
                print('in addExtraScoreWithoutGraphing')
                pdb.set_trace()
        if(PBG.verbose):
            print('adding extra score %s of %f' % (extraScoreName, float(score)))
        if((extraScoreName in self.memberVars['extraScoresWithoutGraphing']) == False):
                self.memberVars['extraScoresWithoutGraphing'][extraScoreName] = []
        self.memberVars['extraScoresWithoutGraphing'][extraScoreName].append(score)
    def addTestScore(self, score, extraScoreName):
        self.addExtraScore(score, extraScoreName)
        if (type(score) is float) == False and (type(score) is int) == False:
                try:
                   score = score.item()
                except:
                    print('Scores added for Perforated Backpropagation should be float, int, or tensor, yours is a:')
                    print(type(score))
                    print('in addTestScore')
                    pdb.set_trace()

        if(PBG.verbose):
            print('adding test score %s of %f' % (extraScoreName, float(score)))
        self.memberVars['testScores'].append(score)
    '''
    Function to add the validation score.  This one is more more complex because it determines neuron and dendrite switching
    WARNING: Do not call self anywhere in this function.  When systems get loaded the actual
    tracker you are working with can change and this function should continue with the new tracker's values
    TODO: clean this up, add comments, and separate it into more functions
    '''
    def addValidationScore(self, accuracy, net, forceSwitch=False):
        saveName = PBG.saveName
        for param_group in PBG.pbTracker.memberVars['optimizerInstance'].param_groups:
            learningRate = param_group['lr']
        PBG.pbTracker.addLearningRate(learningRate)        
        
        if(len(PBG.pbTracker.memberVars['paramCounts']) == 0):
            pytorch_total_params = sum(p.numel() for p in net.parameters())
            PBG.pbTracker.memberVars['paramCounts'].append(pytorch_total_params)

        
        if(not PBG.silent):
            print('Adding validation score %.8f' % accuracy)
        # Make sure you are passing in the model and not the dataparallel wrapper
        if issubclass(type(net), nn.DataParallel):
            print('Need to call .module when using add validation score')
            import pdb; pdb.set_trace()
            sys.exit(-1)
        if 'module' in net.__dir__():
            print('Need to call .module when using add validation score')
            import pdb; pdb.set_trace()
            sys.exit(-1)
        if (type(accuracy) is float) == False and (type(accuracy) is int) == False:
            try:
                accuracy = accuracy.item()
            except:
                print('Scores added for Perforated Backpropagation should be float, int, or tensor, yours is a:')
                print(type(accuracy))
                print('in addValidationScore')
                import pdb; pdb.set_trace()

        file_name = 'best_model'
        if(len(PBG.pbTracker.memberVars['switchEpochs']) == 0):
            epochsSinceCycleSwitch = PBG.pbTracker.memberVars['numEpochsRun']
        else:
            epochsSinceCycleSwitch = (PBG.pbTracker.memberVars['numEpochsRun'] - PBG.pbTracker.memberVars['switchEpochs'][-1])
        # Dont update running accuracy during dendrite training
        if(PBG.pbTracker.memberVars['mode'] == 'n' or PBG.learnPBLive):
            if(epochsSinceCycleSwitch < PBG.initialHistoryAfterSwitches):
                if epochsSinceCycleSwitch == 0:
                    PBG.pbTracker.memberVars['runningAccuracy'] = accuracy
                else:
                    PBG.pbTracker.memberVars['runningAccuracy'] = PBG.pbTracker.memberVars['runningAccuracy'] * (1-(1.0/(epochsSinceCycleSwitch+1))) + accuracy * (1.0/(epochsSinceCycleSwitch+1))
            else:
                PBG.pbTracker.memberVars['runningAccuracy'] = PBG.pbTracker.memberVars['runningAccuracy'] * (1.0 - 1.0 / PBG.historyLookback) + accuracy * (1.0 / PBG.historyLookback)

        PBG.pbTracker.memberVars['accuracies'].append(accuracy)
        if(PBG.pbTracker.memberVars['mode'] == 'n'):
            PBG.pbTracker.memberVars['nAccuracies'].append(accuracy)
            
        if PBG.drawingPB or PBG.pbTracker.memberVars['mode'] == 'n' or PBG.learnPBLive:
            PBG.pbTracker.memberVars['runningAccuracies'].append(PBG.pbTracker.memberVars['runningAccuracy'])
        
        PBG.pbTracker.stopEpoch(internalCall=True)
        # If it is neuron training mode
        if(PBG.pbTracker.memberVars['mode'] == 'n') or PBG.learnPBLive:
            if( #score improved, or no score yet, and (always switching or enough time to do a switch)
                (
                    (PBG.pbTracker.memberVars['maximizingScore'] and 
                     (PBG.pbTracker.memberVars['runningAccuracy']*(1.0 - PBG.improvementThreshold) > PBG.pbTracker.memberVars['currentBestValidationScore'])
                    and PBG.pbTracker.memberVars['runningAccuracy'] - PBG.improvementThresholdRaw > PBG.pbTracker.memberVars['currentBestValidationScore'])
                or
                    ((not PBG.pbTracker.memberVars['maximizingScore']) and 
                    (PBG.pbTracker.memberVars['runningAccuracy']*(1.0 + PBG.improvementThreshold) < PBG.pbTracker.memberVars['currentBestValidationScore'])
                    and (PBG.pbTracker.memberVars['runningAccuracy']  + PBG.improvementThresholdRaw) < PBG.pbTracker.memberVars['currentBestValidationScore'])
                or 
                  (PBG.pbTracker.memberVars['currentBestValidationScore'] == 0)
                )
                and
                ((epochsSinceCycleSwitch > PBG.initialHistoryAfterSwitches) or (PBG.pbTracker.memberVars['switchMode'] == PBG.doingSwitchEveryTime))):
                if(PBG.pbTracker.memberVars['maximizingScore']):
                    if(PBG.verbose):
                        print('\n\ngot score of %.10f (average %f, *%f=%f) which is higher than %.10f by %f so setting epoch to %d\n\n' % 
                            (accuracy, 
                            PBG.pbTracker.memberVars['runningAccuracy'], 
                            1-PBG.improvementThreshold,
                            PBG.pbTracker.memberVars['runningAccuracy']*(1.0 - PBG.improvementThreshold),
                            PBG.pbTracker.memberVars['currentBestValidationScore'],
                            PBG.improvementThresholdRaw,
                            PBG.pbTracker.memberVars['numEpochsRun']))
                else:
                    if(PBG.verbose):
                        print('\n\ngot score of %.10f (average %f, *%f=%f) which is lower than %.10f so setting epoch to %d\n\n' %
                              (accuracy, 
                               PBG.pbTracker.memberVars['runningAccuracy'], 
                               1+PBG.improvementThreshold,
                               PBG.pbTracker.memberVars['runningAccuracy']*(1.0 + PBG.improvementThreshold), 
                               PBG.pbTracker.memberVars['currentBestValidationScore'], 
                               PBG.pbTracker.memberVars['numEpochsRun']))
                # Set the new best score
                PBG.pbTracker.memberVars['currentBestValidationScore'] = PBG.pbTracker.memberVars['runningAccuracy']
                if((PBG.pbTracker.memberVars['maximizingScore'] and PBG.pbTracker.memberVars['currentBestValidationScore'] > PBG.pbTracker.memberVars['globalBestValidationScore'])
                   or (not PBG.pbTracker.memberVars['maximizingScore'] and PBG.pbTracker.memberVars['currentBestValidationScore'] < PBG.pbTracker.memberVars['globalBestValidationScore']) or (PBG.pbTracker.memberVars['globalBestValidationScore'] == 0)):
                    if(PBG.verbose):
                        print('this also beats global best of %f so saving' % PBG.pbTracker.memberVars['globalBestValidationScore'])
                    PBG.pbTracker.memberVars['globalBestValidationScore'] = PBG.pbTracker.memberVars['currentBestValidationScore']
                    PBG.pbTracker.memberVars['currentNSetGlobalBest'] = True
                    PBU.saveSystem(net, saveName, file_name)
                    if(PBG.paiSaves):
                        PBU.paiSaveSystem(net, saveName, file_name)
                PBG.pbTracker.memberVars['epochLastImproved'] = PBG.pbTracker.memberVars['numEpochsRun']
                if(PBG.verbose):
                    print('2 epoch improved is %d' % PBG.pbTracker.memberVars['epochLastImproved'])
            else:
                if(PBG.verbose):
                    print('Not saving new best because:')
                    if(epochsSinceCycleSwitch <= PBG.initialHistoryAfterSwitches):
                        print('not enough history since switch%d <= %d' % (epochsSinceCycleSwitch, PBG.initialHistoryAfterSwitches))
                    elif(PBG.pbTracker.memberVars['maximizingScore']):
                        print('got score of %f (average %f, *%f=%f) which is not higher than %f' %(accuracy, PBG.pbTracker.memberVars['runningAccuracy'], 1-PBG.improvementThreshold,PBG.pbTracker.memberVars['runningAccuracy']*(1.0 - PBG.improvementThreshold), PBG.pbTracker.memberVars['currentBestValidationScore']))
                    else:
                        print('got score of %f (average %f, *%f=%f) which is not lower than %f' %(accuracy, PBG.pbTracker.memberVars['runningAccuracy'], 1+PBG.improvementThreshold,PBG.pbTracker.memberVars['runningAccuracy']*(1.0 + PBG.improvementThreshold), PBG.pbTracker.memberVars['currentBestValidationScore']))

                # If its the first epoch save a model so there is never a problem with not finding a model
                if(len(PBG.pbTracker.memberVars['accuracies']) == 1):
                    if(PBG.verbose):
                        print('Saving first model or all models')
                    PBU.saveSystem(net, saveName, file_name)

                    if(PBG.paiSaves):
                        PBU.paiSaveSystem(net, saveName, file_name)

        # Save the latest model
        if(PBG.testSaves):
            PBU.saveSystem(net, saveName, 'latest')
        if(PBG.paiSaves):
            PBU.paiSaveSystem(net, saveName, 'latest')

        PBG.pbTracker.memberVars['lastImprovedAccuracies'].append(PBG.pbTracker.memberVars['epochLastImproved'])
        restructured = False
        # If it is time to switch based on scores and counter
        if((PBG.pbTracker.switchTime() == True) or forceSwitch):
            # If testing dendrite capacity switch after enough dendrites have been added
            if((PBG.pbTracker.memberVars['mode'] == 'n') and
                (PBG.pbTracker.memberVars['numPBNeuronLayers'] > 3)
                and PBG.testingDendriteCapacity):
                PBG.pbTracker.saveGraphs()
                print('Successfully added 3 dendrites with PBG.testingDendriteCapacity = True (default).  You may now set that to False and run a real experiment.')
                # set_trace is here for huggingface which doesnt end cleanly
                import pdb; pdb.set_trace()
                # net, did not restructure, training is over
                return net, False, True
            
            # If its doing neuron training but this dendrite count didn't improve upon previous one
            if(((PBG.pbTracker.memberVars['mode'] == 'n') or PBG.learnPBLive)
               and (PBG.pbTracker.memberVars['currentNSetGlobalBest'] == False)
               ): #then restart with a new set of PB nodes
                if(PBG.verbose):
                    print('Planning to switch to p mode but best beat last: %d current start lr steps: %f and last maximum lr steps: %d for rate: %.8f' % (PBG.pbTracker.memberVars['currentNSetGlobalBest'],
                                                                        PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'], PBG.pbTracker.memberVars['lastMaxLearningRateSteps'], PBG.pbTracker.memberVars['lastMaxLearningRateValue'])) 
                now = datetime.now()
                dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")
                if(PBG.verbose):
                    print('1 saving break %s' % (dt_string+'_noImprove_lr_'+str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'])))
                PBG.pbTracker.saveGraphs(dt_string+'_noImprove_lr_'+str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))

                if(PBG.pbTracker.memberVars['numDendriteTries'] < (PBG.maxDendriteTries)):
                    if(not PBG.silent):
                        print('Dendrites did not improve but current tries %d is less than max tries %d so loading last switch and trying new Dendrites.' % (PBG.pbTracker.memberVars['numDendriteTries'], PBG.maxDendriteTries))
                    oldTries = PBG.pbTracker.memberVars['numDendriteTries']
                    # If its here it didn't improve so changing learning modes to p again will load the best model which is from the previous n mode not this one.
                    net = PBU.changeLearningModes(net, saveName, file_name, PBG.pbTracker.memberVars['doingPB'])
                    PBG.pbTracker.memberVars['numDendriteTries'] = oldTries + 1
                else:
                    if(not PBG.silent):
                        print('Dendrites did not improve system and %d >= %f so returning trainingComplete.' % (PBG.pbTracker.memberVars['numDendriteTries'], PBG.maxDendriteTries))
                        print('You should now exit your training loop and best_model will be your final model for inference')
                    PBU.loadSystem(net, saveName, file_name, switchCall=True)
                    PBG.pbTracker.saveGraphs()
                    PBU.paiSaveSystem(net, saveName, 'final_clean')
                    return net, True, True
            # Else if did improve keep the dendrites and switch back to a new p mode adding more
            else: 
                if(PBG.verbose):
                    print('calling switchMode with %d, %d, %d, %f' % (PBG.pbTracker.memberVars['currentNSetGlobalBest'],
                                                                        PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'], PBG.pbTracker.memberVars['lastMaxLearningRateSteps'], PBG.pbTracker.memberVars['lastMaxLearningRateValue']))
                if((PBG.pbTracker.memberVars['mode'] == 'n') and 
                   (PBG.maxDendrites == PBG.pbTracker.memberVars['numPBNeuronLayers'])):
                    if(not PBG.silent):
                        print('Last Dendrites were good and this hit the max of %d' % (PBG.maxDendrites))
                    PBU.loadSystem(net, saveName, file_name, switchCall=True)
                    PBG.pbTracker.saveGraphs()
                    PBU.paiSaveSystem(net, saveName, 'final_clean')
                    return net, True, True
                if(PBG.pbTracker.memberVars['mode'] == 'n'):
                    PBG.pbTracker.memberVars['numDendriteTries'] = 0
                    if(PBG.verbose):
                        print('Adding new dendrites without resetting which means the last ones improved.  Resetting numDendriteTries')
                PBG.pbTracker.saveGraphs('_beforeSwitch_'+str(len(PBG.pbTracker.memberVars['switchEpochs'])))
                if(PBG.testSaves):
                    PBU.saveSystem(net, saveName, 'beforeSwitch_' + str(len(PBG.pbTracker.memberVars['switchEpochs'])))
                    # In addition to saving the system also copy the current best model from this set of dendrites
                    shutil.copyfile(saveName+'/best_model.pt', saveName+'/best_model_beforeSwitch_' + str(len(PBG.pbTracker.memberVars['switchEpochs'])) + '.pt')
                    if(PBG.extraVerbose):
                        import pdb; pdb.set_trace()
                    net = PBU.changeLearningModes(net, saveName, file_name, PBG.pbTracker.memberVars['doingPB'])
            
            # If restructured is true then you're just about to reset the scheduler and optimizer to clear them before saving
            restructured = True
            PBG.pbTracker.clearOptimizerAndScheduler() 
            # Save the model from after the switch
            PBU.saveSystem(net, saveName, 'switch_' + str(len(PBG.pbTracker.memberVars['switchEpochs'])))
            
        # If its not time to switch and you have a scheduler, increment it
        elif(PBG.pbTracker.memberVars['scheduler'] != None):
            '''
            Need to add more comments inline here but the process is as follows
                1 To find the best initial learning rate for dendrites start at the default rate
                2 learn at that rate until scheduler has incremented twice
                3 save that version, and then start the dendrites at LR current increment - 1
                4 repeat 2 and 3 until finding a version where the additional repeat has a worse final score at a set LR
                5 Load the previous model that had the best accuracy at that LR as the initial rate
            '''
            for param_group in PBG.pbTracker.memberVars['optimizerInstance'].param_groups:
                learningRate1 = param_group['lr']
            if(type(PBG.pbTracker.memberVars['schedulerInstance']) is torch.optim.lr_scheduler.ReduceLROnPlateau):
                if(epochsSinceCycleSwitch > PBG.initialHistoryAfterSwitches or PBG.pbTracker.memberVars['mode'] == 'p'):
                    if(PBG.verbose):
                        print('updating scheduler with last improved %d from current %d' % (PBG.pbTracker.memberVars['epochLastImproved'],PBG.pbTracker.memberVars['numEpochsRun']))
                    if(PBG.pbTracker.memberVars['scheduler'] != None):
                        PBG.pbTracker.memberVars['schedulerInstance'].step(metrics=accuracy)
                        if(PBG.pbTracker.memberVars['scheduler'] is torch.optim.lr_scheduler.ReduceLROnPlateau):
                            if(PBG.verbose):
                                print('scheduler is now at %d bad epochs' % PBG.pbTracker.memberVars['schedulerInstance'].num_bad_epochs)
                else:
                    if(PBG.verbose):
                        print('not stepping optimizer since hasnt initialized')
            elif(PBG.pbTracker.memberVars['scheduler'] != None):
                if(epochsSinceCycleSwitch > PBG.initialHistoryAfterSwitches or PBG.pbTracker.memberVars['mode'] == 'p'):
                    if(PBG.verbose):
                        print('incrementing scheduler to count %d' % PBG.pbTracker.memberVars['schedulerInstance']._step_count)
                    PBG.pbTracker.memberVars['schedulerInstance'].step()
                    if(PBG.pbTracker.memberVars['scheduler'] is torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if(PBG.verbose):
                            print('scheduler is now at %d bad epochs' % PBG.pbTracker.memberVars['schedulerInstance'].num_bad_epochs)
            if(epochsSinceCycleSwitch <= PBG.initialHistoryAfterSwitches and PBG.pbTracker.memberVars['mode'] == 'n'):
                if(PBG.verbose):
                    print('not stepping with history %d and current %d' % (PBG.initialHistoryAfterSwitches, epochsSinceCycleSwitch))
            for param_group in PBG.pbTracker.memberVars['optimizerInstance'].param_groups:
                learningRate2 = param_group['lr']
            stepped = False
            atLastCount = False
            if(PBG.verbose):
                print('checking if at last with scores %d, count since switch %d and last total lr step count %d' % (len(PBG.pbTracker.memberVars['currentCycleLRMaxScores']), epochsSinceCycleSwitch, PBG.pbTracker.memberVars['initialLRTestEpochCount']))
            #Then if either it is double that (first value 1->2) or exactly that, 
            #(start at 2) then go into this check even though the learning rate didnt just step because it might never again 
            if(((len(PBG.pbTracker.memberVars['currentCycleLRMaxScores']) == 0) and epochsSinceCycleSwitch == PBG.pbTracker.memberVars['initialLRTestEpochCount']*2)
               or ((len(PBG.pbTracker.memberVars['currentCycleLRMaxScores']) == 1) and epochsSinceCycleSwitch == PBG.pbTracker.memberVars['initialLRTestEpochCount'])):
                atLastCount = True
            if(PBG.verbose):
                print('at last count %d with count %d and last LR count %d' % (atLastCount, epochsSinceCycleSwitch,  PBG.pbTracker.memberVars['initialLRTestEpochCount']))
            
            if(learningRate1 != learningRate2):
                stepped = True
                PBG.pbTracker.memberVars['currentStepCount'] += 1
                if(PBG.verbose):
                    print('learning learning rate just stepped to %.10e with %d total steps' % (learningRate2, PBG.pbTracker.memberVars['currentStepCount']))
                if(PBG.pbTracker.memberVars['currentStepCount'] == PBG.pbTracker.memberVars['lastMaxLearningRateSteps']):
                    if(PBG.verbose):
                        print('%d steps is the max of the last switch mode' % PBG.pbTracker.memberVars['currentStepCount'])
                    #If this was the first step and it is the max then set it.  Want to set when 1->2 gets to 2, not when 0->1 hits 2 as its stopping point
                    if(PBG.pbTracker.memberVars['currentStepCount'] - PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'] == 1):
                        PBG.pbTracker.memberVars['initialLRTestEpochCount'] = epochsSinceCycleSwitch

            if(PBG.verbose):
                print('learning rates were %.8e and %.8e started with %f, and is now at %d commited %d then either this (non zero) or eventually comparing to %d steps or rate %.8f' %
                                                        (learningRate1, learningRate2, 
                                                         PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'],
                                                         PBG.pbTracker.memberVars['currentStepCount'],
                                                         PBG.pbTracker.memberVars['committedToInitialRate'],
                                                         PBG.pbTracker.memberVars['lastMaxLearningRateSteps'],
                                                         PBG.pbTracker.memberVars['lastMaxLearningRateValue']))
            

            #if the learning rate just stepped check in on the restart at lower rate
            if((PBG.pbTracker.memberVars['scheduler'] != None) 
                #if its currently in n mode, or its learning live, i.e. if it potentially might have higher accuracy
                and ((PBG.pbTracker.memberVars['mode'] == 'n') or PBG.learnPBLive) 
                #and the learning rate just stepped
                and (stepped or atLastCount)): 
                #if it hasnt commited to a learning rate for this cycle yet
                if(PBG.pbTracker.memberVars['committedToInitialRate'] == False): 
                    bestScoreSoFar = PBG.pbTracker.memberVars['globalBestValidationScore']
                    if(PBG.verbose):
                        print('in statements to check next learning rate with stepped %d and max count %d' % (stepped, atLastCount))
                    # If there are currently no scores saved for this dendrite
                    if(len(PBG.pbTracker.memberVars['currentCycleLRMaxScores']) == 0 
                        # and that initial LR test just did its second step
                        and (PBG.pbTracker.memberVars['currentStepCount'] - PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'] == 2
                        #or it didnt do a second step, but the second LR epochs has matched the epoch count of the first LR
                        or atLastCount)): 
                        restructured = True
                        PBG.pbTracker.clearOptimizerAndScheduler() 
                        # Save the system for this initial condition
                        # Save old global so if it doesnt beat it it wont overwrite during loading
                        oldGlobal = PBG.pbTracker.memberVars['globalBestValidationScore']
                        # Save old accuracy to track it
                        oldAccuracy = PBG.pbTracker.memberVars['currentBestValidationScore']
                        # If old counts is not -1 that means its on the last max learning rate so want to retain it and use the same one for the next time
                        oldCounts = PBG.pbTracker.memberVars['initialLRTestEpochCount']
                        skip1 = PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']
                        now = datetime.now()
                        dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")
                        PBG.pbTracker.saveGraphs(dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))
                        if(PBG.testSaves):
                            PBU.saveSystem(net, saveName, 'PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_'  + str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))
                        if(PBG.verbose):
                            print('saving with initial steps: %s with current best %f' % (dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']), oldAccuracy))
                        # Then load back at the start and try with the lower initial learning rate
                        net = PBU.loadSystem(net, saveName, 'switch_' + str(len(PBG.pbTracker.memberVars['switchEpochs'])), switchCall=True)
                        PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'] = skip1 + 1
                        # If this next one is going to be at the min learning rate of last switch mode
                        PBG.pbTracker.memberVars['currentCycleLRMaxScores'].append(oldAccuracy)
                        PBG.pbTracker.memberVars['globalBestValidationScore'] = oldGlobal
                        PBG.pbTracker.memberVars['initialLRTestEpochCount'] = oldCounts
                    # If there is one score already, then this is the first step at the next score
                    elif(len(PBG.pbTracker.memberVars['currentCycleLRMaxScores']) == 1):
                        PBG.pbTracker.memberVars['currentCycleLRMaxScores'].append(PBG.pbTracker.memberVars['currentBestValidationScore'])
                        # If this LRs score was worse than the last LRs score
                        if((PBG.pbTracker.memberVars['maximizingScore'] 
                            and PBG.pbTracker.memberVars['currentCycleLRMaxScores'][0] > PBG.pbTracker.memberVars['currentCycleLRMaxScores'][1])

                           or ((not PBG.pbTracker.memberVars['maximizingScore']) 
                           and PBG.pbTracker.memberVars['currentCycleLRMaxScores'][0] < PBG.pbTracker.memberVars['currentCycleLRMaxScores'][1])):
                           
                            restructured = True
                            PBG.pbTracker.clearOptimizerAndScheduler() 
                            # Then reload the last one, and start training
                            if(PBG.verbose):
                                print('Got initial %d step score %f and %d score at step %f so loading old score' % (PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']-1,PBG.pbTracker.memberVars['currentCycleLRMaxScores'][0], PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'],PBG.pbTracker.memberVars['currentCycleLRMaxScores'][1])) 
                            priorBest = PBG.pbTracker.memberVars['currentCycleLRMaxScores'][0]
                            now = datetime.now()
                            dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")
                            PBG.pbTracker.saveGraphs(dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))
                            if(PBG.testSaves):
                                PBU.saveSystem(net, saveName, 'PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_'  + str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))
                            if(PBG.verbose):
                                print('saving with initial steps: %s' % (dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'])))
                            if(PBG.testSaves):
                                net = PBU.loadSystem(net, saveName, 'PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_'  + str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']-1), switchCall=True)
                            # Also save graphs for this one that gets chosen
                            now = datetime.now()
                            dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")
                            PBG.pbTracker.saveGraphs(dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']) + 'PICKED')
                            if(PBG.testSaves):
                                PBU.saveSystem(net, saveName, 'PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_'  + str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))
                            if(PBG.verbose):
                                print('saving with initial steps: %s' % (dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'])))
                            PBG.pbTracker.memberVars['committedToInitialRate'] = True
                            PBG.pbTracker.memberVars['lastMaxLearningRateSteps'] = PBG.pbTracker.memberVars['currentStepCount']
                            PBG.pbTracker.memberVars['lastMaxLearningRateValue'] = learningRate2
                            #set the best score to be the higher schore to not overwrite it
                            PBG.pbTracker.memberVars['currentBestValidationScore'] = priorBest
                            if(PBG.verbose):
                                print('Setting laxt max steps to %d and lr %f' % (PBG.pbTracker.memberVars['lastMaxLearningRateSteps'], PBG.pbTracker.memberVars['lastMaxLearningRateValue']))
                        else: # If the current LR score is better then want to check the next lower LR without reloading
                            if(PBG.verbose):
                                print('Got initial %d step score %f and %d score at step %f so NOT loading old score and continuing with this score' % (PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']-1,PBG.pbTracker.memberVars['currentCycleLRMaxScores'][0], PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'],PBG.pbTracker.memberVars['currentCycleLRMaxScores'][1])) 
                            if(atLastCount): #If this is the last one though, then also set it to be the one that is picked (dont use LR lower than previous dendrites lowest)

                                restructured = True
                                PBG.pbTracker.clearOptimizerAndScheduler() 
                                now = datetime.now()
                                dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")
                                PBG.pbTracker.saveGraphs(dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']) + 'PICKED')
                                if(PBG.testSaves):
                                    PBU.saveSystem(net, saveName, 'PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_'  + str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps']))
                                if(PBG.verbose):
                                    print('saving with initial steps: %s' % (dt_string+'_PBCount_' + str(PBG.pbTracker.memberVars['numPBNeuronLayers']) + '_startSteps_' +str(PBG.pbTracker.memberVars['currentNLearningRateInitialSkipSteps'])))
                                PBG.pbTracker.memberVars['committedToInitialRate'] = True
                                PBG.pbTracker.memberVars['lastMaxLearningRateSteps'] = PBG.pbTracker.memberVars['currentStepCount']
                                PBG.pbTracker.memberVars['lastMaxLearningRateValue'] = learningRate2
                                if(PBG.verbose):
                                    print('Setting laxt max steps to %d and lr %f' % (PBG.pbTracker.memberVars['lastMaxLearningRateSteps'], PBG.pbTracker.memberVars['lastMaxLearningRateValue']))
                        PBG.pbTracker.memberVars['currentCycleLRMaxScores'] = []
                    elif(len(PBG.pbTracker.memberVars['currentCycleLRMaxScores']) == 2):
                        print('Shouldnt ever be here.  Please let Perforated AI know if this happened.')
                        pdb.set_trace()
                    PBG.pbTracker.memberVars['globalBestValidationScore'] = bestScoreSoFar
                else:
                    if(PBG.verbose):
                        print('Setting last max steps to %d and lr %f' % (PBG.pbTracker.memberVars['lastMaxLearningRateSteps'], PBG.pbTracker.memberVars['lastMaxLearningRateValue']))
                    PBG.pbTracker.memberVars['lastMaxLearningRateSteps'] += 1
                    PBG.pbTracker.memberVars['lastMaxLearningRateValue'] = learningRate2
        
        PBG.pbTracker.startEpoch(internalCall=True)
        PBG.pbTracker.saveGraphs()
        if(restructured):
            PBG.pbTracker.memberVars['epochLastImproved'] = PBG.pbTracker.memberVars['numEpochsRun']
            if(PBG.verbose):
                print('Setting epoch last improved to %d' % PBG.pbTracker.memberVars['epochLastImproved'])
            now = datetime.now()
            dt_string = now.strftime("_%d.%m.%Y.%H.%M.%S")
            if(PBG.verbose):
                print('not saving restructure right now')
            for param in net.parameters(): param.data = param.data.contiguous()

        if(PBG.verbose):
            print('completed adding score.  restructured is %d, \ncurrent switch list is:' % (restructured))
            print(PBG.pbTracker.memberVars['switchEpochs'])
                
        # Always False for training complete if nothing triggered that training is over
        return net, restructured, False 
            
    def clearAllProcessors(self):
        for layer in self.PBNeuronLayerVector:
            layer.clearProcessors()

    def addPBLayer(self):
        for layer in self.PBNeuronLayerVector:
            layer.addPBLayer()
