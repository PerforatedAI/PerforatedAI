# Copyright (c) 2025 Perforated AI
############### PB configuration file ###############
import math
import torch 
import torch.nn as nn
import sys

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Pointer to the PB Tracker which handles adding dendrites
pbTracker = []

# Debug input dimensions all at once
debuggingInputDimensions = 0
# Debugging input tensor sizes.  
# This will slow things down very slightly and is not neccesary but can help catch when dimensions were not filled in correctly.
confirmCorrectSizes = False

# Confirm options chosen that are not reccomended
unwrappedModulesConfirmed = False
weightDecayAccepted = False
checkedSkippedLayers = False

# Print debugging on epoch scale or batch scale
verbose = False
extraVerbose = False
# Supress all PB prints
silent = False

# Save additional scores to see what would have happened without the addition of dendrites
saveOldGraphScores = True

# Run a test to add a few dendrites quickly before running a full experiment
testingDendriteCapacity = True

# Use safe tensors rather than torch.save
usingSafeTensors = True

'''
This take in an array of layers.  for example:

    PBG..PBSequential([nn.Linear(2 * hidden_dim, seqWidth),
            nn.LayerNorm(seqWidth)])
    
    This should be used for:
        -all normalization layers
    This can be used for:
        -final output layer and softmax
'''
class PBSequential(nn.Sequential):
        def __init__(self, layerArray):
            super(PBSequential, self).__init__()
            self.model = nn.Sequential(*layerArray)
        def forward(self, *args, **kwargs):
            return self.model(*args, **kwargs)

# Lists for module types and names to add dendrites to
modulesToConvert = [nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear]
moduleNamesToConvert = ['PBSequential']
# All modules should either be converted or tracked to ensure all modules are accounted for
modulesToTrack = []
moduleNamesToTrack = []
moduleIDsToTrack = []
# Relacement modules happen before the conversion, 
# so replaced modules will then also be run through the converstion steps
# These are for modules that need to be replaced before addition of dendrites
# See the resnet example in pb_models
modulesToReplace = []
# Modules to replace the above modules with
replacementModules = []

# Dendrites default to modules which are one tensor input and one tensor output in forward()
# Other modules require to be labeled as modules with processing and assigned procesing classes
# This can be done by module type or module name see customization.md in API for example
modulesWithProcessing = []
moduleProcessingClasses = []
moduleNamesWithProcessing = []
moduleByNameProcessingClasses = []

# Module names can be added to this list to not convert a specific module even though other ones are converted
# It is reccomended to not use this, but if you are working with pretrained model that you can't edit
# and a module is causing problems, you can add it here to skip it
moduleNamesToSkip = []

# Similarly here as above.  Some huggingface models have multiple pointers to the same modules which cause problems
# If you want to only save one of the multiple pointers you can set which ones not to save here
moduleNamesToNotSave = ['.base_model']

# In place for future implimentation options of adding multiple candidate dendrites together
globalCandidates = 1

# A graph setting which can be set to false if you want to do your own training visualizaitons
drawingPB = True
# Saving test intermediary models, good for experimentation, bad for memory
testSaves = True
# To be filled in later.  paiSaves will remove some extra scaffolding for slight memory and speed improvements
paiSaves = False

# inputDimensions needs to be set every time. It is set to what format of planes you are expecting.  
# Neuron index should be set to 0, variable indexes should be set to -1.  For example, if your format is [batchsize, nodes, x, y]
# inputDimensions is [-1,0,-1-1].  
# if your format is, [batchsize, time index, nodes] inputDimensions is [-1,-1,0]
inputDimensions = [-1, 0, -1, -1]

# Percentage Improvement increase needed to call a new best validation score
improvementThreshold = 0.0001
# Raw increase needed
improvementThresholdRaw = 1e-5

# Multiplier when randomizing dendrite weights
candidateWeightInitializationMultiplier = 0.01

# SWITCH MODE SETTINGS

# Add dendrites every time to debug implimentation
doingSwitchEveryTime = 0

# Switch when validation hasn't improved over x epochs
doingHistory = 1
# Epochs to try before deciding to load previous best and add dendrites
# Be sure this is higher than scheduler patience
nEpochsToSwitch = 10  
# Number to average validation scores over
historyLookback = 1
# Amount of epochs to run after adding a new set of dendrites before checking to add more
initialHistoryAfterSwitches = 0

# Switch after a fixed number of epochs
doingFixedSwitch = 2
# Number of epochs to complete before switching
fixedSwitchNum = 250
# A additional flag if you want your first switch to occer later than all the rest for initial pretraining
firstFixedSwitchNum = 249

# A setting to not add dendrites and just do regular training
# Warning, this will also never trigger trainingCompelte
doingNoSwitch = 3

# Default
switchMode = doingHistory

# Resets score on switch
# This can be useful if you need many epochs to catch up to the best score
# from the previous version after adding dendrites
resetBestScoreOnSwitch = False

# Not used in open source implimentation, leave as default
learnPBLive = False
noExtraNModes = True

# Type for new modules and dendrite to dendrite / dendrite to neuron weights
dType = torch.float

# A setting to keep dendrites even if they do not improve scores
retainAllPB = False

# A setting to automatically sweep over previously used learning rates when adding new dendrites
# Sometimes it's best to go back to initial LR, but often its best to start at a lower LR
findBestLR = True
# Enforces the above even if the previous epoch didnt lower the learning rate
dontGiveUpUnlessLearningRateLowered = True

# Set to 1 if you want to quit as soon as one dendrite fails
# Higher values will try new random dendrite weights this many times before accepting that more dendrites dont improve
maxDendriteTries = 5
# Max dendrites to add even if they do continue improving scores
maxDendrites = 100

# Settings to initialize the a scheduler
# Have learning rate params be by total epoch
paramValsByTotalEpoch = 0
# Reset the params at every switch
paramValsByUpdateEpoch = 1
# Reset params for PBStarts but not for Normal restarts
# Not used for open source version
paramValsByNormalEpochStart = 2
# Default setting
paramValsSetting = paramValsByUpdateEpoch

# The activation function to use for dendrites
PBForwardFunction = torch.sigmoid
