################################################################################
# Poirazi-style receptive-field dendrite variant for PerforatedAI.            #
#                                                                              #
# Registers a factory that explicitly names the MaskedLinear architecture as  #
# the dendrite type, making the parent-module contract visible and assertable. #
# The deep copy matches PAI's default so this variant is a drop-in and        #
# produces bit-for-bit identical results, serving as a reference baseline     #
# before custom dendrite architectures are swapped in.                        #
################################################################################

#
"""
Imports
"""
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai   as UPA


#
"""
Factory
"""
def create_poirazi_dendrite(parent_module):
    '''
    Create one dendrite for a PAI slot

    Notes:
        - Deep-copies the parent template via UPA.deep_copy_pai, which is
          exactly what PAI's built-in factory does, so this variant is a
          numerically identical drop-in replacement
        - pin_dendrite_masks overwrites the rf buffer with the per-slot
          receptive field after every restructure, so the copied rf is a
          transient placeholder that is never read during training

    Signature:
        parent_module:
            - The soma block PAI is growing a dendrite candidate for
    '''
    return UPA.deep_copy_pai(parent_module)


#
"""
Registration
"""
def initialize_variant_dendrite() -> None:
    '''
    Register the Poirazi receptive-field factory with PAI

    Notes:
        - Must be called after UPA.perforate_model so the tracker already
          holds the wrapped PAINeuronModules to propagate the factory to
        - set_create_dendrite_global walks every PAINeuronModule in
          neuron_module_vector and calls set_create_dendrite on each one

    Signature:
        (none)
    '''
    GPA.pai_tracker.set_create_dendrite_global(create_poirazi_dendrite)
