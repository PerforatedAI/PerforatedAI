################################################################################
# Poirazi-style receptive-field dendrite variant for PerforatedAI.            #
################################################################################

#
"""
Imports
"""
from typing import Optional, Tuple

import numpy as np
import torch

from perforatedai import globals_perforatedai as GPA

from .model import MaskedLinear


#
"""
State
"""
_rf_mode     : str                    = 'random'
_synapses    : int                    = 0
_img_shape   : Optional[Tuple]        = None
_soma_centers: dict                   = {}


#
"""
Receptive Field Helpers
"""
def _nb_vals(
    matrix   : np.ndarray,
    indices,
    size     : int  = 1,
    perimeter: bool = False,
) -> np.ndarray:
    M, N  = matrix.shape
    r     = int(np.atleast_1d(indices)[0])
    c     = int(np.atleast_1d(indices)[1])
    r_min = max(0, r - size)
    r_max = min(M - 1, r + size)
    c_min = max(0, c - size)
    c_max = min(N - 1, c + size)
    rr, cc = np.meshgrid(
        np.arange(r_min, r_max + 1),
        np.arange(c_min, c_max + 1),
        indexing = 'ij',
    )
    if perimeter:
        dist = np.maximum(np.abs(rr - r), np.abs(cc - c))
        return np.column_stack((rr[dist == size], cc[dist == size]))
    return np.column_stack((rr.flatten(), cc.flatten()))


def _allocate_synapses(
    nb          : tuple,
    matrix      : np.ndarray,
    num_synapses: int,
    num_channels: int = 1,
) -> np.ndarray:
    M, N        = matrix.shape
    mask        = np.zeros((M, N))
    syn_indices = _nb_vals(matrix, list(nb))

    if len(syn_indices) < num_synapses:
        radius = 2
        while len(syn_indices) < num_synapses:
            extra = _nb_vals(matrix, list(nb), size = radius, perimeter = True)
            if len(extra) == 0:
                break
            diff = num_synapses - len(syn_indices)
            if len(extra) > diff:
                extra = extra[np.random.choice(len(extra), diff, replace = False)]
            syn_indices = np.concatenate((syn_indices, extra))
            radius += 1
    elif len(syn_indices) > num_synapses:
        syn_indices = syn_indices[
            np.random.choice(len(syn_indices), num_synapses, replace = False)
        ]

    mask[syn_indices[:, 0], syn_indices[:, 1]] = 1
    if num_channels > 1:
        mask = np.tile(np.expand_dims(mask, axis = 2), (1, 1, num_channels))
    return mask.reshape(M * N * num_channels)


#
"""
Factory
"""
def create_poirazi_dendrite(parent_module: MaskedLinear) -> MaskedLinear:
    '''
    Create one dendrite for a PAI slot

    Notes:
        - rf_mode controls the connectivity style (set via initialize_variant_dendrite):
            all_to_all: every input is connected
            random:     uniform random sparse connections
            somatic:    spatially local patch, shared center per soma
            dendritic:  spatially local patch, independent center per dendrite
        - Constructs a fresh MaskedLinear with Xavier weight init and zero bias

    Signature:
        parent_module (MaskedLinear):
            - The soma block PAI is growing a dendrite candidate for
    '''
    out_f, in_f = parent_module.weight.shape
    synapses    = _synapses

    if _rf_mode == 'all_to_all':
        rf = torch.ones(out_f, in_f)

    elif _rf_mode == 'random':
        rf = torch.zeros(out_f, in_f)
        for j in range(out_f):
            rf[j, torch.randperm(in_f)[:synapses]] = 1.0

    elif _rf_mode in ('somatic', 'dendritic'):
        W, H, C  = _img_shape
        matrix   = np.zeros((W, H))
        rf_np    = np.zeros((out_f, W * H * C))

        if _rf_mode == 'somatic':
            soma_id = id(parent_module)
            if soma_id not in _soma_centers:
                _soma_centers[soma_id] = (
                    int(np.random.randint(0, W)),
                    int(np.random.randint(0, H)),
                )
            center = _soma_centers[soma_id]
            for j in range(out_f):
                rf_np[j] = _allocate_synapses(center, matrix, synapses, C)

        else:
            for j in range(out_f):
                center   = (int(np.random.randint(0, W)), int(np.random.randint(0, H)))
                rf_np[j] = _allocate_synapses(center, matrix, synapses, C)

        rf = torch.as_tensor(rf_np, dtype = torch.float32)

    return MaskedLinear(in_f, out_f, rf)


#
"""
Registration
"""
def initialize_variant_dendrite(
    synapses : int,
    rf_mode  : str             = 'random',
    img_shape: Optional[Tuple] = None,
) -> None:
    '''
    Register the dendrite factory with PAI

    Notes:
        - rf_mode selects the connectivity style for new dendrites
        - img_shape is required for somatic and dendritic modes
        - _soma_centers is cleared on each call so re-runs start fresh

    Signature:
        rf_mode (str):
            - One of all_to_all, random, somatic, dendritic
        img_shape (Optional[Tuple]):
            - (width, height, channels) of the input image; required for
              somatic and dendritic modes
    '''
    global _rf_mode, _synapses, _img_shape, _soma_centers
    _rf_mode      = rf_mode
    _synapses     = synapses
    _img_shape    = img_shape
    _soma_centers = {}
    GPA.pai_tracker.set_create_dendrite_global(create_poirazi_dendrite)
