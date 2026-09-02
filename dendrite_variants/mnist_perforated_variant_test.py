################################################################################
# Author:                                                                      #
# Nicholas Mesa-Cucalon (nicholas@perforatedai.com)                            #
#                                                                              #
# Variant-framework re-implementation of main_pai.py.                         #
#                                                                              #
# Functionally identical to main_pai.py but registers the dendrite factory    #
# explicitly via the variant framework instead of relying on PAI's built-in   #
# deep-copy default.  Self-contained: all model, mask, and training code is   #
# inlined so this file has no dependency on the internal-projects tree.       #
#                                                                              #
# Run from dendrite_variants/ or any working directory:                       #
#   CUDA_VISIBLE_DEVICES=0 python mnist_perforated_variant_test.py \          #
#     GPU SEQ ESTOP TRIAL MODEL_TYPE SIGMA DATASET \                          #
#     NUM_DENDS NUM_SOMA NUM_LAYERS SYNAPSES \                                 #
#     DROP_FLAG DROP_RATE LR PAI_OUT \                                        #
#     4 0 none 160                                                             #
#                                                                              #
# DATASETS are expected at DATASETS/ beside this script, or torchvision will  #
# download them there automatically.                                           #
################################################################################

#
"""
Imports
"""
import os
import sys
import copy
import time
import csv
import json
import torch
import random
import pickle
import pathlib
import numpy as np
import torchvision

from dataclasses      import dataclass
from datetime         import datetime, timezone
from torch            import Tensor, nn
from torch.nn         import functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing           import Any, Dict, List, Optional, Tuple

# Only this script's directory needs to be on the path, for the local
# poirazi_receptive_field_dendrites package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai   as UPA

from poirazi_receptive_field_dendrites import (
    initialize_variant_dendrite,
    MaskedLinear,
    PerforatedDendriticANN,
)


#
"""
Receptive Fields
"""
def nb_vals(
    matrix   : np.ndarray,
    indices  : List[int],
    size     : int  = 1,
    perimeter: bool = False,
) -> np.ndarray:
    M, N = matrix.shape
    r = int(np.atleast_1d(indices)[0])
    c = int(np.atleast_1d(indices)[1])

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
        mask = (dist == size)
        return np.column_stack((rr[mask], cc[mask]))

    return np.column_stack((rr.flatten(), cc.flatten()))


def random_connectivity(
    inputs : int,
    outputs: int,
    opt    : str           = 'random',
    conns  : Optional[int] = None,
    rng    : Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros(shape = (inputs, outputs))
    if opt == 'one_to_one':
        idxs = rng.integers(
            low  = 0,
            high = mask.shape[0],
            size = mask.shape[1],
        )
        for i in range(mask.shape[1]):
            mask[idxs[i], i] = 1

    elif opt == 'random':
        if conns is None or conns <= 0 or not isinstance(conns, int):
            raise ValueError('Specify `conns` as positive integer.')
        elif conns > mask.size:
            raise ValueError(
                'Specify `conns` as positive integer lower than `inputs*outputs`'
            )
        indices = rng.choice(inputs * outputs, conns, replace = False)
        mask.flat[indices] = 1

    elif opt == 'constant':
        if conns is None or conns <= 0 or not isinstance(conns, int):
            raise ValueError('Specify `conns` as positive integer.')
        if conns > mask.shape[0]:
            raise ValueError('`conns` cannot be more than input nodes.')
        for i in range(mask.shape[1]):
            idx = rng.choice(mask.shape[0], conns, replace = False)
            mask[idx, i] = 1
    else:
        raise ValueError(
            'Not a valid option. `opt` should be `one_to_one`, `random` or `constant`'
        )

    return mask.T.astype('int')


def connectivity(inputs: int, outputs: int) -> np.ndarray:
    if outputs <= 0:
        raise ValueError('Number of outputs must be greater than zero.')
    if inputs <= 0:
        raise ValueError('Number of inputs must be greater than zero.')
    if inputs % outputs != 0:
        raise ValueError(
            'Inputs must be divisible by outputs without a remainder.'
        )

    connectivity_matrix = np.zeros((inputs, outputs), dtype = int)
    in_per_out          = inputs // outputs
    for j in range(outputs):
        start_index = in_per_out * j
        end_index   = start_index + in_per_out
        connectivity_matrix[start_index:end_index, j] = 1

    return connectivity_matrix.T


def allocate_synapses(
    nb             : List[int],
    matrix         : np.ndarray,
    num_of_synapses: int,
    num_channels   : int = 1,
    rng            : Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    M, N = matrix.shape
    mask = np.zeros((M, N))

    syn_indices = nb_vals(matrix, list(nb))

    if len(syn_indices) < num_of_synapses:
        current_radius = 2
        while len(syn_indices) < num_of_synapses:
            extra_syns = nb_vals(
                matrix, list(nb), size = current_radius, perimeter = True
            )
            if len(extra_syns) == 0:
                break
            diff = num_of_synapses - len(syn_indices)
            if len(extra_syns) > diff:
                chosen_idx  = rng.choice(
                    len(extra_syns), size = diff, replace = False
                )
                syn_indices = np.concatenate(
                    (syn_indices, extra_syns[chosen_idx])
                )
            else:
                syn_indices = np.concatenate((syn_indices, extra_syns))
            current_radius += 1

    elif len(syn_indices) > num_of_synapses:
        idx         = rng.choice(
            len(syn_indices), size = num_of_synapses, replace = False
        )
        syn_indices = syn_indices[idx]

    if len(syn_indices) != num_of_synapses:
        raise ValueError(
            f'Could not find {num_of_synapses} pixels. '
            f'Image might be too small!'
        )

    row_indices = syn_indices[:, 0]
    col_indices = syn_indices[:, 1]
    mask[row_indices, col_indices] = 1

    if num_channels > 1:
        mask = np.expand_dims(mask, axis = 2)
        mask = np.tile(mask, (1, 1, num_channels))

    return mask.reshape(M * N * num_channels)


def make_mask_matrix(
    centers_ids    : List[Tuple[int, int]],
    matrix         : np.ndarray,
    dendrites      : int,
    somata         : int,
    num_of_synapses: int,
    num_channels   : int = 1,
    rfs_type       : str = 'somatic',
    rng            : Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    M, N       = matrix.shape
    mask_final = np.zeros((dendrites * somata, matrix.size * num_channels))
    counter    = 0

    if rfs_type == 'somatic':
        for center in centers_ids:
            nb_indices = nb_vals(matrix, list(center), size = 1)

            if len(nb_indices) < dendrites:
                current_radius = 2
                while len(nb_indices) < dendrites:
                    extra_centers = nb_vals(
                        matrix, list(center),
                        size      = current_radius,
                        perimeter = True,
                    )
                    if len(extra_centers) == 0:
                        break
                    diff = dendrites - len(nb_indices)
                    if len(extra_centers) > diff:
                        chosen_idx = rng.choice(
                            len(extra_centers), size = diff, replace = False
                        )
                        nb_indices = np.concatenate(
                            (nb_indices, extra_centers[chosen_idx])
                        )
                    else:
                        nb_indices = np.concatenate(
                            (nb_indices, extra_centers)
                        )
                    current_radius += 1

            if len(nb_indices) > dendrites:
                chosen_idx = rng.choice(
                    len(nb_indices), size = dendrites, replace = False
                )
                nb_indices = nb_indices[chosen_idx]

            for nb in nb_indices:
                mask_final[counter, :] = allocate_synapses(
                    nb, matrix, num_of_synapses,
                    num_channels = num_channels, rng = rng,
                )
                counter += 1

    elif rfs_type == 'dendritic':
        for center in centers_ids:
            mask_final[counter, :] = allocate_synapses(
                center, matrix, num_of_synapses,
                num_channels = num_channels, rng = rng,
            )
            counter += 1

    return mask_final


def receptive_fields(
    matrix         : np.ndarray,
    somata         : int,
    dendrites      : int,
    num_of_synapses: int,
    opt            : str = 'random',
    rfs_type       : str = 'somatic',
    step           : Optional[int]   = None,
    prob           : Optional[float] = None,
    num_channels   : int             = 1,
    num_rfs        : Optional[int]   = None,
    centers_ids    : Optional[List[Tuple[int, int]]] = None,
    rng            : Optional[np.random.Generator]   = None,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    if rng is None:
        rng = np.random.default_rng()
    M, N = matrix.shape

    if rfs_type == 'somatic':
        nodes = somata
    elif rfs_type == 'dendritic':
        nodes = dendrites * somata

    if not centers_ids:
        if opt == 'random':
            flat_indices = rng.choice(M * N, size = nodes, replace = True)
            centers_w, centers_h = np.unravel_index(flat_indices, (M, N))
            centers_ids = list(zip(centers_w, centers_h))

        elif opt == 'random_limited':
            if num_rfs is None:
                raise ValueError(
                    '`num_rfs` should be a positive integer under '
                    '`random_limited`. Found `None`'
                )
            limited_w = rng.integers(0, M, size = num_rfs)
            limited_h = rng.integers(0, N, size = num_rfs)
            chosen_indices = rng.choice(num_rfs, size = nodes, replace = True)
            centers_w   = limited_w[chosen_indices]
            centers_h   = limited_h[chosen_indices]
            centers_ids = list(zip(centers_w, centers_h))

        elif opt == 'semirandom':
            if prob is None:
                raise ValueError(
                    '`prob` should be a float in [0,1] under `semirandom`. '
                    'Found `None`'
                )
            p       = rng.random(nodes)
            somata1 = int(np.sum(p > prob))
            somata2 = int(np.sum(p < prob))

            w1, w2 = M // 4, 3 * M // 4
            h1, h2 = N // 4, 3 * N // 4

            centers_ids2 = []
            if somata2 > 0:
                in_w = rng.integers(w1, w2, size = somata2)
                in_h = rng.integers(h1, h2, size = somata2)
                centers_ids2 = list(zip(in_w, in_h))

            centers_ids1 = []
            if somata1 > 0:
                w_coords, h_coords = np.meshgrid(
                    np.arange(M), np.arange(N), indexing = 'ij'
                )
                center_mask    = (
                    (w_coords >= w1) & (w_coords < w2)
                    & (h_coords >= h1) & (h_coords < h2)
                )
                valid_periphery      = np.argwhere(~center_mask)
                chosen_periphery_idx = rng.choice(
                    len(valid_periphery), size = somata1, replace = True
                )
                out_coords   = valid_periphery[chosen_periphery_idx]
                centers_ids1 = list(
                    zip(out_coords[:, 0], out_coords[:, 1])
                )

            centers_ids = centers_ids1 + centers_ids2

        elif opt == 'serial':
            if step is None:
                raise ValueError(
                    '`step` should be a positive integer under `serial`. '
                    'Found `None`'
                )
            xv, yv = np.meshgrid(
                range(M), range(N), sparse = False, indexing = 'ij'
            )
            centers_ids = list(
                zip(xv.flatten()[::step], yv.flatten()[::step])
            )

    mask_final = make_mask_matrix(
        centers_ids, matrix, dendrites, somata,
        num_of_synapses, num_channels, rfs_type, rng,
    )

    return (mask_final.astype('int'), centers_ids)


#
"""
Data
"""
torchvision_sets = {
    'mnist' : torchvision.datasets.MNIST,
    'fmnist': torchvision.datasets.FashionMNIST,
}

datasets_dir = pathlib.Path(__file__).resolve().parent / 'DATASETS'


def as_numpy_arrays(
    dataset: torchvision.datasets.VisionDataset,
) -> Tuple[np.ndarray, np.ndarray]:
    images = dataset.data
    labels = dataset.targets
    if isinstance(images, torch.Tensor):
        images = images.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    return np.asarray(images), np.asarray(labels).squeeze()


def perturb_array(
    arr         : np.ndarray,
    perturbation: np.ndarray,
    amin        : float = 0,
    amax        : float = 1,
) -> np.ndarray:
    return np.clip(arr + perturbation, amin, amax)


def check_common_member(a: Any, b: Any) -> bool:
    return len(set(a).intersection(set(b))) > 0


def sequential_preprocess(
    input_train     : np.ndarray,
    target_train    : np.ndarray,
    batch_size      : int,
    validation_split: float,
    rng             : Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    target_train = target_train.squeeze()
    a, b = np.unique(target_train, return_counts = True)

    val_size = int(
        validation_split * input_train.shape[0] / (batch_size * len(a))
    )

    k1     = b // batch_size
    ktrain = (k1 - val_size) * batch_size
    kval   = b - ktrain

    val_set = []
    for i in range(len(a)):
        idx = np.argwhere(target_train == i).squeeze()
        val_set += list(rng.choice(idx, size = kval[i], replace = False))

    train_set = list(
        set(list(range(target_train.shape[0]))) - set(val_set)
    )

    if check_common_member(train_set, val_set):
        raise ValueError('Error in indices.')

    x_val   = input_train[val_set]
    y_val   = target_train[val_set]
    x_train = input_train[train_set]
    y_train = target_train[train_set]

    idx     = np.argsort(y_train)
    x_train = x_train[idx]
    y_train = y_train[idx]

    return {
        'xtrain': x_train,
        'ytrain': y_train,
        'xval'  : x_val,
        'yval'  : y_val,
    }


def get_data(
    validation_split: float,
    dtype           : str             = 'mnist',
    normalize       : bool            = True,
    add_noise       : bool            = False,
    sigma           : Optional[float] = None,
    sequential      : bool            = False,
    batch_size      : Optional[int]   = None,
    seed            : Optional[int]   = None,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], int, int, int]:
    rng = np.random.default_rng(seed)

    if dtype not in torchvision_sets:
        raise ValueError(
            f'Unknown dataset {dtype}, expected one of '
            f'{sorted(torchvision_sets)}.'
        )

    dataset_cls = torchvision_sets[dtype]
    x_train, y_train = as_numpy_arrays(
        dataset_cls(root = datasets_dir, train = True,  download = True)
    )
    x_test, y_test   = as_numpy_arrays(
        dataset_cls(root = datasets_dir, train = False, download = True)
    )

    if len(x_train.shape) == 3:
        img_height, img_width = x_train.shape[1:]
        channels = 1
    elif len(x_train.shape) > 3:
        img_height, img_width, channels = x_train.shape[1:]

    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype('float32') / 255.

    x_train = np.reshape(x_train, (-1, channels * img_width * img_height))
    x_test  = np.reshape(x_test,  (-1, channels * img_width * img_height))

    if sequential:
        dataset = sequential_preprocess(
            x_train, y_train,
            batch_size       = batch_size,
            validation_split = validation_split,
            rng              = rng,
        )
        x_train = dataset['xtrain']
        y_train = dataset['ytrain']
        x_val   = dataset['xval']
        y_val   = dataset['yval']
    else:
        indices = np.arange(x_train.shape[0])
        rng.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        valsize = int(validation_split * x_train.shape[0])
        x_val   = x_train[-valsize:]
        y_val   = y_train[-valsize:]
        x_train = x_train[:-valsize]
        y_train = y_train[:-valsize]

    data   = {'train': x_train, 'val': x_val, 'test': x_test}
    labels = {'train': y_train, 'val': y_val, 'test': y_test}

    if add_noise:
        for key in data.keys():
            perturbation = rng.normal(
                loc   = 0.0,
                scale = sigma,
                size  = data[key].shape,
            )
            data[key] = perturb_array(data[key], perturbation)

    return data, labels, img_height, img_width, channels


def make_loader(
    x         : np.ndarray,
    y         : np.ndarray,
    batch_size: int,
    shuffle   : bool,
    generator : Optional[torch.Generator] = None,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(np.asarray(x), dtype = torch.float32),
        torch.as_tensor(np.asarray(y), dtype = torch.long),
    )
    return DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle    = shuffle,
        generator  = generator,
    )


def count_correct(logits: Tensor, targets: Tensor) -> int:
    return int((logits.argmax(dim = 1) == targets).sum().item())


def evaluate(
    model  : nn.Module,
    loss_fn: nn.Module,
    loader : DataLoader,
    device : str,
) -> Tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct      = 0
    seen         = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            logits  = model(x_batch)

            running_loss += loss_fn(logits, y_batch).item()
            correct      += count_correct(logits, y_batch)
            seen         += y_batch.shape[0]

    return running_loss / len(loader), correct / seen


#
"""
Masks
"""
@dataclass(frozen = True)
class ModelSetup:
    name        : str
    rfs         : bool          = False
    rfs_type    : str           = 'somatic'
    input_sample: Optional[str] = None

model_setups = {
    0 : ModelSetup('dend_ann_random'),
    1 : ModelSetup('dend_ann_global_rfs', rfs = True),
    2 : ModelSetup('dend_ann_local_rfs',  rfs = True, rfs_type = 'dendritic'),
    10: ModelSetup('dend_ann_all_to_all', input_sample = 'all_to_all'),
}


def make_masks(
    dends       : List[int],
    soma        : List[int],
    synapses    : int,
    num_layers  : int,
    img_width   : int,
    img_height  : int,
    num_classes : int  = 10,
    channels    : int  = 1,
    rfs         : bool = True,
    rfs_type    : str  = 'somatic',
    rfs_mode    : str  = 'random',
    input_sample: Optional[str] = None,
    seed        : Optional[int] = None,
) -> List[np.ndarray]:
    rng   = np.random.default_rng(seed)
    masks = []

    for i in range(num_layers):
        if i == 0:
            matrix = np.zeros((img_width, img_height))
        else:
            divisors = [
                j for j in range(1, soma[i - 1] + 1) if soma[i - 1] % j == 0
            ]
            ix = len(divisors) // 2
            if len(divisors) % 2 == 0:
                matrix = np.zeros((divisors[ix], divisors[ix - 1]))
            else:
                matrix = np.zeros((divisors[ix], divisors[ix]))

        if rfs:
            mask_s_d, centers = receptive_fields(
                matrix,
                somata          = soma[i],
                dendrites       = dends[i],
                num_of_synapses = synapses,
                opt             = rfs_mode,
                rfs_type        = rfs_type,
                prob            = 0.7,
                num_channels    = channels if i == 0 else 1,
                rng             = rng,
            )
        else:
            inputs_size = matrix.size
            factor      = channels if i == 0 else 1
            mask_s_d = random_connectivity(
                inputs  = inputs_size * factor,
                outputs = soma[i] * dends[i],
                conns   = synapses * soma[i] * dends[i],
                rng     = rng,
            )
        masks.append(mask_s_d)
        masks.append(np.ones((mask_s_d.shape[0], )).astype('int'))

        mask_d_s = connectivity(
            inputs  = dends[i] * soma[i],
            outputs = soma[i],
        )
        masks.append(mask_d_s)
        masks.append(np.ones((mask_d_s.shape[0], )).astype('int'))

    if input_sample == 'all_to_all':
        for i, m in enumerate(masks):
            if i % 4 == 0:
                masks[i] = np.ones_like(m)

    masks.append(np.ones((num_classes, masks[-2].shape[0])).astype('int'))
    masks.append(np.ones((num_classes, )).astype('int'))

    return masks


def split_dendrite_masks(
    masks       : List[np.ndarray],
    dends       : List[int],
    soma        : List[int],
    num_layers  : int,
    direct_input: str = 'none',
) -> Tuple[List[Tensor], List[List[Tensor]]]:
    if direct_input not in ('none', 'union'):
        raise ValueError(
            f"direct_input must be none or union, got {direct_input!r}."
        )

    soma_masks : List[Tensor]       = []
    slot_masks : List[List[Tensor]] = []

    for i in range(num_layers):
        dend_mask = np.asarray(masks[4 * i])
        expected  = dends[i] * soma[i]
        if dend_mask.shape[0] != expected:
            raise ValueError(
                f'Layer {i} dendrite mask has {dend_mask.shape[0]} rows, '
                f'expected {expected}.'
            )

        layer_slots = [
            torch.as_tensor(
                dend_mask[k::dends[i], :].copy(),
                dtype = torch.float32,
            )
            for k in range(dends[i])
        ]
        slot_masks.append(layer_slots)

        if direct_input == 'union':
            union = np.stack([
                dend_mask[j * dends[i]:(j + 1) * dends[i], :].max(axis = 0)
                for j in range(soma[i])
            ])
            soma_masks.append(
                torch.as_tensor(union, dtype = torch.float32)
            )
        else:
            soma_masks.append(
                torch.zeros(soma[i], dend_mask.shape[1])
            )

    return soma_masks, slot_masks


def get_perforated_model(
    input_shape: Tuple[int, ...],
    num_layers : int,
    soma       : List[int],
    soma_masks : List[Tensor],
    num_classes: int,
    fname_model: str,
    relu_slope : float = 0.1,
    dropout    : bool  = False,
    rate       : float = 0.0,
) -> PerforatedDendriticANN:
    return PerforatedDendriticANN(
        input_size  = input_shape[0],
        num_layers  = num_layers,
        soma        = soma,
        soma_masks  = soma_masks,
        num_classes = num_classes,
        name        = fname_model,
        relu_slope  = relu_slope,
        dropout     = dropout,
        rate        = rate,
    )


#
"""
PAI Utilities
"""
def configure_pai(
    model              : PerforatedDendriticANN,
    num_dends          : int,
    epochs_per_dendrite: int,
    save_name          : str,
    relu_slope         : float = 0.1,
    test_capacity      : bool  = False,
) -> None:
    GPA.pc.set_module_names_to_perforate([])
    GPA.pc.set_module_ids_to_perforate(
        [f'.{name}' for name in model.soma_names]
    )
    GPA.pc.set_module_ids_to_track(['.output'])
    GPA.pc.set_output_dimensions([-1, 0])

    GPA.pc.set_pai_forward_function(
        nn.LeakyReLU(negative_slope = relu_slope)
    )

    GPA.pc.set_max_dendrites(num_dends)
    GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
    GPA.pc.set_first_fixed_switch_num(1)
    GPA.pc.set_fixed_switch_num(epochs_per_dendrite)

    GPA.pc.set_retain_all_dendrites(True)
    GPA.pc.set_max_dendrite_tries(max(num_dends * 4, 1000))

    GPA.pc.set_save_name(save_name)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_testing_dendrite_capacity(test_capacity)


def dendrite_report(model: PerforatedDendriticANN) -> str:
    parts = []
    for name, soma in zip(model.soma_names, model.soma_modules()):
        dendrites = getattr(soma, 'dendrite_module', None)
        grown     = 0 if dendrites is None else int(dendrites.num_dendrites)
        parts.append(f'{name}: {grown}')
    return ', '.join(parts)


def trainable_parameter_count(model: nn.Module) -> int:
    scaffolding = (
        'parent_module',
        'candidate_module',
        'best_candidate_module',
        'dendrites_to_candidates',
    )

    excluded = set()
    for module in model.modules():
        for name in scaffolding:
            holder = getattr(module, name, None)
            if holder is None:
                continue
            for param in holder.parameters():
                excluded.add(id(param))

        to_top = getattr(module, 'dendrites_to_top', None)
        if to_top is not None and len(to_top) > 0:
            for entry in list(to_top)[:-1]:
                excluded.add(id(entry))

    masked = {}
    for module in model.modules():
        rf = getattr(module, 'rf', None)
        if rf is None or not isinstance(module.weight, nn.Parameter):
            continue
        masked[id(module.weight)] = int(rf.sum().item())

    total = 0
    for param in model.parameters():
        if id(param) in excluded or not param.requires_grad:
            continue
        total += masked.get(id(param), param.numel())

    return total


#
"""
Config
"""
# Positional CLI arguments, in the order the shell scripts pass them. The
# first fifteen match main.py and main_pai.py exactly
gpu_id       = int(sys.argv[1])     # cuda device index
seq_flag     = int(sys.argv[2])     # 1 to present classes sequentially
estop_flag   = int(sys.argv[3])     # 1 to enable early stopping
trial        = int(sys.argv[4])     # seeds every generator the run touches
model_type   = int(sys.argv[5])     # row of model_setups to build
sigma        = float(sys.argv[6])   # standard deviation of input noise
datatype     = sys.argv[7]          # mnist or fmnist
num_dends    = int(sys.argv[8])     # dendrites per soma, PAI slots
num_soma     = int(sys.argv[9])     # somata per layer
num_layers   = int(sys.argv[10])    # somatic layers
synapses     = int(sys.argv[11])    # inputs sampled per dendrite
drop_flag    = int(sys.argv[12])    # 1 to add dropout after each activation
rate_of_drop = float(sys.argv[13])  # dropout probability
lr           = float(sys.argv[14])  # adam learning rate
output_dir   = sys.argv[15]         # directory the run writes into

# PAI arguments, appended so the existing sweeps still line up
argc          = len(sys.argv)
switch_epochs = int(sys.argv[16]) if argc > 16 else 5
test_capacity = bool(int(sys.argv[17])) if argc > 17 else False
direct_input  = sys.argv[18] if argc > 18 else 'none'
converge_eps  = int(sys.argv[19]) if argc > 19 else 0

# Run settings that no shell script varies
save             = True
noise            = True
batch_size       = 128
validation_split = 0.1

epoch_counts = {
    'mnist' : (15, 30),
    'fmnist': (25, 50),
}

epoch_slack = 4


def seed_everything(seed: int) -> dict:
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark    = False
    torch.backends.cudnn.deterministic = True

    try:
        torch.use_deterministic_algorithms(True)
        deterministic_algorithms = 1
    except Exception:
        deterministic_algorithms = 0

    return {
        'seed_python'             : seed,
        'seed_numpy'              : seed,
        'seed_torch'              : seed,
        'seed_cuda'               : seed,
        'cudnn_benchmark'         : 0,
        'cudnn_deterministic'     : 1,
        'deterministic_algorithms': deterministic_algorithms,
    }


def csv_value(value):
    if value is None:
        return ''
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys = True)
    if isinstance(value, bool):
        return '1' if value else '0'
    return str(value)


def csv_row(values):
    return {key: csv_value(value) for key, value in values.items()}


#
"""
Setup
"""
device = torch.device(
    f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
)
sequential = bool(seq_flag)
early_stop = bool(estop_flag)
dropout    = bool(drop_flag)

seed_info = seed_everything(trial)

setup       = model_setups[model_type]
fname_model = f'pai_{setup.name}'

file_tag = '_sequential' if sequential else ''
if dropout:
    fname_model += f'_dropout_{rate_of_drop}'
if lr != 0.001:
    file_tag += f'_lr_{lr}'

base_epochs = epoch_counts[datatype][1 if sequential else 0]
max_epochs  = base_epochs + (num_dends + epoch_slack) * switch_epochs * 2

sub_tag     = f'results_{datatype}_{num_layers}_layer{file_tag}'
dirname     = pathlib.Path(output_dir).resolve() / sub_tag
outdir_name = dirname / fname_model
postfix     = (
    f'sigma_{sigma}_trial_{trial}_dends_{num_dends}_soma_{num_soma}'
)

variant_postfix = f'variant_{postfix}'

outdir_name.mkdir(parents = True, exist_ok = True)

pai_save_name = f'pai_{variant_postfix}'


#
"""
Data
"""
data, labels, img_height, img_width, channels = get_data(
    validation_split = validation_split,
    dtype            = datatype,
    normalize        = True,
    add_noise        = noise,
    sigma            = sigma,
    sequential       = sequential,
    batch_size       = batch_size,
    seed             = trial,
)

x_train, x_val, x_test = data['train'], data['val'], data['test']
y_train, y_val, y_test = labels['train'], labels['val'], labels['test']

num_classes = len(set(y_train))
dends       = num_layers * [num_dends]
soma        = num_layers * [num_soma]
input_shape = (img_width * img_height * channels, )

train_loader = make_loader(
    x_train,
    y_train,
    batch_size,
    shuffle   = not sequential,
    generator = torch.Generator().manual_seed(trial),
)
val_loader  = make_loader(x_val, y_val, batch_size, shuffle = False)
test_loader = make_loader(x_test, y_test, batch_size, shuffle = False)


#
"""
Model
"""
masks = make_masks(
    dends,
    soma,
    synapses,
    num_layers,
    img_width,
    img_height,
    num_classes,
    channels,
    rfs          = setup.rfs,
    rfs_type     = setup.rfs_type,
    rfs_mode     = 'random',
    input_sample = setup.input_sample,
    seed         = trial,
)

soma_masks, slot_masks = split_dendrite_masks(
    masks,
    dends,
    soma,
    num_layers,
    direct_input = direct_input,
)

model = get_perforated_model(
    input_shape,
    num_layers,
    soma,
    soma_masks,
    num_classes,
    fname_model = fname_model,
    dropout     = dropout,
    rate        = rate_of_drop,
)

os.chdir(outdir_name)

configure_pai(
    model,
    num_dends,
    switch_epochs,
    save_name     = pai_save_name,
    test_capacity = test_capacity,
)

model = UPA.perforate_model(
    model,
    save_name        = pai_save_name,
    maximizing_score = True,
)
model.to(device)

# Register the factory and wire in rf-mask pinning via the variant framework
initialize_variant_dendrite(
    synapses  = synapses,
    rf_mode   = 'random',  # all_to_all | random | somatic | dendritic
    img_shape = (img_width, img_height, channels),
)

GPA.pai_tracker.set_optimizer(torch.optim.Adam)
optimizer, _ = GPA.pai_tracker.setup_optimizer(
    model,
    {'params': model.parameters(), 'lr': lr, 'eps': 1e-7},
    {},
)
loss_fn = torch.nn.CrossEntropyLoss()


#
"""
Train
"""
print(f'\nModel: {fname_model}, trial: {trial}, layers: {num_layers}, '
      f'noise: {sigma}, dataset: {datatype}, tag: {file_tag}')
print(f'Growing up to {num_dends} dendrites per soma, switching every '
      f'{switch_epochs} epochs, capped at {max_epochs} epochs')
print(f'Converging for {converge_eps} epochs after the last dendrite '
      f'integrates\n')
print('Variant: poirazi_receptive_field_dendrites\n')

train_loss_list, train_acc_list = [], []
val_loss_list, val_acc_list     = [], []
dendrite_counts, switch_epoch   = [], []

start_time       = time.time()
epoch            = -1
growth_end       = -1
converging       = False
best_val_acc     = -float('inf')
best_val_epoch   = -1
best_val_weights = None

while True:
    epoch += 1
    budget = growth_end + 1 + converge_eps if converging else max_epochs
    if epoch >= budget:
        print(f'\nEpoch cap of {budget} reached, stopping.')
        break

    print(f'\nepoch {epoch + 1}/{budget}')
    epoch_start = time.time()

    model.train()
    running_train_loss = 0.0
    train_correct      = 0
    train_seen         = 0
    for x_batch_train, y_batch_train in train_loader:
        x_batch_train = x_batch_train.to(device)
        y_batch_train = y_batch_train.to(device)

        train_logits = model(x_batch_train)
        train_loss   = loss_fn(train_logits, y_batch_train)

        optimizer.zero_grad(set_to_none = True)
        train_loss.backward()
        optimizer.step()

        running_train_loss += train_loss.item()
        train_correct      += count_correct(train_logits, y_batch_train)
        train_seen         += y_batch_train.shape[0]

    train_acc_list.append(train_correct / train_seen)
    train_loss_list.append(running_train_loss / len(train_loader))

    val_loss, val_acc = evaluate(model, loss_fn, val_loader, device)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)

    print(f'train_loss: {train_loss_list[-1]:.4f} - '
          f'val_loss: {val_loss_list[-1]:.4f}')
    print(f'Training acc over epoch: {train_acc_list[-1]:.4f}, '
          f'Validation acc over epoch: {val_acc_list[-1]:.4f}')
    print(f'Time taken for epoch {epoch}: '
          f'{time.time() - epoch_start:.2f}s')

    if converging:
        dendrite_counts.append(dendrite_report(model))
        if val_acc_list[-1] > best_val_acc:
            best_val_acc     = val_acc_list[-1]
            best_val_epoch   = epoch
            best_val_weights = copy.deepcopy(model.state_dict())
        continue

    GPA.pai_tracker.add_extra_score(train_acc_list[-1], 'train')
    model, restructured, training_complete = (
        GPA.pai_tracker.add_validation_score(val_acc_list[-1], model)
    )
    model.to(device)

    dendrite_counts.append(dendrite_report(model))

    if training_complete:
        print('\nPAI reports training complete.')
        growth_end = epoch

        if converge_eps <= 0:
            break

        converging = True
        optimizer  = torch.optim.Adam(model.parameters(), lr = lr, eps = 1e-7)
        print(f'Converging the grown network for {converge_eps} epochs.\n')
        continue

    if restructured:
        switch_epoch.append(epoch)
        print(f'Restructured, dendrites now {dendrite_counts[-1]}')
        optimizer, _ = GPA.pai_tracker.setup_optimizer(
            model,
            {'params': model.parameters(), 'lr': lr},
            {},
        )

test_loss, test_acc = evaluate(model, loss_fn, test_loader, device)

out = {
    'train_loss'      : train_loss_list,
    'train_acc'       : train_acc_list,
    'val_loss'        : val_loss_list,
    'val_acc'         : val_acc_list,
    'test_loss'       : test_loss,
    'test_acc'        : test_acc,
    'time'            : time.time() - start_time,
    'switch_epochs'   : switch_epoch,
    'dendrite_counts' : dendrite_counts,
    'trainable_params': trainable_parameter_count(model),
    'growth_end'      : growth_end,
    'converge_epochs' : converge_eps,
    'Masks'           : masks,
}

print(f'\ntest_loss: {test_loss:.4f} - test_acc: {test_acc:.4f}')

if best_val_weights is not None:
    out['test_acc_final']  = test_acc
    out['test_loss_final'] = test_loss

    model.load_state_dict(best_val_weights)
    best_loss, best_acc = evaluate(model, loss_fn, test_loader, device)

    out['test_acc_best']  = best_acc
    out['test_loss_best'] = best_loss
    out['best_val_acc']   = best_val_acc
    out['best_epoch']     = best_val_epoch

    print(f'Best val epoch {best_val_epoch + 1} '
          f'(val_acc {best_val_acc:.4f}) | test acc {best_acc:.4f}')

print(f'Trainable parameters: {out["trainable_params"]}')

run_started_at = (
    datetime.now(timezone.utc).isoformat(timespec = 'seconds')
    .replace('+00:00', 'Z')
)
history_csv = dirname / 'run_history.csv'
run_summary = {
    'run_date_utc'        : run_started_at,
    'script'              : 'variant_poirazi_rf',
    'trial'               : trial,
    **seed_info,
    'gpu_id'              : gpu_id,
    'seq_flag'            : seq_flag,
    'estop_flag'          : estop_flag,
    'model_type'          : model_type,
    'model_name'          : fname_model,
    'dataset'             : datatype,
    'sigma'               : sigma,
    'num_dends'           : num_dends,
    'num_soma'            : num_soma,
    'num_layers'          : num_layers,
    'synapses'            : synapses,
    'drop_flag'           : drop_flag,
    'rate_of_drop'        : rate_of_drop,
    'lr'                  : lr,
    'switch_epochs_target': switch_epochs,
    'test_capacity'       : int(test_capacity),
    'direct_input'        : direct_input,
    'converge_eps'        : converge_eps,
    'train_loss'          : train_loss_list,
    'train_acc'           : train_acc_list,
    'val_loss'            : val_loss_list,
    'val_acc'             : val_acc_list,
    'switch_epochs_observed': switch_epoch,
    'dendrite_counts'     : dendrite_counts,
    'growth_end'          : growth_end,
    'best_epoch'          : out.get('best_epoch', ''),
    'best_val_acc'        : out.get('best_val_acc', ''),
    'test_loss'           : out['test_loss'],
    'test_acc'            : out['test_acc'],
    'test_loss_final'     : out.get('test_loss_final', ''),
    'test_acc_final'      : out.get('test_acc_final', ''),
    'test_loss_best'      : out.get('test_loss_best', ''),
    'test_acc_best'       : out.get('test_acc_best', ''),
    'time_seconds'        : out['time'],
    'trainable_params'    : out['trainable_params'],
    'result_pickle'       : str(pathlib.Path(
        f'{outdir_name}/results_{variant_postfix}.pkl'
    )),
    'checkpoint_pt'       : str(pathlib.Path(
        f'{outdir_name}/model_{variant_postfix}.pt'
    )),
}
run_summary_csv = csv_row(run_summary)

# Load prior rows and compare against the earliest (the main_pai.py run).
# 'script' is excluded from comparison so only result metrics are checked
prior_rows = []
if history_csv.exists():
    with open(history_csv, newline = '') as handle:
        prior_rows = list(csv.DictReader(handle))

volatile_keys = {
    'run_date_utc',
    'time_seconds',
    'result_pickle',
    'checkpoint_pt',
    'script',
}
comparison_keys = [
    key for key in run_summary_csv
    if key not in volatile_keys
]
if prior_rows:
    prior_rows.sort(key = lambda row: row.get('run_date_utc', ''))
    earliest_row = prior_rows[0]
    shared_keys  = [
        key for key in comparison_keys
        if key in earliest_row
    ]
    identical = all(
        run_summary_csv.get(key, '') == earliest_row.get(key, '')
        for key in shared_keys
    )
    earliest_script = earliest_row.get('script', 'main_pai')
    print(
        f'\nVariant results identical to earliest recorded run '
        f'({earliest_script}): {identical}'
    )
else:
    print(
        '\nNo earlier CSV run found for comparison. '
        'Run main_pai.py first with the same arguments, '
        'pointing PAI_OUT at the same output directory.'
    )


#
"""
Save
"""
if save:
    checkpoint = {
        'state_dict': model.state_dict(),
        'config'    : {
            'input_size'  : input_shape[0],
            'num_layers'  : num_layers,
            'soma'        : soma,
            'num_classes' : num_classes,
            'name'        : fname_model,
            'dropout'     : dropout,
            'rate'        : rate_of_drop,
            'dends'       : dends,
            'direct_input': direct_input,
        },
    }
    torch.save(
        checkpoint,
        pathlib.Path(f'{outdir_name}/model_{variant_postfix}.pt'),
    )

    fname_res = pathlib.Path(
        f'{outdir_name}/results_{variant_postfix}.pkl'
    )
    with open(fname_res, 'wb') as handle:
        pickle.dump(out, handle, protocol = pickle.HIGHEST_PROTOCOL)

    write_header = not history_csv.exists()
    with open(history_csv, 'a', newline = '') as handle:
        writer = csv.DictWriter(
            handle, fieldnames = list(run_summary_csv.keys())
        )
        if write_header:
            writer.writeheader()
        writer.writerow(run_summary_csv)

    print(f'\nResults have been saved in: {dirname}/{fname_model}')
