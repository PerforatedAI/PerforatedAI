################################################################################
# Perforated dendritic ANN on MNIST / FashionMNIST.                           #
#                                                                              #
# Registers a receptive-field dendrite factory via the PAI variant framework. #
# Somas contribute no direct input signal; all signal flows through dendrites  #
# grown by PAI.                                                                #
#                                                                              #
# Run from dendrite_variants/:                                                 #
#   CUDA_VISIBLE_DEVICES=0 python mnist_perforated_variant_test.py \          #
#     GPU _ _ TRIAL MODEL_TYPE SIGMA DATASET \                                 #
#     NUM_DENDS NUM_SOMA NUM_LAYERS SYNAPSES \                                 #
#     DROP_FLAG DROP_RATE LR PAI_OUT \                                        #
#     [SWITCH_EPOCHS] [TEST_CAPACITY] [_] [CONVERGE_EPS]                      #
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
import torch
import pickle
import pathlib
import numpy as np
import torchvision

from torch            import nn
from collections      import OrderedDict
from torch.utils.data import DataLoader, TensorDataset
from typing           import Any, Dict, List, Optional, Tuple

# Only this script's directory needs to be on the path, for the local
# receptive_field_dendrites package
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai   as UPA

from receptive_field_dendrites import initialize_variant_dendrite
from clean_somas import CleanSomas


#
"""
Data
"""
datasets_dir = pathlib.Path(__file__).resolve().parent / 'DATASETS'

_dataset_cls = {
    'mnist' : torchvision.datasets.MNIST,
    'fmnist': torchvision.datasets.FashionMNIST,
}


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


def load_data(
    dtype           : str,
    sigma           : float,
    seed            : int,
    validation_split: float = 0.1,
    sequential      : bool  = False,
    batch_size      : int   = 128,
) -> Tuple[np.ndarray, ...]:
    if dtype not in _dataset_cls:
        raise ValueError(f'Unknown dataset {dtype!r}, expected one of {sorted(_dataset_cls)}.')

    rng = np.random.default_rng(seed)

    raw_train = _dataset_cls[dtype](root = datasets_dir, train = True,  download = True)
    raw_test  = _dataset_cls[dtype](root = datasets_dir, train = False, download = True)

    x_train = raw_train.data.numpy().astype('float32') / 255.
    y_train = raw_train.targets.numpy()
    x_test  = raw_test.data.numpy().astype('float32') / 255.
    y_test  = raw_test.targets.numpy()

    if x_train.ndim == 3:
        _, H, W = x_train.shape
        C = 1
    else:
        _, H, W, C = x_train.shape

    x_train = x_train.reshape(-1, H * W * C)
    x_test  = x_test.reshape(-1,  H * W * C)

    if sequential:
        splits  = sequential_preprocess(
            input_train      = x_train,
            target_train     = y_train,
            batch_size       = batch_size,
            validation_split = validation_split,
            rng              = rng,
        )
        x_train = splits['xtrain']
        y_train = splits['ytrain']
        x_val   = splits['xval']
        y_val   = splits['yval']
    else:
        idx     = np.arange(len(x_train))
        rng.shuffle(idx)
        x_train = x_train[idx]
        y_train = y_train[idx]
        val_n   = int(validation_split * len(x_train))
        x_val   = x_train[-val_n:].copy()
        y_val   = y_train[-val_n:].copy()
        x_train = x_train[:-val_n]
        y_train = y_train[:-val_n]

    def _add_noise(x):
        return np.clip(x + rng.normal(0.0, sigma, x.shape), 0.0, 1.0).astype('float32')

    x_train = _add_noise(x_train)
    x_val   = _add_noise(x_val)
    x_test  = _add_noise(x_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, H, W, C


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
            correct      += int((logits.argmax(dim = 1) == y_batch).sum().item())
            seen         += y_batch.shape[0]

    return running_loss / len(loader), correct / seen


#
"""
Model
"""
class PerforatedDendriticANN(nn.Module):
    '''
    Dendritic ANN whose dendrites come from PAI, not from a layer

    Notes:
        - Each soma module is provided externally (e.g. CleanSomas) and PAI
          grows dendrites on top of it:
            soma_j output + sum_k(dendrite_jk output) → LeakyReLU → next layer
        - Somas are registered as self.somas (ModuleList); PAI perforates
          them via module IDs .somas.0, .somas.1, …

    Signature:
        soma (List[int]):
            - Output width of each somatic layer
        soma_modules (List[nn.Module]):
            - Pre-built soma module for each layer
        num_classes (int):
            - Number of output classes
        name (str):
            - Model name used when building output paths
        relu_slope (float):
            - Negative slope of the leaky relu activations
        dropout (bool):
            - Whether a dropout op follows each activation
        rate (float):
            - Dropout probability, ignored when dropout is False
    '''
    def __init__(
        self,
        soma        : List[int],
        soma_modules: List[nn.Module],
        num_classes : int,
        name        : str,
        relu_slope  : float = 0.1,
        dropout     : bool  = False,
        rate        : float = 0.0,
    ) -> None:
        super().__init__()
        self.name = name

        layers = OrderedDict()
        for j, sm in enumerate(soma_modules):
            layers[f'soma_{j}'] = sm
            layers[f'relu_{j}'] = nn.LeakyReLU(negative_slope = relu_slope)
            if dropout:
                layers[f'drop_{j}'] = nn.Dropout(p = rate)
        self.net = nn.Sequential(layers)

        self.output = nn.Linear(soma[-1], num_classes)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(self.net(x))


#
"""
Model types
"""
model_names = {
    0 : 'dend_ann_random',
    1 : 'dend_ann_global_rfs',
    2 : 'dend_ann_local_rfs',
    10: 'dend_ann_all_to_all',
}

rf_modes = {
    0 : 'random',
    1 : 'somatic',
    2 : 'dendritic',
    10: 'all_to_all',
}


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
        [f'.net.{name}' for name, _ in model.net.named_children()
         if name.startswith('soma_')]
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
    for name, soma in model.net.named_children():
        if not name.startswith('soma_'):
            continue
        dendrites = getattr(soma, 'dendrite_module', None)
        grown     = 0 if dendrites is None else int(dendrites.num_dendrites)
        parts.append(f'soma.{name[5:]}: {grown}')
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
# Positional CLI arguments (argv[3] is an unused legacy slot)
gpu_id       = int(sys.argv[1])     # cuda device index
seq_flag     = int(sys.argv[2])     # 1 to use sequential (sorted) training
trial        = int(sys.argv[4])     # seeds every generator the run touches
model_type   = int(sys.argv[5])     # selects entry in model_names / rf_modes
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

# Optional PAI arguments (argv[18] is an unused legacy slot)
argc          = len(sys.argv)
switch_epochs = int(sys.argv[16]) if argc > 16 else 5
test_capacity = bool(int(sys.argv[17])) if argc > 17 else False
converge_eps  = int(sys.argv[19]) if argc > 19 else 0

# Run settings that no shell script varies
batch_size       = 128
validation_split = 0.1

_base, _base_seq = {'mnist': (15, 30), 'fmnist': (25, 50)}[datatype]
epoch_slack      = 4


#
"""
Setup
"""
device = torch.device(
    f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu'
)
dropout    = bool(drop_flag)
sequential = bool(seq_flag)

np.random.seed(trial)
torch.manual_seed(trial)

fname_model = f'pai_{model_names[model_type]}'

if dropout:
    fname_model += f'_dropout_{rate_of_drop}'

lr_tag     = f'_lr_{lr}' if lr != 0.001 else ''
base_epochs = _base_seq if sequential else _base
max_epochs  = base_epochs + (num_dends + epoch_slack) * switch_epochs * 2
dirname    = pathlib.Path(output_dir).resolve() / f'results_{datatype}_{num_layers}_layer{lr_tag}'
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
x_train, y_train, x_val, y_val, x_test, y_test, img_height, img_width, channels = load_data(
    dtype            = datatype,
    sigma            = sigma,
    seed             = trial,
    validation_split = validation_split,
    sequential       = sequential,
    batch_size       = batch_size,
)

num_classes = len(set(y_train))
soma        = num_layers * [num_soma]
input_shape = (img_width * img_height * channels, )

train_loader = DataLoader(
    TensorDataset(
        torch.as_tensor(np.asarray(x_train), dtype = torch.float32),
        torch.as_tensor(np.asarray(y_train), dtype = torch.long),
    ),
    batch_size = batch_size,
    shuffle    = not sequential,
)
val_loader = DataLoader(
    TensorDataset(
        torch.as_tensor(np.asarray(x_val), dtype = torch.float32),
        torch.as_tensor(np.asarray(y_val), dtype = torch.long),
    ),
    batch_size = batch_size,
    shuffle    = False,
)
test_loader = DataLoader(
    TensorDataset(
        torch.as_tensor(np.asarray(x_test), dtype = torch.float32),
        torch.as_tensor(np.asarray(y_test), dtype = torch.long),
    ),
    batch_size = batch_size,
    shuffle    = False,
)


#
"""
Model
"""
in_f         = input_shape[0]
soma_modules = []
for j in range(num_layers):
    soma_modules.append(CleanSomas(
        soma[j],
        config = {
            'in_features' : in_f,
            'out_features': soma[j],
        },
    ))
    in_f = soma[j]

model = PerforatedDendriticANN(
    soma         = soma,
    soma_modules = soma_modules,
    num_classes  = num_classes,
    name         = fname_model,
    dropout      = dropout,
    rate         = rate_of_drop,
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
    rf_mode   = rf_modes[model_type],
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
      f'noise: {sigma}, dataset: {datatype}')
print(f'Growing up to {num_dends} dendrites per soma, switching every '
      f'{switch_epochs} epochs, capped at {max_epochs} epochs')
print(f'Converging for {converge_eps} epochs after the last dendrite '
      f'integrates\n')
print('Variant: receptive_field_dendrites\n')

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
        train_correct      += int((train_logits.argmax(dim = 1) == y_batch_train).sum().item())
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


#
"""
Save
"""
checkpoint = {
    'state_dict': model.state_dict(),
    'config'    : {
        'input_size' : input_shape[0],
        'num_layers' : num_layers,
        'soma'       : soma,
        'num_classes': num_classes,
        'name'       : fname_model,
        'dropout'    : dropout,
        'rate'       : rate_of_drop,
    },
}
torch.save(
    checkpoint,
    pathlib.Path(f'{outdir_name}/model_{variant_postfix}.pt'),
)

fname_res = pathlib.Path(f'{outdir_name}/results_{variant_postfix}.pkl')
with open(fname_res, 'wb') as handle:
    pickle.dump(out, handle, protocol = pickle.HIGHEST_PROTOCOL)

print(f'\nResults have been saved in: {dirname}/{fname_model}')
