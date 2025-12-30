
"""
Original:

Epoch: [89]  [500/506]  eta: 0:00:00  lr: 0.0010000000000000002  img/s: 12316.37788483597  loss: 0.5689 (0.5906)  acc1: 83.2031 (83.4479)  acc5: 95.7031 (95.5636)  time: 0.0253  data: 0.0049  max mem: 3617
Epoch: [89] Total time: 0:00:13
Test:   [ 0/20]  eta: 0:00:18  loss: 0.8479 (0.8479)  acc1: 74.6094 (74.6094)  acc5: 95.3125 (95.3125)  time: 0.9173  data: 0.9109  max mem: 3617
Test:  Total time: 0:00:01
Test:  Acc@1 70.300 Acc@5 89.820
Training time 0:22:53

with PAI:

Epoch: [1125]  [500/506]  eta: 0:00:01  lr: 0.00010000000000000003  img/s: 1421.8356206467333  loss: 0.3493 (0.3511)  acc1: 90.2344 (90.1432)  acc5: 96.8750 (97.4715)  time: 0.1801  data: 0.0001  max mem: 9340
Epoch: [1125] Total time: 0:01:32
Adding extra score Train Acc 1 of 90.16036168026996
Adding extra score Train Acc 5 of 97.47517291776981
Test:   [ 0/20]  eta: 0:00:22  loss: 0.6772 (0.6772)  acc1: 85.1562 (85.1562)  acc5: 96.8750 (96.8750)  time: 1.1152  data: 1.0578  max mem: 9340
Test:  Total time: 0:00:02
Test:  Acc@1 73.040 Acc@5 91.420


18_thin
Test:  Acc@1 61.660 Acc@5 85.200

current sweep whrtmlct

"""



import datetime
import os
import time
import warnings
import argparse

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from sampler import RASampler
from torch import nn
from torch.utils.data.dataloader import default_collate
from torchvision.transforms.functional import InterpolationMode
from transforms import get_mixup_cutmix

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# Import custom ResNet models
import resnet as custom_resnet

import wandb
from types import SimpleNamespace


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema=None, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                # we should unscale the gradients of optimizer's assigned params if do gradient clipping
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                # Reset ema buffer to keep copying weights during warmup period
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    
    # Add training accuracies to PerforatedAI tracker
    GPA.pai_tracker.add_extra_score(metric_logger.acc1.global_avg, "Train Acc 1")
    GPA.pai_tracker.add_extra_score(metric_logger.acc5.global_avg, "Train Acc 5")


def evaluate(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            # FIXME need to take into account that the datasets
            # could have been padded in distributed setup
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size
    
    # gather the stats from all processes
    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        # See FIXME above
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    # Add validation score to PerforatedAI tracker and check for restructuring
    GPA.pai_tracker.add_extra_score(metric_logger.acc5.global_avg, "Val Acc 5")
    model, restructured, trainingComplete = GPA.pai_tracker.add_validation_score(metric_logger.acc1.global_avg, model)
    
    return model, metric_logger.acc1.global_avg, restructured, trainingComplete


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


# ImageNet-100 standard class indices (commonly used subset)
IMAGENET100_CLASSES = [
    'n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475',
    'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878',
    'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544',
    'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084',
    'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779',
    'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777',
    'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541',
    'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366',
    'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811',
    'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457',
    'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322',
    'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381',
    'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939',
    'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244',
    'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797',
    'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675',
    'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143',
    'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313',
    'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805',
    'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672'
]


def filter_imagenet100(dataset):
    """Filter dataset to only include ImageNet-100 classes."""
    # Get original class_to_idx mapping
    original_class_to_idx = dataset.class_to_idx
    
    # Create mapping from old indices to new indices
    valid_classes = [cls for cls in IMAGENET100_CLASSES if cls in original_class_to_idx]
    new_class_to_idx = {cls: new_idx for new_idx, cls in enumerate(valid_classes)}
    old_to_new_idx = {original_class_to_idx[cls]: new_idx for cls, new_idx in new_class_to_idx.items()}
    
    # Filter samples
    filtered_samples = []
    for path, old_idx in dataset.samples:
        if old_idx in old_to_new_idx:
            filtered_samples.append((path, old_to_new_idx[old_idx]))
    
    # Update dataset
    dataset.samples = filtered_samples
    dataset.targets = [s[1] for s in filtered_samples]
    dataset.classes = valid_classes
    dataset.class_to_idx = new_class_to_idx
    
    print(f"Filtered dataset to {len(valid_classes)} classes with {len(filtered_samples)} samples")
    return dataset


def create_optimizer_and_scheduler(model, args, custom_keys_weight_decay, epoch=None):
    """Create optimizer and scheduler for the model using PerforatedAI setup.
    
    Args:
        model: The model to create optimizer for
        args: Training arguments
        custom_keys_weight_decay: List of (key, weight_decay) tuples for custom weight decay
        epoch: Current epoch (used for warmup adjustment after restructuring), None for initial setup
    
    Returns:
        optimizer, lr_scheduler tuple
    """
    # Set up parameter groups with different weight decay
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    # Set optimizer class
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        GPA.pai_tracker.set_optimizer(torch.optim.SGD)
        optimArgs = {
            "params": parameters,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "nesterov": "nesterov" in opt_name,
        }
    elif opt_name == "rmsprop":
        GPA.pai_tracker.set_optimizer(torch.optim.RMSprop)
        optimArgs = {
            "params": parameters,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "eps": 0.0316,
            "alpha": 0.9,
        }
    elif opt_name == "adamw":
        GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
        optimArgs = {
            "params": parameters,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    # Set scheduler class and prepare scheduler args
    args.lr_scheduler = args.lr_scheduler.lower()
    warmup_epochs_remaining = args.lr_warmup_epochs if epoch is None else max(0, args.lr_warmup_epochs - epoch)
    
    # Prepare main scheduler args
    if args.lr_scheduler == "steplr":
        main_schedArgs = {
            "step_size": args.lr_step_size,
            "gamma": args.lr_gamma,
        }
    elif args.lr_scheduler == "cosineannealinglr":
        main_schedArgs = {
            "T_max": args.epochs - args.lr_warmup_epochs,
            "eta_min": args.lr_min,
        }
    elif args.lr_scheduler == "exponentiallr":
        main_schedArgs = {
            "gamma": args.lr_gamma,
        }
    elif args.lr_scheduler == "reducelronplateau":
        main_schedArgs = {
            "mode": "max",
            "factor": 0.1,
            "patience": 10,
        }
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR, ExponentialLR and ReduceLROnPlateau "
            "are supported."
        )

    # If warmup is needed, create main scheduler manually and wrap with warmup using SequentialLR
    # Note: ReduceLROnPlateau cannot be used with SequentialLR, so skip warmup for it
    if warmup_epochs_remaining > 0 and args.lr_scheduler != "reducelronplateau":
        # Set scheduler to SequentialLR for PerforatedAI
        GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.SequentialLR)
        
        # Determine main scheduler class
        if args.lr_scheduler == "steplr":
            main_scheduler_class = torch.optim.lr_scheduler.StepLR
        elif args.lr_scheduler == "cosineannealinglr":
            main_scheduler_class = torch.optim.lr_scheduler.CosineAnnealingLR
        elif args.lr_scheduler == "exponentiallr":
            main_scheduler_class = torch.optim.lr_scheduler.ExponentialLR
        
        # Determine warmup scheduler class and args
        if args.lr_warmup_method == "linear":
            warmup_scheduler_class = torch.optim.lr_scheduler.LinearLR
            warmup_schedArgs = {
                "start_factor": args.lr_warmup_decay,
                "total_iters": warmup_epochs_remaining,
            }
        elif args.lr_warmup_method == "constant":
            warmup_scheduler_class = torch.optim.lr_scheduler.ConstantLR
            warmup_schedArgs = {
                "factor": args.lr_warmup_decay,
                "total_iters": warmup_epochs_remaining,
            }
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        
        # Create SequentialLR args with both scheduler classes and their kwargs
        sequential_schedArgs = {
            "schedulers": [
                (warmup_scheduler_class, warmup_schedArgs),
                (main_scheduler_class, main_schedArgs)
            ],
            "milestones": [warmup_epochs_remaining]
        }
        optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, sequential_schedArgs)
    else:
        # No warmup needed, just create optimizer and scheduler through PerforatedAI
        if args.lr_scheduler == "steplr":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.StepLR)
        elif args.lr_scheduler == "cosineannealinglr":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR)
        elif args.lr_scheduler == "exponentiallr":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ExponentialLR)
        elif args.lr_scheduler == "reducelronplateau":
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
        
        optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, main_schedArgs)
        
        
        
    return optimizer, lr_scheduler


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        # Filter to ImageNet-100
        dataset = filter_imagenet100(dataset)
        
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        # Filter to ImageNet-100
        dataset_test = filter_imagenet100(dataset_test)
        
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")
    if args.distributed:
        if hasattr(args, "ra_sampler") and args.ra_sampler:
            train_sampler = RASampler(dataset, shuffle=True, repetitions=args.ra_reps)
        else:
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args, run=None):
    # Get config from wandb run if available
    if run is not None:
        config = run.config
        
        # Decode model architecture (0-3 maps to model names)
        model_map = {
            0: "resnet18",
            1: "resnet18_thin",
            2: "resnet10_shallow",
            3: "resnet12_balanced"
        }
        args.model = model_map.get(config.model_arch, "resnet18")
        
        # Decode convert_count (0-4)
        args.convert_count = config.convert_count
        
        # Apply weight decay from config
        args.weight_decay = config.weight_decay
        
        # Decode LR scheduler mode: 0=steplr, 1=cosineannealinglr, 2=reducelronplateau
        scheduler_map = {0: "steplr", 1: "cosineannealinglr", 2: "reducelronplateau"}
        args.lr_scheduler = scheduler_map.get(config.lr_scheduler_mode, "steplr")
        
        # Decode warmup epochs: 0=default (0), 1=custom (5)
        args.lr_warmup_epochs = 5 if config.warmup_epochs_mode == 1 else 0
        
        # Decode warmup method: 0=default (constant), 1=custom (linear)
        args.lr_warmup_method = "linear" if config.warmup_method_mode == 1 else "constant"
        
        # Decode label smoothing: 0=default (0.0), 1=custom (0.1)
        args.label_smoothing = 0.1 if config.label_smoothing_mode == 1 else 0.0
        
        # Decode mixup alpha: 0=default (0.0), 1=custom (0.2)
        args.mixup_alpha = 0.2 if config.mixup_alpha_mode == 1 else 0.0
        
        # Decode cutmix alpha: 0=default (0.0), 1=custom (0.2)
        args.cutmix_alpha = 0.2 if config.cutmix_alpha_mode == 1 else 0.0
        
        # Decode random erase: 0=default (0.0), 1=custom (0.1)
        args.random_erase = 0.1 if config.random_erase_mode == 1 else 0.0
        
        # Decode dropout: 0=default (0.0), 1=custom (0.2)
        args.dropout = 0.2 if config.dropout_mode == 1 else 0.0
        
        # Decode auto augment: 0=default (None), 1=custom (ta_wide)
        args.auto_augment = "ta_wide" if config.auto_augment_mode == 1 else None
        
        print(f"Sweep config: model={args.model}, convert_count={args.convert_count}, weight_decay={args.weight_decay}")
        print(f"LR config: scheduler={args.lr_scheduler}, warmup_epochs={args.lr_warmup_epochs}, warmup_method={args.lr_warmup_method}")
        print(f"Aug config: label_smooth={args.label_smoothing}, mixup={args.mixup_alpha}, cutmix={args.cutmix_alpha}, "
              f"random_erase={args.random_erase}, dropout={args.dropout}, auto_aug={args.auto_augment}")
        
        # Set wandb run name
        run.name = f"{args.model}_c{args.convert_count}_wd{args.weight_decay}_dmode{config.dendrite_mode}_sched{config.lr_scheduler_mode}"
    
    if args.output_dir:
        utils.mkdir(args.output_dir)

    # Apply batch_lr_factor scaling
    if args.batch_lr_factor != 1.0:
        original_batch_size = args.batch_size
        original_lr = args.lr
        args.batch_size = int(args.batch_size * args.batch_lr_factor)
        args.lr = args.lr * args.batch_lr_factor
        print(f"Applied batch_lr_factor={args.batch_lr_factor}: batch_size {original_batch_size}->{args.batch_size}, lr {original_lr}->{args.lr}")

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    num_classes = len(dataset.classes)
    print(f"Training with {num_classes} classes (ImageNet-100 subset)")
    
    # Set up PerforatedAI global parameters
    GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_n_epochs_to_switch(40)
    GPA.pc.set_p_epochs_to_switch(40)
    GPA.pc.set_cap_at_n(True)
    GPA.pc.set_initial_history_after_switches(2)
    GPA.pc.set_test_saves(True)
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.append_module_names_to_convert(["BasicBlock", "Bottleneck"])
    GPA.pc.set_verbose(False)
    GPA.pc.set_max_dendrites(3)
    
    # Apply wandb config to PAI settings if available, otherwise use command-line args
    if run is not None:
        config = run.config
        
        # Decode improvement_threshold
        if config.improvement_threshold == 0:
            thresh = [0.01, 0.001, 0.0001, 0]
        elif config.improvement_threshold == 1:
            thresh = [0.001, 0.0001, 0]
        elif config.improvement_threshold == 2:
            thresh = [0]
        GPA.pc.set_improvement_threshold(thresh)
        
        GPA.pc.set_candidate_weight_initialization_multiplier(
            config.candidate_weight_initialization_multiplier
        )
        
        # Decode pai_forward_function
        if config.pai_forward_function == 0:
            pai_forward_function = torch.sigmoid
        elif config.pai_forward_function == 1:
            pai_forward_function = torch.relu
        elif config.pai_forward_function == 2:
            pai_forward_function = torch.tanh
        else:
            pai_forward_function = torch.sigmoid
        GPA.pc.set_pai_forward_function(pai_forward_function)
        
        # Set dendrite mode
        if config.dendrite_mode == 0:
            GPA.pc.set_max_dendrites(0)
        elif config.dendrite_mode == 1:
            GPA.pc.set_max_dendrites(3)
            GPA.pc.set_perforated_backpropagation(False)
        elif config.dendrite_mode == 2:
            GPA.pc.set_max_dendrites(3)
            GPA.pc.set_perforated_backpropagation(True)
        
        print(f"PAI config: improvement_threshold={config.improvement_threshold}, "
              f"init_mult={config.candidate_weight_initialization_multiplier}, "
              f"forward_fn={config.pai_forward_function}, dendrite_mode={config.dendrite_mode}")
    else:
        # Use command-line args for PAI settings
        if args.improvement_threshold == 0:
            thresh = [0.01, 0.001, 0.0001, 0]
        elif args.improvement_threshold == 1:
            thresh = [0.001, 0.0001, 0]
        elif args.improvement_threshold == 2:
            thresh = [0]
        GPA.pc.set_improvement_threshold(thresh)
        
        GPA.pc.set_candidate_weight_initialization_multiplier(args.candidate_weight_init_mult)
        
        # Decode pai_forward_function from string
        if args.pai_forward_function == "sigmoid":
            pai_forward_function = torch.sigmoid
        elif args.pai_forward_function == "relu":
            pai_forward_function = torch.relu
        elif args.pai_forward_function == "tanh":
            pai_forward_function = torch.tanh
        else:
            pai_forward_function = torch.sigmoid
        GPA.pc.set_pai_forward_function(pai_forward_function)
        
        # Set dendrite mode
        if args.dendrite_mode == 0:
            GPA.pc.set_max_dendrites(0)
        elif args.dendrite_mode == 1:
            GPA.pc.set_max_dendrites(3)
            GPA.pc.set_perforated_backpropagation(False)
        elif args.dendrite_mode == 2:
            GPA.pc.set_max_dendrites(3)
            GPA.pc.set_perforated_backpropagation(True)
        
        print(f"PAI config: improvement_threshold={args.improvement_threshold}, "
              f"init_mult={args.candidate_weight_init_mult}, "
              f"forward_fn={args.pai_forward_function}, dendrite_mode={args.dendrite_mode}")
    
    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha, cutmix_alpha=args.cutmix_alpha, num_classes=num_classes, use_v2=args.use_v2
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    # Check if it's one of our custom models
    if args.model in ['resnet18_thin', 'resnet10_shallow', 'resnet12_balanced']:
        model_fn = getattr(custom_resnet, args.model)
        model = model_fn(num_classes=num_classes)
        print(f"Created custom model: {args.model}")
    else:
        model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    
    # Apply dropout if specified (add dropout after global average pooling, before final classifier)
    if args.dropout > 0.0:
        # For ResNet models, insert dropout before the final fc layer
        if hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout),
                nn.Linear(in_features, num_classes)
            )
            print(f"Applied dropout rate: {args.dropout}")
    
    # Apply stochastic depth if specified (for ResNet models)
    if args.stochastic_depth_prob > 0.0:
        print(f"Note: Stochastic depth rate {args.stochastic_depth_prob} specified, but requires model recreation with stochastic_depth parameter")
        print(f"Consider using: torchvision.models.resnet18(weights=None, num_classes={num_classes}, stochastic_depth_prob={args.stochastic_depth_prob})")
    
    # Note on width/depth multipliers
    if args.width_multiplier != 1.0 or args.depth_multiplier != 1.0:
        print(f"Note: Width multiplier {args.width_multiplier} and/or depth multiplier {args.depth_multiplier} specified")
        print(f"These require custom model creation. Consider using smaller models like resnet18 or using torchvision.models.efficientnet with different variants")
    

    skip_layers = 4-args.convert_count

    for i in range(skip_layers):
        GPA.pc.append_module_ids_to_track(['.layer'+str(i+1)])
    # Wrap model with PerforatedAI
    model = custom_resnet.ResNetPAI(model)
    
    # Build run name with priority ordering
    excluded = ['method', 'metric', 'parameters']
    priorities = ['dendrite_mode', 'model_arch']
    # Add priority keys first with their names
    name_parts = [f"{k}={wandb.config[k]}" for k in priorities if k in wandb.config]
    # Add remaining keys in default order without names
    remaining_keys = [k for k in parameters_dict.keys() if k not in excluded and k not in priorities]
    name_parts.extend(str(wandb.config[k]) for k in remaining_keys if k in wandb.config)
    name_str = "_".join(name_parts)
    run.name = name_str

    model = UPA.initialize_pai(model, save_name=run.name)

    model.to(device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, args, custom_keys_weight_decay)

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    model_ema = None
    if args.model_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])

    if args.test_only:
        # We disable the cudnn benchmarking because it can noticeably affect the accuracy
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            evaluate(model, criterion, data_loader_test, device=device)
        return

    print("Start training")
    start_time = time.time()
    epoch = args.start_epoch - 1
    
    # Initialize tracking variables for wandb logging
    max_val_acc1 = 0
    max_train_acc1 = 0
    max_params = 0
    dendrite_count = 0
    original_model = model

    while True:
        epoch += 1
#    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        # This is done in the pai backend now
        # lr_scheduler.step()
        
        model, acc1, restructured, trainingComplete = evaluate(model, criterion, data_loader_test, device=device)
        
        # Get training accuracy from PAI tracker extra scores
        train_acc1 = GPA.pai_tracker.member_vars.get("extra_scores", {}).get("Train Acc 1", 0)
        
        # Update max values
        if acc1 > max_val_acc1:
            max_val_acc1 = acc1
            max_train_acc1 = train_acc1
            max_params = UPA.count_params(model)
        
        # Log to wandb
        if run is not None:
            run.log({
                "ValAcc": acc1,
                "TrainAcc": train_acc1,
                "Param Count": UPA.count_params(model),
                "Dendrite Count": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
                "epoch": epoch
            })
            
            # Log architecture maximums when dendrites are added
            if restructured:
                if GPA.pai_tracker.member_vars["mode"] == "n" and (
                    dendrite_count != GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
                ):
                    dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
                    run.log({
                        "Arch Max Val": max_val_acc1,
                        "Arch Max Train": max_train_acc1,
                        "Arch Param Count": max_params,
                        "Arch Dendrite Count": dendrite_count - 1,
                    })
        
        # If model was restructured by PerforatedAI, reset optimizer and scheduler
        if restructured:
            model.to(device)
            optimizer, lr_scheduler = create_optimizer_and_scheduler(model, args, custom_keys_weight_decay, epoch=epoch)

        if model_ema:
            evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
        
        # Check if PerforatedAI training is complete
        if trainingComplete:
            print("PerforatedAI training complete!")
            
            # Log final architecture max
            if run is not None:
                run.log({
                    "Final Max Val": max_val_acc1,
                    "Final Max Train": max_train_acc1,
                    "Final Param Count": max_params,
                    "Final Dendrite Count": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
                })
            break
    print("Final Param Count:", UPA.count_params(model))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training with PerforatedAI (Fast - ImageNet-100, Half Resolution)", add_help=add_help)

    parser.add_argument("--data-path", default="/home/rbrenner/Datasets/imagenet", type=str, help="dataset path")
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=256, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "--batch-lr-factor", default=1.0, type=float, help="factor to scale batch size and learning rate (e.g., 0.5 halves batch size and scales lr accordingly)"
    )
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "--norm-weight-decay",
        default=None,
        type=float,
        help="weight decay for Normalization layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--bias-weight-decay",
        default=None,
        type=float,
        help="weight decay for bias parameters of all layers (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--transformer-embedding-decay",
        default=None,
        type=float,
        help="weight decay for embedding parameters for vision transformer models (default: None, same value as --wd)",
    )
    parser.add_argument(
        "--label-smoothing", default=0.0, type=float, help="label smoothing (default: 0.0)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)")
    parser.add_argument("--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=None, type=str, help="path to save outputs")
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: None)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=0.0, type=float, help="random erasing probability (default: 0.0)")

    # Regularization parameters to reduce overfitting (train-val gap)
    parser.add_argument("--dropout", default=0.0, type=float, help="dropout rate (default: 0.0, no dropout)")
    parser.add_argument("--width-multiplier", default=1.0, type=float, help="network width multiplier to reduce capacity (default: 1.0, full width)")
    parser.add_argument("--depth-multiplier", default=1.0, type=float, help="network depth multiplier to reduce capacity (default: 1.0, full depth)")
    parser.add_argument(
        "--stochastic-depth-prob",
        default=0.0,
        type=float,
        help="stochastic depth drop probability for ResNet (default: 0.0, no stochastic depth)",
    )

    # Mixed precision training parameters
    parser.add_argument("--amp", action="store_true", help="Use torch.cuda.amp for mixed precision training")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--model-ema", action="store_true", help="enable tracking Exponential Moving Average of model parameters"
    )
    parser.add_argument(
        "--model-ema-steps",
        type=int,
        default=32,
        help="the number of iterations that controls how often to update the EMA model (default: 32)",
    )
    parser.add_argument(
        "--model-ema-decay",
        type=float,
        default=0.99998,
        help="decay factor for Exponential Moving Average of model parameters (default: 0.99998)",
    )
    parser.add_argument(
        "--use-deterministic-algorithms", action="store_true", help="Forces the use of deterministic algorithms only."
    )
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    # Half resolution defaults (128 instead of 256, 112 instead of 224)
    parser.add_argument(
        "--val-resize-size", default=128, type=int, help="the resize size used for validation (default: 128 for fast training)"
    )
    parser.add_argument(
        "--val-crop-size", default=112, type=int, help="the central crop size used for validation (default: 112 for fast training)"
    )
    parser.add_argument(
        "--train-crop-size", default=112, type=int, help="the random crop size used for training (default: 112 for fast training)"
    )
    parser.add_argument("--convert-count", default=4, type=int, help="total number of layers to convert")
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    
    # PerforatedAI parameters
    parser.add_argument("--improvement-threshold", default=0, type=int, choices=[0, 1, 2], 
                        help="PAI improvement threshold mode: 0=[0.01,0.001,0.0001,0], 1=[0.001,0.0001,0], 2=[0]")
    parser.add_argument("--candidate-weight-init-mult", default=0.1, type=float,
                        help="PAI candidate weight initialization multiplier (default: 0.1)")
    parser.add_argument("--pai-forward-function", default="sigmoid", type=str, choices=["sigmoid", "relu", "tanh"],
                        help="PAI forward function (default: sigmoid)")
    parser.add_argument("--dendrite-mode", default=2, type=int, choices=[0, 1, 2],
                        help="Dendrite mode: 0=no dendrites, 1=GD dendrites, 2=PB dendrites (default: 2)")
    
    # Wandb sweep parameters
    parser.add_argument("--sweep-id", type=str, default="main", help='Sweep ID to join, or "main" to create new sweep')
    parser.add_argument("--use-wandb", action="store_true", help="Use wandb for sweep")
    parser.add_argument("--count", type=int, default=50, help="Number of sweep runs to perform")
    
    return parser


def get_parameters_dict():
    """Return the parameters dictionary for the sweep."""
    parameters_dict = {
        # Model architecture: 0=resnet18, 1=resnet18_thin, 2=resnet10_shallow, 3=resnet12_balanced
        "model_arch": {"values": [0, 1, 2, 3]},
        # Number of layers to convert (0-4)
        "convert_count": {"values": [0, 1, 2, 3, 4]},
        # Weight decay values
        "weight_decay": {"values": [0, 0.0001, 0.001]},
        # Used for all dendritic models:
        # 0 = [0.01, 0.001, 0.0001, 0], 1 = [0.001, 0.0001, 0], 2 = [0]
        "improvement_threshold": {"values": [0, 1, 2]},
        "candidate_weight_initialization_multiplier": {"values": [0.1, 0.01]},
        # 0 = sigmoid, 1 = relu, 2 = tanh
        "pai_forward_function": {"values": [0, 1, 2]},
        # dendrite_mode: 0 = no dendrites, 1 = GD dendrites, 2 = CC dendrites (PB)
        "dendrite_mode": {"values": [1, 2]},
        
        # Training hyperparameters from command line: 0=default, 1=custom value
        # LR scheduler: 0=steplr (default), 1=cosineannealinglr, 2=reducelronplateau
        "lr_scheduler_mode": {"values": [0, 1, 2]},
        # Warmup epochs: 0=default (0), 1=custom (5)
        "warmup_epochs_mode": {"values": [0, 1]},
        # Warmup method: 0=default (constant), 1=custom (linear)
        "warmup_method_mode": {"values": [0, 1]},
        # Weight decay: 0=default (1e-4), 1=custom (2e-4) - already in weight_decay above
        # Label smoothing: 0=default (0.0), 1=custom (0.1)
        "label_smoothing_mode": {"values": [0, 1]},
        # Mixup alpha: 0=default (0.0), 1=custom (0.2)
        "mixup_alpha_mode": {"values": [0, 1]},
        # Cutmix alpha: 0=default (0.0), 1=custom (0.2)
        "cutmix_alpha_mode": {"values": [0, 1]},
        # Random erase: 0=default (0.0), 1=custom (0.1)
        "random_erase_mode": {"values": [0, 1]},
        # Dropout: 0=default (0.0), 1=custom (0.2)
        "dropout_mode": {"values": [0, 1]},
        # Auto augment: 0=default (None), 1=custom (ta_wide)
        "auto_augment_mode": {"values": [0, 1]},
    }
    return parameters_dict


def run_sweep():
    """Wrapper function for wandb sweep."""
    try:
        with wandb.init() as wandb_run:
            # Parse args without wandb parameters
            args = get_args_parser().parse_args()
            main(args, wandb_run)
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import sys
    
    # Parse minimal args for sweep control
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sweep-id", type=str, default="")
    parser.add_argument("--count", type=int, default=50)
    prelim_args, _ = parser.parse_known_args()
    
    if prelim_args.sweep_id != "":
        # Run with wandb sweep
        wandb.login()
        project = "ImageNet-100 PerforatedAI Sweep"
        sweep_config = {"method": "random"}
        metric = {"name": "Final Max Val", "goal": "maximize"}
        sweep_config["metric"] = metric
        parameters_dict = get_parameters_dict()
        sweep_config["parameters"] = parameters_dict

        if prelim_args.sweep_id == "main":
            sweep_id = wandb.sweep(sweep_config, project=project)
            print(
                f"\nInitialized sweep. Use --sweep-id {sweep_id} to join on other machines.\n"
            )
            # Run the agent on this machine
            wandb.agent(sweep_id, run_sweep, count=prelim_args.count)
        else:
            # Join the existing sweep as an agent
            wandb.agent(prelim_args.sweep_id, run_sweep, count=prelim_args.count, project=project)
    else:
        # Run without wandb
        args = get_args_parser().parse_args()
        main(args)
