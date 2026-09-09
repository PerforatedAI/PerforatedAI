"""
Training MobileNetV3 Small with PerforatedAI on ImageNet.

Based on PyTorch official training recipe for MobileNetV3.

Single GPU default command (scaled from the original 8-GPU recipe):
python train_perforated_mobilenetv3.py \
  --model mobilenet_v3_small --epochs 600 --opt rmsprop --batch-size 128 --lr 0.008 \
  --wd 0.00001 --lr-step-size 2 --lr-gamma 0.973 --auto-augment imagenet --random-erase 0.2 \
  --full-dataset --data-path /home/rbrenner/Datasets/imagenet \
  --convert-count 0 --dendrite-mode 2 --improvement-threshold 1 \
  --candidate-weight-init-mult 0.1 --pai-forward-function relu

This matches the original 8-GPU setup with 128 images/GPU and LR 0.064, scaled for a single GPU.

Note: For MobileNetV3 Large, use:
  --model mobilenet_v3_large
  
Note: This script implements checkpoint averaging using PerforatedAI's checkpoint system:
  - Tracks the top 3 checkpoints during 'n' mode (neuron addition) by accuracy
  - Uses UPA.save_system() to save checkpoints with PAI's complete system state
  - When transitioning to 'p' mode (pruning), loads all 3 checkpoints using UPA.load_system()
  - Averages their weights while preserving the PAI system state from the best checkpoint
  - This ensures proper dendrite state, tracker history, and optimizer state are maintained
"""

import datetime
import os
import time
import warnings
import argparse
import collections
import shutil

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
from perforatedai import network_perforatedai as NPA

from clean_somas import CleanSomas
from rf_dendrites_original import initialize_variant_dendrite, SparseLinear

import wandb
from types import SimpleNamespace


def pai_identity(x):
    return x


class Top3CheckpointTracker:
    """
    Tracks the top 3 checkpoints during 'n' mode and provides averaging functionality.
    Uses _pai.pt file copying for checkpoint management.
    """
    def __init__(self, save_name, model_name, num_classes):
        self.save_name = save_name
        self.model_name = model_name
        self.num_classes = num_classes
        self.top3_checkpoints = []  # List of (acc1, epoch) tuples
        self.current_mode = None
        
    def update(self, acc1, epoch, model, current_mode):
        """Update tracker with new checkpoint if it's in top 3."""
        self.current_mode = current_mode
        
        # Only track during 'n' mode
        if current_mode != "n":
            return
        
        # Only track after at least one dendrite has been added
        # (to ensure checkpoint has layer_array structure with dendrites)
        dendrite_count = GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
        if dendrite_count == 0:
            return
        
        # Copy first, then evict — so if this epoch ranks outside top 3 its file
        # is created and immediately deleted in the same cleanup pass below
        source_file = f"{self.save_name}/latest_pai.pt"
        dest_file = f"{self.save_name}/top3_epoch_{epoch}_pai.pt"

        if os.path.exists(source_file):
            shutil.copy2(source_file, dest_file)
            print(f"Saved top-3 candidate: epoch {epoch} with Acc@1 {acc1:.3f}")
        else:
            print(f"Warning: {source_file} not found - checkpoint not saved")

        # Add to list, sort, and evict anything outside top 3
        self.top3_checkpoints.append((acc1, epoch))
        self.top3_checkpoints.sort(key=lambda x: x[0], reverse=True)

        if len(self.top3_checkpoints) > 3:
            removed = self.top3_checkpoints[3:]
            self.top3_checkpoints = self.top3_checkpoints[:3]

            for _, old_epoch in removed:
                checkpoint_file = f"{self.save_name}/top3_epoch_{old_epoch}_pai.pt"
                if os.path.exists(checkpoint_file):
                    os.remove(checkpoint_file)
                    print(f"Removed checkpoint file: top3_epoch_{old_epoch}_pai.pt")
    
    def get_top3_info(self):
        """Return information about top 3 checkpoints."""
        return [(acc, epoch) for acc, epoch in self.top3_checkpoints]
    
    def get_top3_epochs(self):
        """Return list of top 3 epochs."""
        return [epoch for _, epoch in self.top3_checkpoints]
    
    def average_and_load(self, model, device):
        """
        Average the top 3 checkpoints and load into model.
        
        Workflow:
        1. For each checkpoint, create a fresh unperforated model
        2. Load checkpoint using NPA.load_pai_model() (which perforates and loads weights)
        3. Extract state dicts and average them
        4. Load averaged weights back into the current model
        """
        if len(self.top3_checkpoints) == 0:
            print("No checkpoints to average!")
            return model
        
        epochs = [epoch for _, epoch in self.top3_checkpoints]
        accs = [acc for acc, _ in self.top3_checkpoints]
        
        print(f"\nAveraging top {len(epochs)} checkpoints:")
        for acc, epoch in zip(accs, epochs):
            print(f"  - Epoch {epoch}: Acc@1 {acc:.3f}")
        
        # Collect state dicts from all top 3 checkpoints
        # NOTE: Checkpoints have N-1 dendrites (saved before final restructure)
        # Current model has N dendrites (just added during transition to 'p' mode)
        all_state_dicts = []
        for epoch in epochs:
            checkpoint_file = f"{self.save_name}/top3_epoch_{epoch}_pai.pt"
            
            if not os.path.exists(checkpoint_file):
                print(f"Warning: {checkpoint_file} not found, skipping")
                continue
            
            # Load checkpoint directly as state dict
            from safetensors.torch import load_file
            checkpoint_state = load_file(checkpoint_file)
            
            # Clone all tensors to CPU for averaging
            state_dict_cpu = {k: v.clone().cpu() for k, v in checkpoint_state.items()}
            all_state_dicts.append(state_dict_cpu)
            print(f"  - Loaded state dict from epoch {epoch}")
        
        if len(all_state_dicts) == 0:
            print("Error: No checkpoints could be loaded!")
            return model
        
        # Average the weights
        print("  - Computing averaged weights...")
        averaged_state = collections.OrderedDict()
        for key in all_state_dicts[0].keys():
            # Stack tensors and compute mean
            tensor_list = [d[key].float() for d in all_state_dicts]
            stacked = torch.stack(tensor_list, dim=0)
            averaged = torch.mean(stacked, dim=0)
            
            # Convert back to original dtype if needed
            if all_state_dicts[0][key].dtype != averaged.dtype:
                averaged = averaged.to(all_state_dicts[0][key].dtype)
            
            averaged_state[key] = averaged
        
        # Load averaged weights into model (strict=False to ignore new dendrite keys)
        missing_keys, unexpected_keys = model.load_state_dict(averaged_state, strict=False)
        if missing_keys:
            print(f"  - Note: {len(missing_keys)} keys not in averaged checkpoints (newly added dendrite)")
        if unexpected_keys:
            print(f"  - Warning: {len(unexpected_keys)} unexpected keys in averaged state")
        model = model.to(device)
        print(f"✓ Loaded averaged weights into model\n")
        
        return model
    
    def clear(self):
        """Clear the tracker (e.g., after averaging)."""
        self.top3_checkpoints = []


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"

    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
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

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )

    # Add validation score to PerforatedAI tracker and check for restructuring
    GPA.pai_tracker.add_extra_score(metric_logger.acc5.global_avg, "Val Acc 5")
    model, restructured, trainingComplete = GPA.pai_tracker.add_validation_score(
        metric_logger.acc1.global_avg, model
    )

    return model, metric_logger.acc1.global_avg, restructured, trainingComplete


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


# ImageNet-100 standard class indices (commonly used subset)
IMAGENET100_CLASSES = [
    "n01440764",
    "n01443537",
    "n01484850",
    "n01491361",
    "n01494475",
    "n01496331",
    "n01498041",
    "n01514668",
    "n01514859",
    "n01518878",
    "n01530575",
    "n01531178",
    "n01532829",
    "n01534433",
    "n01537544",
    "n01558993",
    "n01560419",
    "n01580077",
    "n01582220",
    "n01592084",
    "n01601694",
    "n01608432",
    "n01614925",
    "n01616318",
    "n01622779",
    "n01629819",
    "n01630670",
    "n01631663",
    "n01632458",
    "n01632777",
    "n01641577",
    "n01644373",
    "n01644900",
    "n01664065",
    "n01665541",
    "n01667114",
    "n01667778",
    "n01669191",
    "n01675722",
    "n01677366",
    "n01682714",
    "n01685808",
    "n01687978",
    "n01688243",
    "n01689811",
    "n01692333",
    "n01693334",
    "n01694178",
    "n01695060",
    "n01697457",
    "n01698640",
    "n01704323",
    "n01728572",
    "n01728920",
    "n01729322",
    "n01729977",
    "n01734418",
    "n01735189",
    "n01737021",
    "n01739381",
    "n01740131",
    "n01742172",
    "n01744401",
    "n01748264",
    "n01749939",
    "n01751748",
    "n01753488",
    "n01755581",
    "n01756291",
    "n01768244",
    "n01770081",
    "n01770393",
    "n01773157",
    "n01773549",
    "n01773797",
    "n01774384",
    "n01774750",
    "n01775062",
    "n01776313",
    "n01784675",
    "n01795545",
    "n01796340",
    "n01797886",
    "n01798484",
    "n01806143",
    "n01806567",
    "n01807496",
    "n01817953",
    "n01818515",
    "n01819313",
    "n01820546",
    "n01824575",
    "n01828970",
    "n01829413",
    "n01833805",
    "n01843065",
    "n01843383",
    "n01847000",
    "n01855032",
    "n01855672",
]


def filter_imagenet100(dataset):
    """Filter dataset to only include ImageNet-100 classes."""
    # Get original class_to_idx mapping
    original_class_to_idx = dataset.class_to_idx

    # Create mapping from old indices to new indices
    valid_classes = [cls for cls in IMAGENET100_CLASSES if cls in original_class_to_idx]
    new_class_to_idx = {cls: new_idx for new_idx, cls in enumerate(valid_classes)}
    old_to_new_idx = {
        original_class_to_idx[cls]: new_idx for cls, new_idx in new_class_to_idx.items()
    }

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

    print(
        f"Filtered dataset to {len(valid_classes)} classes with {len(filtered_samples)} samples"
    )
    return dataset


def initialize_dendrites(model, n):
    """Initialize PAI module shapes and pre-grow n dendrites.

    Runs a single dummy forward+backward (no optimizer step) so PAI's
    internal out_channels arrays are set, then calls simulate_cycles with
    2*n cycles which adds n dendrites (each n→p transition adds one).
    """
    device = next(model.parameters()).device
    dummy_x = torch.zeros(1, 3, 224, 224, device=device)
    out = model(dummy_x)
    out.sum().backward()
    model.zero_grad()

    for module in model.modules():
        if hasattr(module, 'dendrite_module'):
            UPA.simulate_cycles(module, n * 2, doing_pai=True)

    from rf_dendrites_original import MaskedLinear, SparseLinear
    filled = 0
    for module in model.modules():
        if hasattr(module, 'dendrites_to_top') and len(module.dendrites_to_top) > 0:
            module.dendrites_to_top[-1].data.fill_(1.0)
            filled += 1
            print(f"  to_top fill: {type(module).__name__}, "
                  f"shape={module.dendrites_to_top[-1].shape}, "
                  f"dendrites_added={module.dendrite_modules_added}")
    print(f"initialize_dendrites: filled {filled} module(s)")

    # The approved dendrite bypasses init_params and uses xavier_uniform_.
    # Reinit to match nn.Linear defaults exactly (kaiming_uniform_ + bias uniform)
    # so CleanSomas + fully-connected identity dendrite == nn.Linear.
    import math
    for module in model.modules():
        if isinstance(module, (MaskedLinear, SparseLinear)):
            fan_in = module.in_features
            a = math.sqrt(5)
            bound_w = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a ** 2) / fan_in)
            nn.init.uniform_(module.weight, -bound_w, bound_w)
            nn.init.uniform_(module.bias, -1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))


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
        custom_keys_weight_decay=(
            custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None
        ),
    )

    # Apply dendrite LR multiplier on subsequent n-phase cycles
    current_mode = GPA.pai_tracker.member_vars.get("mode", "n")
    num_cycles = GPA.pai_tracker.member_vars.get("num_cycles", 0)
    effective_lr = args.lr
    if current_mode == "n" and num_cycles > 1 and args.dendrite_lr_multiplier != 1.0:
        effective_lr = args.lr * args.dendrite_lr_multiplier
        print(f"[dendrite_lr_multiplier] mode=n, num_cycles={num_cycles}, lr {args.lr} -> {effective_lr}")

    # Set optimizer class
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        GPA.pai_tracker.set_optimizer(torch.optim.SGD)
        optimArgs = {
            "params": parameters,
            "lr": effective_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "nesterov": "nesterov" in opt_name,
        }
    elif opt_name == "rmsprop":
        GPA.pai_tracker.set_optimizer(torch.optim.RMSprop)
        optimArgs = {
            "params": parameters,
            "lr": effective_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "eps": 0.0316,
            "alpha": 0.9,
        }
    elif opt_name == "adamw":
        GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
        optimArgs = {
            "params": parameters,
            "lr": effective_lr,
            "weight_decay": args.weight_decay,
        }
    else:
        raise RuntimeError(
            f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported."
        )

    # Set scheduler class and prepare scheduler args
    args.lr_scheduler = args.lr_scheduler.lower()
    warmup_epochs_remaining = (
        args.lr_warmup_epochs
        if epoch is None
        else max(0, args.lr_warmup_epochs - epoch)
    )

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
                (main_scheduler_class, main_schedArgs),
            ],
            "milestones": [warmup_epochs_remaining],
        }
        optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, sequential_schedArgs
        )
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

        optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, main_schedArgs
        )

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
        # Filter to ImageNet-100 unless full dataset is requested
        if not args.full_dataset:
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
                preprocessing = torchvision.transforms.Compose(
                    [torchvision.transforms.PILToTensor(), preprocessing]
                )

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
        # Filter to ImageNet-100 unless full dataset is requested
        if not args.full_dataset:
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
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset_test, shuffle=False
        )
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


def main(args):
    TESTING = False  # True = DOING_FIXED_SWITCH with fixed_switch_num=3 (load testing)

    # Initialize wandb if enabled
    run = None
    if args.use_wandb:
        run = wandb.init(
            project="ImageNet MobileNetV3 PerforatedAI",
            name=f"{args.model}_c{args.convert_count}_wd{args.weight_decay}_dmode{args.dendrite_mode}",
            config=vars(args),
        )
        print(f"Logging to wandb run: {run.name}")

    print(
        f"Config: model={args.model}, convert_count={args.convert_count}, weight_decay={args.weight_decay}"
    )
    print(
        f"LR config: scheduler={args.lr_scheduler}, warmup_epochs={args.lr_warmup_epochs}, warmup_method={args.lr_warmup_method}"
    )
    print(
        f"Aug config: label_smooth={args.label_smoothing}, mixup={args.mixup_alpha}, cutmix={args.cutmix_alpha}, "
        f"random_erase={args.random_erase}, dropout={args.dropout}, auto_aug={args.auto_augment}"
    )
    print(
        f"PAI config: improvement_threshold={args.improvement_threshold}, "
        f"init_mult={args.candidate_weight_init_mult}, "
        f"forward_fn={args.pai_forward_function}, dendrite_mode={args.dendrite_mode}"
    )

    if args.output_dir:
        utils.mkdir(args.output_dir)

    # Apply batch_lr_factor scaling
    if args.batch_lr_factor != 1.0:
        original_batch_size = args.batch_size
        original_lr = args.lr
        args.batch_size = int(args.batch_size * args.batch_lr_factor)
        args.lr = args.lr * args.batch_lr_factor
        print(
            f"Applied batch_lr_factor={args.batch_lr_factor}: batch_size {original_batch_size}->{args.batch_size}, lr {original_lr}->{args.lr}"
        )

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
    dataset, dataset_test, train_sampler, test_sampler = load_data(
        train_dir, val_dir, args
    )

    num_classes = len(dataset.classes)
    dataset_type = "full ImageNet" if args.full_dataset else "ImageNet-100 subset"
    print(f"Training with {num_classes} classes ({dataset_type})")

    # Set up PerforatedAI global parameters
    if TESTING:
        print("=" * 60)
        print("TESTING MODE ENABLED")
        print("Using DOING_FIXED_SWITCH with fixed_switch_num=3")
        print("=" * 60)
        GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
        GPA.pc.set_fixed_switch_num(3)
        GPA.pc.set_first_fixed_switch_num(3)
        GPA.pc.set_max_dendrites(3)
    else:
        GPA.pc.set_switch_mode(GPA.pc.DOING_HISTORY)
        GPA.pc.set_n_epochs_to_switch(40)
        GPA.pc.set_p_epochs_to_switch(40)
    
    GPA.pc.set_output_dimensions([-1,0])
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_cap_at_n(True)
    GPA.pc.set_initial_history_after_switches(2)
    GPA.pc.set_test_saves(True)
    GPA.pc.set_pai_saves(True)  # Enable _pai.pt checkpoint creation for averaging
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_verbose(False)

    # Apply PAI settings from command-line args
    if args.improvement_threshold == 0:
        thresh = [0.01, 0.001, 0.0001, 0]
    elif args.improvement_threshold == 1:
        thresh = [0.001, 0.0001, 0]
    elif args.improvement_threshold == 2:
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)

    GPA.pc.set_candidate_weight_initialization_multiplier(
        args.candidate_weight_init_mult
    )

    # Decode pai_forward_function from string
    if args.pai_forward_function == "sigmoid":
        pai_forward_function = torch.sigmoid
    elif args.pai_forward_function == "relu":
        pai_forward_function = torch.relu
    elif args.pai_forward_function == "tanh":
        pai_forward_function = torch.tanh
    elif args.pai_forward_function == "identity":
        pai_forward_function = pai_identity
    elif args.pai_forward_function == "hardswish":
        pai_forward_function = nn.functional.hardswish
    else:
        pai_forward_function = torch.sigmoid
    GPA.pc.set_pai_forward_function(pai_forward_function)

    # Set dendrite mode
    if args.dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
    elif args.dendrite_mode == 1:
        GPA.pc.set_max_dendrites(5)
        GPA.pc.set_perforated_backpropagation(False)
    elif args.dendrite_mode == 2:
        GPA.pc.set_max_dendrites(5)
        GPA.pc.set_perforated_backpropagation(True)

    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        num_classes=num_classes,
        use_v2=args.use_v2,
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
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )

    print("Creating model")
    # Load model from torchvision
    model = torchvision.models.get_model(
        args.model, weights=args.weights, num_classes=num_classes
    )

    # Apply dropout if specified (add dropout after global average pooling, before final classifier)
    if args.dropout > 0.0:
        # For models with classifier attribute (like MobileNet)
        if hasattr(model, "classifier"):
            # MobileNetV3 has a classifier Sequential with Dropout and Linear
            # We can adjust the dropout rate
            for module in model.classifier:
                if isinstance(module, nn.Dropout):
                    module.p = args.dropout
            print(f"Applied dropout rate: {args.dropout}")
        # For ResNet models, insert dropout before the final fc layer
        elif hasattr(model, "fc"):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=args.dropout), nn.Linear(in_features, num_classes)
            )
            print(f"Applied dropout rate: {args.dropout}")

    # Replace classifier[0] with CleanSomas — soma emits only a bias; all input
    # signal flows through the RF dendrites PAI will grow on this module.
    clf0_in = model.classifier[0].in_features
    clf0_out = model.classifier[0].out_features
    model.classifier[0] = CleanSomas(
        clf0_out,
        config={'in_features': clf0_in, 'out_features': clf0_out},
    )
    print(f"Replaced classifier[0] with CleanSomas({clf0_in}→{clf0_out})")
    # Identity activation: dendrite output is linear, making CleanSomas + fully
    # connected dendrite functionally identical to nn.Linear.
    GPA.pc.set_pai_forward_function(pai_identity)
    # CleanSomas has no gradient path through the soma (ignores x).
    # preprocess_pb would detach x before MaskedLinear, killing backbone grads.
    # Disabling dendrite_graph_mode lets gradients flow normally through n-mode.
    GPA.pc.set_dendrite_graph_mode(False)
    GPA.pc.append_module_ids_to_track([".features", ".avgpool", ".classifier.3"])
    GPA.pc.append_module_ids_to_perforate([".classifier.0"])

    # Build save name
    save_name = f"{args.model}_sparse_c{args.convert_count}_wd{args.weight_decay}_dmode{args.dendrite_mode}"
    if run is not None:
        run.name = save_name

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_name_with_timestamp = f"{save_name}_{timestamp}"

    # Load from checkpoint if path provided, otherwise initialize new
    if args.perforated_load_path != "":
        model = UPA.perforate_model(model, save_name=args.perforated_load_path)
        initialize_variant_dendrite(synapses=clf0_in // 4, rf_mode='random', sparse=True)
        model = UPA.load_system(model, args.perforated_load_path, args.load_checkpoint_name, True)
    else:
        model = UPA.perforate_model(model, save_name=save_name_with_timestamp)
        # Must be called after perforate_model so GPA.pai_tracker is initialized.
        initialize_variant_dendrite(synapses=clf0_in // 4, rf_mode='random', sparse=True)
    model.to(device)

    if args.perforated_load_path == "":
        print("Pre-growing 1 RF dendrite on classifier[0]...")
        initialize_dendrites(model, 1)
        print(f"Pre-grown dendrites complete. Param count: {UPA.count_params(model):,}")

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in [
            "class_token",
            "position_embedding",
            "relative_position_bias_table",
        ]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))

    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(
        model, args, custom_keys_weight_decay
    )

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
        model_ema = utils.ExponentialMovingAverage(
            model_without_ddp, device=device, decay=1.0 - alpha
        )

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
            evaluate(
                model_ema, criterion, data_loader_test, device=device, log_suffix="EMA"
            )
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
    
    # Initialize checkpoint averaging tracker (uses _pai.pt files and fresh model creation)
    checkpoint_tracker = Top3CheckpointTracker(save_name_with_timestamp, args.model, num_classes)
    last_mode = None

    while True:
        epoch += 1
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(
            model,
            criterion,
            optimizer,
            data_loader,
            device,
            epoch,
            args,
            model_ema,
            scaler,
        )

        model, acc1, restructured, trainingComplete = evaluate(
            model, criterion, data_loader_test, device=device
        )

        # Get training accuracy from PAI tracker extra scores
        train_acc1 = GPA.pai_tracker.member_vars.get("extra_scores", {}).get(
            "Train Acc 1", 0
        )
        
        # Get current mode
        current_mode = GPA.pai_tracker.member_vars.get("mode", "n")

        # Update max values
        if acc1 > max_val_acc1:
            max_val_acc1 = acc1
            max_train_acc1 = train_acc1
            max_params = UPA.count_params(model)

        # Log to wandb
        if run is not None:
            run.log(
                {
                    "ValAcc": acc1,
                    "TrainAcc": train_acc1,
                    "Param Count": UPA.count_params(model),
                    "Dendrite Count": GPA.pai_tracker.member_vars.get(
                        "num_dendrites_added", 0
                    ),
                    "epoch": epoch,
                    "mode": current_mode,
                }
            )

            # Log architecture maximums when dendrites are added
            if restructured:
                if current_mode == "n" and (
                    dendrite_count
                    != GPA.pai_tracker.member_vars.get("num_dendrites_added", 0)
                ):
                    dendrite_count = GPA.pai_tracker.member_vars.get(
                        "num_dendrites_added", 0
                    )
                    run.log(
                        {
                            "Arch Max Val": max_val_acc1,
                            "Arch Max Train": max_train_acc1,
                            "Arch Param Count": max_params,
                            "Arch Dendrite Count": dendrite_count - 1,
                        }
                    )

        # Track top 3 checkpoints during 'n' mode (PAI handles actual saving)
        if current_mode == "n":
            checkpoint_tracker.update(acc1, epoch, model_without_ddp, current_mode)
            
            # Print current top 3
            top3_info = checkpoint_tracker.get_top3_info()
            if top3_info:
                print(f"\n=== Top 3 checkpoints in 'n' mode ===")
                for i, (acc, ep) in enumerate(top3_info, 1):
                    print(f"  {i}. Epoch {ep}: Acc@1 {acc:.3f}")
                print("=" * 40 + "\n")
        
        # Check for mode transition from 'n' to 'p'
        if last_mode == "n" and current_mode == "p":
            print("\n" + "="*60)
            print("MODE TRANSITION: 'n' → 'p' detected!")
            print("Performing checkpoint averaging with PAI's checkpoint system...")
            print("="*60 + "\n")
            
            # Average the top 3 checkpoints and load into model
            model_without_ddp = checkpoint_tracker.average_and_load(model_without_ddp, device)
            if args.distributed:
                model.module = model_without_ddp
            
            # Log to wandb
            if run is not None:
                run.log({
                    "checkpoint_averaging": 1,
                    "averaged_at_epoch": epoch,
                })
            
            # Clear the tracker for the next cycle
            checkpoint_tracker.clear()
        
        last_mode = current_mode

        # If model was restructured by PerforatedAI, reset optimizer and scheduler
        if restructured:
            model.to(device)
            optimizer, lr_scheduler = create_optimizer_and_scheduler(
                model, args, custom_keys_weight_decay, epoch=epoch
            )

        if model_ema:
            evaluate(
                model_ema, criterion, data_loader_test, device=device, log_suffix="EMA"
            )

        # Check if PerforatedAI training is complete
        if trainingComplete:
            print("PerforatedAI training complete!")

            # Log final architecture max
            if run is not None:
                run.log(
                    {
                        "Final Max Val": max_val_acc1,
                        "Final Max Train": max_train_acc1,
                        "Final Param Count": max_params,
                        "Final Dendrite Count": GPA.pai_tracker.member_vars.get(
                            "num_dendrites_added", 0
                        ),
                    }
                )
            break
    
    print("Final Param Count:", UPA.count_params(model))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="PyTorch MobileNetV3 Training with PerforatedAI on ImageNet",
        add_help=add_help,
    )

    parser.add_argument(
        "--data-path",
        default="/home/rbrenner/Datasets/imagenet",
        type=str,
        help="dataset path",
    )
    parser.add_argument("--model", default="mobilenet_v3_small", type=str, help="model name")
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=128,
        type=int,
        help="images per gpu; default matches original 8-GPU recipe per-GPU batch size",
    )
    parser.add_argument(
        "--batch-lr-factor",
        default=1.0,
        type=float,
        help="factor to scale batch size and learning rate (e.g., 0.5 halves batch size and scales lr accordingly)",
    )
    parser.add_argument(
        "--epochs",
        default=600,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=16,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 16)",
    )
    parser.add_argument("--opt", default="rmsprop", type=str, help="optimizer (default: rmsprop for MobileNetV3)")
    parser.add_argument(
        "--lr",
        default=0.008,
        type=float,
        help="initial learning rate; default is 0.064/8=0.008, matching the original 8-GPU LR scaled for 1 GPU",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-5,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-5 for MobileNetV3)",
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
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing (default: 0.0)",
        dest="label_smoothing",
    )
    parser.add_argument(
        "--mixup-alpha", default=0.0, type=float, help="mixup alpha (default: 0.0)"
    )
    parser.add_argument(
        "--cutmix-alpha", default=0.0, type=float, help="cutmix alpha (default: 0.0)"
    )
    parser.add_argument(
        "--lr-scheduler",
        default="steplr",
        type=str,
        help="the lr scheduler (default: steplr for MobileNetV3)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=0,
        type=int,
        help="the number of epochs to warmup (default: 0)",
    )
    parser.add_argument(
        "--lr-warmup-method",
        default="constant",
        type=str,
        help="the warmup method (default: constant)",
    )
    parser.add_argument(
        "--lr-warmup-decay", default=0.01, type=float, help="the decay for lr"
    )
    parser.add_argument(
        "--lr-step-size",
        default=2,
        type=int,
        help="decrease lr every step-size epochs (default: 2 for MobileNetV3)",
    )
    parser.add_argument(
        "--lr-gamma",
        default=0.973,
        type=float,
        help="decrease lr by a factor of lr-gamma (default: 0.973 for MobileNetV3)",
    )
    parser.add_argument(
        "--lr-min",
        default=0.0,
        type=float,
        help="minimum lr of lr schedule (default: 0.0)",
    )
    parser.add_argument("--print-freq", default=500, type=int, help="print frequency")
    parser.add_argument(
        "--output-dir", default=None, type=str, help="path to save outputs"
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")
    parser.add_argument(
        "--start-epoch", default=0, type=int, metavar="N", help="start epoch"
    )
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
    parser.add_argument(
        "--auto-augment",
        default="imagenet",
        type=lambda x: None if x == "None" else x,
        help="auto augment policy (default: imagenet for MobileNetV3)",
    )
    parser.add_argument(
        "--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy"
    )
    parser.add_argument(
        "--augmix-severity", default=3, type=int, help="severity of augmix policy"
    )
    parser.add_argument(
        "--random-erase",
        default=0.2,
        type=float,
        help="random erasing probability (default: 0.2 for MobileNetV3)",
    )

    # Regularization parameters to reduce overfitting (train-val gap)
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="dropout rate (default: 0.0, no dropout)",
    )
    parser.add_argument(
        "--width-multiplier",
        default=1.0,
        type=float,
        help="network width multiplier to reduce capacity (default: 1.0, full width)",
    )
    parser.add_argument(
        "--depth-multiplier",
        default=1.0,
        type=float,
        help="network depth multiplier to reduce capacity (default: 1.0, full depth)",
    )
    parser.add_argument(
        "--stochastic-depth-prob",
        default=0.0,
        type=float,
        help="stochastic depth drop probability for ResNet (default: 0.0, no stochastic depth)",
    )

    # Mixed precision training parameters
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use torch.cuda.amp for mixed precision training",
    )

    # distributed training parameters
    parser.add_argument(
        "--world-size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist-url",
        default="env://",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--model-ema",
        action="store_true",
        help="enable tracking Exponential Moving Average of model parameters",
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
        "--use-deterministic-algorithms",
        action="store_true",
        help="Forces the use of deterministic algorithms only.",
    )
    parser.add_argument(
        "--interpolation",
        default="bilinear",
        type=str,
        help="the interpolation method (default: bilinear)",
    )
    # Standard ImageNet resolution for MobileNetV3
    parser.add_argument(
        "--val-resize-size",
        default=256,
        type=int,
        help="the resize size used for validation (default: 256)",
    )
    parser.add_argument(
        "--val-crop-size",
        default=224,
        type=int,
        help="the central crop size used for validation (default: 224)",
    )
    parser.add_argument(
        "--train-crop-size",
        default=224,
        type=int,
        help="the random crop size used for training (default: 224)",
    )
    parser.add_argument(
        "--convert-count", default=0, type=int, help="total number of layers to convert"
    )
    parser.add_argument(
        "--clip-grad-norm",
        default=None,
        type=float,
        help="the maximum gradient norm (default None)",
    )
    parser.add_argument(
        "--ra-sampler",
        action="store_true",
        help="whether to use Repeated Augmentation in training",
    )
    parser.add_argument(
        "--ra-reps",
        default=3,
        type=int,
        help="number of repetitions for Repeated Augmentation (default: 3)",
    )
    parser.add_argument(
        "--weights", default=None, type=str, help="the weights enum name to load"
    )
    parser.add_argument(
        "--backend",
        default="PIL",
        type=str.lower,
        help="PIL or tensor - case insensitive",
    )
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument(
        "--full-dataset",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use full ImageNet-1000 instead of ImageNet-100 subset (default: True)",
    )

    # PerforatedAI parameters
    parser.add_argument(
        "--improvement-threshold",
        default=1,
        type=int,
        choices=[0, 1, 2],
        help="PAI improvement threshold mode: 0=[0.01,0.001,0.0001,0], 1=[0.001,0.0001,0], 2=[0]",
    )
    parser.add_argument(
        "--candidate-weight-init-mult",
        default=0.1,
        type=float,
        help="PAI candidate weight initialization multiplier (default: 0.1)",
    )
    parser.add_argument(
        "--pai-forward-function",
        default="relu",
        type=str,
        choices=["sigmoid", "relu", "tanh", "identity", "hardswish"],
        help="PAI forward function (default: relu)",
    )
    parser.add_argument(
        "--dendrite-mode",
        default=2,
        type=int,
        choices=[0, 1, 2],
        help="Dendrite mode: 0=no dendrites, 1=GD dendrites, 2=PB dendrites (default: 2)",
    )
    parser.add_argument(
        "--perforated-load-path",
        default="",
        type=str,
        help="Path to load PerforatedAI checkpoint from (default: '', initialize new)",
    )
    parser.add_argument(
        "--dendrite-lr-multiplier",
        default=1.0,
        type=float,
        dest="dendrite_lr_multiplier",
        help="Multiply LR by this factor when mode=n and num_cycles>1 (default: 1.0, no change)",
    )
    parser.add_argument(
        "--load-checkpoint-name",
        default="latest",
        type=str,
        dest="load_checkpoint_name",
        help="Checkpoint name to load when resuming (default: 'latest'; e.g. 'switch_2', 'best_model', 'beforeSwitch_1')",
    )

    # Wandb logging
    parser.add_argument("--use-wandb", action="store_true", help="Enable wandb logging")

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
