"""
Training script for ResNet models with WandB sweep support.

Supports multiple model variants:
- resnet-18-perforated-cascor-pretrained: Perforated model from HuggingFace
- resnet-18-perforated-cascor-fc: ResNet-18 with pretrained ImageNet weights
- resnet-18-perforated-cascor-pre-fc: ResNet-18 with pretrained ImageNet weights
- resnet-34: ResNet-34 with pretrained ImageNet weights

Usage:
python train_from_hf_wandb_sweep.py --model resnet-18-perforated-cascor-pretrained --dataset flowers102
python train_from_hf_wandb_sweep.py --model resnet-34 --dataset flowers102
"""

import datetime
import os
import sys
import time
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torch import nn
from torchvision.transforms.functional import InterpolationMode

import wandb

# Add parent directory to path for resnet_double import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "imagenet"))
from resnet_double import ResNetPAI

# Import GPA for global configuration
from perforatedai import globals_perforatedai as GPA

# TODO: Remove later and do not duplicate if you are making a new sweep
# Allows loading old checkpoints that are missing new fields like nodes_improved_any
GPA.pc.set_strict_loading(False)


def get_dataset_config(dataset_name):
    """Get recommended hyperparameters for each dataset

    NOTE: Smaller datasets (flowers102, pets, food101) are designed for
    transfer learning with pretrained ImageNet weights.
    """
    configs = {
        "flowers102": {
            "num_classes": 102,
            "image_size": 224,
            "epochs": 200,
            "batch_size": 32,
            "lr": 0.001,  # Lower LR for fine-tuning
            "lr_scheduler": "cosineannealinglr",
            "weight_decay": 1e-4,
            "lr_warmup_epochs": 5,
            "label_smoothing": 0.1,
            "use_pretrained": True,  # Use pretrained weights
        },
        "pets": {
            "num_classes": 37,
            "image_size": 224,
            "epochs": 50,
            "batch_size": 32,
            "lr": 0.001,  # Lower LR for fine-tuning
            "lr_scheduler": "cosineannealinglr",
            "weight_decay": 1e-4,
            "lr_warmup_epochs": 5,
            "label_smoothing": 0.0,
            "use_pretrained": True,  # Use pretrained weights
        },
        "food101": {
            "num_classes": 101,
            "image_size": 224,
            "epochs": 30,
            "batch_size": 64,
            "lr": 0.001,  # Lower LR for fine-tuning
            "lr_scheduler": "cosineannealinglr",
            "weight_decay": 1e-4,
            "lr_warmup_epochs": 5,
            "label_smoothing": 0.0,
            "use_pretrained": True,  # Use pretrained weights
        },
    }
    return configs.get(dataset_name.lower(), configs["flowers102"])


def train_one_epoch(
    model, criterion, optimizer, data_loader, device, epoch, print_freq=10
):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        output = model(image)
        if hasattr(output, "logits"):
            output = output.logits
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

    # Return training metrics
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            if hasattr(output, "logits"):
                output = output.logits
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def measure_inference_latency(model, data_loader, device, warmup_batches=10):
    """Measure inference latency and throughput (FPS) of the model."""
    model.eval()

    print("\n" + "=" * 80)
    print("MEASURING INFERENCE LATENCY")
    print("=" * 80)

    batch_times = []
    total_images = 0

    with torch.inference_mode():
        # Warmup phase
        print(f"Warmup: Running {warmup_batches} batches...")
        for i, (image, _) in enumerate(data_loader):
            if i >= warmup_batches:
                break
            image = image.to(device, non_blocking=True)
            output = model(image)
            if hasattr(output, "logits"):
                output = output.logits
            if device.type == "cuda":
                torch.cuda.synchronize()

        # Timing phase
        print("Measuring latency...")
        for i, (image, _) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.time()
            output = model(image)
            if hasattr(output, "logits"):
                output = output.logits

            if device.type == "cuda":
                torch.cuda.synchronize()

            end_time = time.time()

            batch_time = end_time - start_time
            batch_times.append(batch_time)
            total_images += image.shape[0]

    # Calculate statistics
    total_time = sum(batch_times)
    mean_batch_time = total_time / len(batch_times)
    fps = total_images / total_time
    mean_latency_ms = mean_batch_time * 1000

    # Calculate percentiles
    sorted_times = sorted(batch_times)
    p50_ms = sorted_times[len(sorted_times) // 2] * 1000
    p95_ms = sorted_times[int(len(sorted_times) * 0.95)] * 1000
    p99_ms = sorted_times[int(len(sorted_times) * 0.99)] * 1000

    results = {
        "fps": fps,
        "mean_latency_ms": mean_latency_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "p99_latency_ms": p99_ms,
        "total_images": total_images,
        "total_batches": len(batch_times),
        "total_time_s": total_time,
    }

    print(f"\nLatency Results:")
    print(f"  Total images processed: {total_images}")
    print(f"  Total batches: {len(batch_times)}")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {fps:.2f} FPS")
    print(f"  Mean latency per batch: {mean_latency_ms:.2f}ms")
    print(f"  P50 latency: {p50_ms:.2f}ms")
    print(f"  P95 latency: {p95_ms:.2f}ms")
    print(f"  P99 latency: {p99_ms:.2f}ms")
    print("=" * 80 + "\n")

    return results


def load_dataset(dataset_name, data_path, batch_size, workers):
    """Load dataset with standard preprocessing."""
    print(f"Loading {dataset_name} dataset from {data_path}")

    # Dataset-specific configurations
    dataset_configs = {
        "flowers102": {
            "num_classes": 102,
            "img_size": 224,
            "train_split": "train",
            "test_split": "test",
            "dataset_class": torchvision.datasets.Flowers102,
        },
        "pets": {
            "num_classes": 37,
            "img_size": 224,
            "train_split": "trainval",
            "test_split": "test",
            "dataset_class": torchvision.datasets.OxfordIIITPet,
        },
        "food101": {
            "num_classes": 101,
            "img_size": 224,
            "train_split": "train",
            "test_split": "test",
            "dataset_class": torchvision.datasets.Food101,
        },
        "cifar100": {
            "num_classes": 100,
            "img_size": 32,
            "train_split": True,  # CIFAR uses True/False
            "test_split": False,
            "dataset_class": torchvision.datasets.CIFAR100,
        },
        "stl10": {
            "num_classes": 10,
            "img_size": 96,
            "train_split": "train",
            "test_split": "test",
            "dataset_class": torchvision.datasets.STL10,
        },
    }

    if dataset_name not in dataset_configs:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. Supported: {list(dataset_configs.keys())}"
        )

    config = dataset_configs[dataset_name]
    img_size = config["img_size"]
    interpolation = InterpolationMode.BILINEAR

    # Training transforms
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomResizedCrop(
                img_size, interpolation=interpolation
            ),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Validation transforms
    val_resize_size = img_size if img_size <= 32 else int(img_size * 256 / 224)
    val_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(val_resize_size, interpolation=interpolation),
            torchvision.transforms.CenterCrop(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # Load datasets based on type
    if dataset_name == "cifar100":
        dataset_train = config["dataset_class"](
            root=data_path,
            train=config["train_split"],
            download=True,
            transform=train_transform,
        )
        dataset_test = config["dataset_class"](
            root=data_path,
            train=config["test_split"],
            download=True,
            transform=val_transform,
        )
    elif dataset_name == "stl10":
        dataset_train = config["dataset_class"](
            root=data_path,
            split=config["train_split"],
            download=True,
            transform=train_transform,
        )
        dataset_test = config["dataset_class"](
            root=data_path,
            split=config["test_split"],
            download=True,
            transform=val_transform,
        )
    else:
        dataset_train = config["dataset_class"](
            root=data_path,
            split=config["train_split"],
            download=True,
            transform=train_transform,
        )
        dataset_test = config["dataset_class"](
            root=data_path,
            split=config["test_split"],
            download=True,
            transform=val_transform,
        )

    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Test dataset size: {len(dataset_test)}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    return train_loader, test_loader, config["num_classes"]


def load_model(model_name, num_classes, perforate=False):
    """Load model based on model name and adapt for target number of classes."""
    print(f"\nLoading model: {model_name}")

    # Import PAI if perforating
    if perforate:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA

    if model_name == "resnet-18-perforated-cascor-pretrained":
        # Load perforated model from HuggingFace
        from perforatedai import utils_perforatedai as UPA
        from perforatedai import library_perforatedai as LPA

        hf_repo_id = "perforated-ai/resnet-18-perforated-cascor"
        # Create base model architecture
        base_model = torchvision.models.get_model(
            "resnet18", weights=None, num_classes=1000
        )
        model = LPA.ResNetPAIPreFC(base_model)
        # Load from HuggingFace (always download latest version)
        model = UPA.from_hf_pretrained(model, hf_repo_id, force_download=True)
        print(f"Successfully loaded perforated model from HuggingFace: {hf_repo_id}")

    elif model_name == "resnet-18-perforated-cascor-fc":
        # Load torchvision ResNet-18 with pretrained ImageNet weights
        model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        print(
            f"Successfully loaded torchvision ResNet-18 with pretrained ImageNet weights"
        )

        # Perforate only the fc layer if requested
        if perforate:
            print("Configuring PAI to perforate only the fc layer...")
            GPA.pc.set_testing_dendrite_capacity(False)  # Full training mode
            GPA.pc.set_max_dendrites(3)
            GPA.pc.set_max_dendrite_tries(1)
            GPA.pc.set_initial_correlation_batches(10)
            GPA.pc.set_cap_at_n(True)  # Cap dendrite epochs to neuron epochs
            GPA.pc.set_module_names_to_perforate(["Linear"])
            GPA.pc.set_module_ids_to_track(
                [".layer1", ".layer2", ".layer3", ".layer4", ".conv1", ".bn1"]
            )  # Skip everything except fc
            GPA.pc.set_output_dimensions([-1, 0])  # fc layer output: [batch, features]

            # Apply dendritic hyperparameters from wandb config if in sweep
            if (
                hasattr(wandb, "run")
                and wandb.run is not None
                and hasattr(wandb, "config")
            ):
                if "improvement_threshold" in wandb.config:
                    GPA.pc.set_improvement_threshold(wandb.config.improvement_threshold)
                if "pai_forward_function" in wandb.config:
                    pai_fwd = wandb.config.pai_forward_function
                    if pai_fwd == "sigmoid":
                        GPA.pc.set_pai_forward_function(torch.sigmoid)
                    elif pai_fwd == "relu":
                        GPA.pc.set_pai_forward_function(torch.relu)
                    elif pai_fwd == "tanh":
                        GPA.pc.set_pai_forward_function(torch.tanh)

            # Build save_name from wandb config if available (for sweeps)
            if (
                hasattr(wandb, "run")
                and wandb.run is not None
                and hasattr(wandb.run, "name")
                and wandb.run.name
            ):
                # Use wandb run name directly to ensure consistency
                save_name = wandb.run.name
            else:
                save_name = f"resnet18_fc_{num_classes}cls"

            model = UPA.perforate_model(
                model,
                save_name=save_name,
                maximizing_score=True,
            )
            print(
                f"Model perforated successfully (fc layer only) - save_name: {save_name}"
            )

    elif model_name == "resnet-18-perforated-cascor-pre-fc":
        # Load ResNetPAI with pretrained weights from pretrained-prefc folder
        from perforatedai import utils_perforatedai as UPA

        # Create base ResNet-18 with pretrained ImageNet weights
        base_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")

        # Wrap in ResNetPAI (adds pre_fc layer)
        model = ResNetPAI(base_model)

        # Perforate only the pre_fc layer if requested
        if perforate:
            from perforatedai import globals_perforatedai as GPA

            print("Configuring PAI to perforate only the pre_fc layer...")
            GPA.pc.set_testing_dendrite_capacity(False)  # Full training mode
            GPA.pc.set_max_dendrites(3)
            GPA.pc.set_max_dendrite_tries(1)
            GPA.pc.set_initial_correlation_batches(10)
            GPA.pc.set_cap_at_n(True)  # Cap dendrite epochs to neuron epochs
            GPA.pc.set_module_names_to_perforate(["Linear"])
            GPA.pc.set_module_ids_to_track(
                [".layer1", ".layer2", ".layer3", ".layer4", ".conv1", ".bn1", ".fc"]
            )  # Skip everything except pre_fc
            GPA.pc.set_output_dimensions(
                [-1, 0]
            )  # pre_fc layer output: [batch, features]

            # Apply dendritic hyperparameters from wandb config if in sweep
            if (
                hasattr(wandb, "run")
                and wandb.run is not None
                and hasattr(wandb, "config")
            ):
                if "improvement_threshold" in wandb.config:
                    GPA.pc.set_improvement_threshold(wandb.config.improvement_threshold)
                if "pai_forward_function" in wandb.config:
                    pai_fwd = wandb.config.pai_forward_function
                    if pai_fwd == "sigmoid":
                        GPA.pc.set_pai_forward_function(torch.sigmoid)
                    elif pai_fwd == "relu":
                        GPA.pc.set_pai_forward_function(torch.relu)
                    elif pai_fwd == "tanh":
                        GPA.pc.set_pai_forward_function(torch.tanh)

            # Build save_name from wandb config if available (for sweeps)
            if (
                hasattr(wandb, "run")
                and wandb.run is not None
                and hasattr(wandb.run, "name")
                and wandb.run.name
            ):
                # Use wandb run name directly to ensure consistency
                save_name = wandb.run.name
            else:
                save_name = f"resnet18_prefc_{num_classes}cls"

            model = UPA.perforate_model(
                model,
                save_name=save_name,
                maximizing_score=True,
            )
            print(
                f"Model perforated successfully (pre_fc layer only) - save_name: {save_name}"
            )

        # Load pretrained pre-fc weights from local folder (if exists)
        pretrained_folder = os.path.join(os.path.dirname(__file__), "pretrained-prefc")
        if os.path.exists(pretrained_folder):
            model = UPA.load_system(model, pretrained_folder, "beforeSwitch_0")
            print(
                f"Successfully loaded ResNetPAI with pretrained weights from {pretrained_folder}"
            )
        else:
            print(
                f"Pretrained folder {pretrained_folder} not found, using base ImageNet weights"
            )

    elif model_name == "resnet-34":
        # Load torchvision ResNet-34 with pretrained ImageNet weights
        model = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        print(
            f"Successfully loaded torchvision ResNet-34 with pretrained ImageNet weights"
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Replace final layer for target number of classes
    if hasattr(model, "fc"):
        # Check if it's a TrackedNeuronModule (from HuggingFace PAI model) or regular Linear
        if hasattr(model.fc, "main_module"):
            in_features = model.fc.main_module.in_features
        else:
            in_features = model.fc.in_features
        new_fc = nn.Linear(in_features, num_classes)
        
        # If perforate is True, wrap the new fc layer in appropriate PAI module
        if perforate:
            from perforatedai.modules_perforatedai import PAINeuronModule, TrackedNeuronModule
            
            if model_name == "resnet-18-perforated-cascor-fc":
                # For fc model: perforate the fc layer
                model.fc = PAINeuronModule(new_fc, "fc")
                print(f"Replaced fc layer for {num_classes} classes and converted to PAINeuronModule")
            elif model_name == "resnet-18-perforated-cascor-pre-fc":
                # For pre-fc model: track the fc layer (don't perforate it)
                model.fc = TrackedNeuronModule(new_fc, "fc")
                print(f"Replaced fc layer for {num_classes} classes and converted to TrackedNeuronModule")
        else:
            model.fc = new_fc
            print(f"Replaced fc layer for {num_classes} classes")
    elif hasattr(model, "classifier"):
        # Transformers models use 'classifier'
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        print(f"Replaced classifier layer for {num_classes} classes")
    else:
        raise ValueError(f"Cannot adapt model - unknown classifier layer")

    return model


def train_single_run(args, train_loader, test_loader, num_classes):
    """Perform a single training run and return best accuracy and epoch."""
    device = torch.device(args.device)

    # Determine if we should perforate this model
    perforate = args.model in [
        "resnet-18-perforated-cascor-fc",
        "resnet-18-perforated-cascor-pre-fc",
    ]

    # Load model
    model = load_model(args.model, num_classes, perforate=perforate)
    model = model.to(device)

    # Count parameters (log once at start)
    from perforatedai import utils_perforatedai as UPA
    param_count = UPA.count_params(model)
    print(f"Model parameters: {param_count:,}")

    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # Setup optimizer
    if perforate:
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import utils_perforatedai as UPA
        
        GPA.pai_tracker.set_optimizer(torch.optim.SGD)
        optimArgs = {
            "params": model.parameters(),
            "lr": args.lr,
            "momentum": 0.9,
            "weight_decay": args.weight_decay,
        }
        
        if args.lr_warmup_epochs > 0:
            # SequentialLR case - describe component schedulers in schedArgs
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.SequentialLR)
            schedArgs = {
                "schedulers": [
                    (torch.optim.lr_scheduler.ConstantLR, {"factor": 0.01, "total_iters": args.lr_warmup_epochs}),
                    (torch.optim.lr_scheduler.CosineAnnealingLR, {"T_max": max(1, args.epochs - args.lr_warmup_epochs), "eta_min": 0.0})
                ],
                "milestones": [args.lr_warmup_epochs]
            }
            optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
        else:
            # Simple scheduler case
            GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.CosineAnnealingLR)
            schedArgs = {
                "T_max": max(1, args.epochs - args.lr_warmup_epochs),
                "eta_min": 0.0
            }
            optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
        # Setup scheduler for non-perforated
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs - args.lr_warmup_epochs), eta_min=0.0
        )
        if args.lr_warmup_epochs > 0:
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=0.01, total_iters=args.lr_warmup_epochs
            )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_lr_scheduler, main_lr_scheduler],
                milestones=[args.lr_warmup_epochs],
            )
        else:
            lr_scheduler = main_lr_scheduler

    # Determine dendrite count for non-perforated models
    dendrite_count_override = None
    if args.model == "resnet-18-perforated-cascor-pretrained":
        dendrite_count_override = 2
    elif args.model == "resnet-34":
        dendrite_count_override = 0

    # Training loop
    print("\nStarting training...")
    start_time = time.time()
    best_acc1 = 0.0
    best_epoch = 0

    # For perforated models: arch logging variables
    if perforate:
        last_logged_integrated = -1  # Track last num_dendrites_integrated we logged (-1 = nothing logged yet)
        max_val = 0
        max_train = 0
        max_params = 0
        global_max_val = 0
        global_max_train = 0
        global_max_params = 0

    # Use while True for perforated models, for loop for non-perforated
    if perforate:
        epoch = -1  # Will increment to 0 at start of loop
        while True:
            epoch += 1
            train_acc1, train_loss = train_one_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                device,
                epoch,
                args.print_freq,
            )
            test_acc1, test_loss = evaluate(model, criterion, test_loader, device)

            # Track best accuracy for this architecture (only during neuron training, not dendrite training)
            current_mode = GPA.pai_tracker.member_vars.get("mode", "n")
            if current_mode == "n" and test_acc1 > max_val:
                max_val = test_acc1
                max_train = train_acc1
                max_params = UPA.count_params(model)

            # Track global best (only during neuron training)
            if current_mode == "n" and test_acc1 > global_max_val:
                global_max_val = test_acc1
                global_max_train = train_acc1
                global_max_params = UPA.count_params(model)

            # Track best for return (only n mode for perforated, since d mode improvements won't be saved)
            if current_mode == "n" and test_acc1 > best_acc1:
                best_acc1 = test_acc1
                best_epoch = epoch + 1

            # Add extra scores for training metrics (must be after add_validation_score)
            GPA.pai_tracker.add_extra_score(train_acc1, "train")
            GPA.pc.set_verbose(True)
            # Add validation score to PAI tracker
            model, restructured, training_complete = (
                GPA.pai_tracker.add_validation_score(test_acc1, model)
            )
            GPA.pc.set_verbose(False)
            model = model.to(device)

            # Log arch max when dendrites are successfully integrated (or first switch for base model)
            if restructured and not training_complete:
                current_integrated = GPA.pai_tracker.member_vars["num_dendrites_integrated"]
                # Log if integrated increased (starting from -1 handles base model case)
                if current_integrated > last_logged_integrated:
                    if hasattr(wandb, "run") and wandb.run is not None:
                        # Debug prints
                        print(f"DEBUG: Logging arch scores for dendrite count {current_integrated}")
                        print(f"  Params: {UPA.count_params(model)}, Integrated: {current_integrated}")
                        print(f"  Max val: {max_val}, Max train: {max_train}")
                        
                        # Log with architecture-specific metric names to prevent overwriting
                        wandb.log(
                            {
                                f"Arch_{current_integrated}_Max_Val": max_val,
                                f"Arch_{current_integrated}_Max_Train": max_train,
                                f"Arch_{current_integrated}_Param_Count": max_params,
                                # Also log generic metrics for backward compatibility
                                "Arch Max Val": max_val,
                                "Arch Max Train": max_train,
                                "Arch Param Count": max_params,
                                "Arch Dendrite Count": current_integrated,
                            }
                        )
                    last_logged_integrated = current_integrated
                    max_val = 0  # Reset for next arch
                    max_train = 0
                    max_params = 0
                else:
                    # Debug: why didn't we log?
                    print(f"DEBUG: NOT logging arch scores")
                    print(f"  current_integrated: {current_integrated}, last_logged_integrated: {last_logged_integrated}")
                    print(f"  num_dendrites_added: {GPA.pai_tracker.member_vars['num_dendrites_added']}")
                    print(f"  mode: {GPA.pai_tracker.member_vars['mode']}")
                    print(f"  current_n_set_global_best: {GPA.pai_tracker.member_vars['current_n_set_global_best']}")
                    print(f"  Max val: {max_val}, Max train: {max_train}")

                # Reinitialize optimizer and scheduler after restructuring (same as initial setup)
                optimArgs = {
                    "params": model.parameters(),
                    "lr": args.lr,
                    "momentum": 0.9,
                    "weight_decay": args.weight_decay,
                }
                
                # Recreate scheduler (same as initial setup)
                main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=max(1, args.epochs - args.lr_warmup_epochs), eta_min=0.0
                )
                if args.lr_warmup_epochs > 0:
                    warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer, factor=0.01, total_iters=args.lr_warmup_epochs
                    )
                    schedArgs = {
                        "schedulers": [warmup_lr_scheduler, main_lr_scheduler],
                        "milestones": [args.lr_warmup_epochs]
                    }
                    optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
                else:
                    schedArgs = {
                        "T_max": max(1, args.epochs - args.lr_warmup_epochs),
                        "eta_min": 0.0
                    }
                    optimizer, lr_scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
            else:
                # Debug: why didn't restructure trigger?
                if not restructured:
                    print(f"DEBUG: No restructure - restructured={restructured}")
                elif training_complete:
                    print(f"DEBUG: Training complete - handling in separate block")

            # Log to WandB - using wandb.md recommended naming
            if hasattr(wandb, "run") and wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "TrainAcc": train_acc1,
                        "TrainLoss": train_loss,
                        "ValAcc": test_acc1,
                        "ValLoss": test_loss,
                        "TestAcc": test_acc1,
                        "Param Count": UPA.count_params(model),
                        "Dendrite Count": GPA.pai_tracker.member_vars[
                            "num_dendrites_added"
                        ],
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            print(
                f"Epoch {epoch+1} - Train Acc@1: {train_acc1:.3f}, Test Acc@1: {test_acc1:.3f}, "
                f"Dendrites added: {GPA.pai_tracker.member_vars['num_dendrites_added']}, "
                f"Integrated: {GPA.pai_tracker.member_vars['num_dendrites_integrated']}"
            )

            if training_complete:
                print("PAI training complete!")
                # Log final arch max and final max
                if hasattr(wandb, "run") and wandb.run is not None:
                    # Log final arch if integrated count increased (max dendrites hit with successful last dendrite)
                    current_integrated = GPA.pai_tracker.member_vars["num_dendrites_integrated"]
                    if current_integrated > last_logged_integrated:
                        # Debug prints
                        print(f"DEBUG: Logging final arch scores for dendrite count {current_integrated}")
                        print(f"  Params: {UPA.count_params(model)}, Integrated: {current_integrated}")
                        print(f"  Max val: {max_val}, Max train: {max_train}")
                        
                        # Log with architecture-specific metric names to prevent overwriting
                        wandb.log(
                            {
                                f"Arch_{current_integrated}_Max_Val": max_val,
                                f"Arch_{current_integrated}_Max_Train": max_train,
                                f"Arch_{current_integrated}_Param_Count": max_params,
                                # Also log generic metrics for backward compatibility
                                "Arch Max Val": max_val,
                                "Arch Max Train": max_train,
                                "Arch Param Count": max_params,
                                "Arch Dendrite Count": current_integrated,
                            }
                        )
                    else:
                        # Debug: why didn't we log final arch?
                        print(f"DEBUG: NOT logging final arch scores")
                        print(f"  current_integrated: {current_integrated}, last_logged_integrated: {last_logged_integrated}")
                        print(f"  num_dendrites_added: {GPA.pai_tracker.member_vars['num_dendrites_added']}")
                        print(f"  Max val: {max_val}, Max train: {max_train}")
                    
                    # Always log Final Max scores
                    wandb.log(
                        {
                            "Final Max Val": global_max_val,
                            "Final Max Train": global_max_train,
                            "Final Param Count": global_max_params,
                            "Final Dendrite Count": GPA.pai_tracker.member_vars[
                                "num_dendrites_integrated"
                            ],
                        }
                    )
                break
    else:
        # Non-perforated training loop
        best_train = 0.0  # Track training accuracy when best val is achieved

        for epoch in range(args.epochs):
            train_acc1, train_loss = train_one_epoch(
                model,
                criterion,
                optimizer,
                train_loader,
                device,
                epoch,
                args.print_freq,
            )
            lr_scheduler.step()
            test_acc1, test_loss = evaluate(model, criterion, test_loader, device)

            # Track best accuracy and corresponding training accuracy
            if test_acc1 > best_acc1:
                best_acc1 = test_acc1
                best_train = train_acc1
                best_epoch = epoch + 1

            # Log to WandB if initialized - using wandb.md recommended naming
            if hasattr(wandb, "run") and wandb.run is not None:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "TrainAcc": train_acc1,
                        "TrainLoss": train_loss,
                        "ValAcc": test_acc1,
                        "ValLoss": test_loss,
                        "TestAcc": test_acc1,  # For transfer learning, val=test
                        "Param Count": param_count,
                        "Dendrite Count": (
                            dendrite_count_override
                            if dendrite_count_override is not None
                            else 0
                        ),
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

            print(
                f"Epoch {epoch+1}/{args.epochs} - Train Acc@1: {train_acc1:.3f}, Test Acc@1: {test_acc1:.3f}, Loss: {test_loss:.4f}"
            )

        # Log final results for non-perforated models
        if hasattr(wandb, "run") and wandb.run is not None:
            # For non-perforated models, Arch scores = Final scores (no restructuring)
            final_dendrite_count = (
                dendrite_count_override if dendrite_count_override is not None else 0
            )

            # Log Arch Max scores
            wandb.log(
                {
                    "Arch Max Val": best_acc1,
                    "Arch Max Train": best_train,
                    "Arch Param Count": param_count,
                    "Arch Dendrite Count": final_dendrite_count,
                }
            )
            # Log Final Max scores
            wandb.log(
                {
                    "Final Max Val": best_acc1,
                    "Final Max Train": best_train,
                    "Final Param Count": param_count,
                    "Final Dendrite Count": final_dendrite_count,
                }
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\nTraining complete! Total time: {total_time_str}")
    print(f"Best Test Accuracy: {best_acc1:.3f}% (achieved at epoch {best_epoch})")

    return best_acc1, best_epoch, model


def get_model_name_from_index(model_index):
    """Map model index to model name for WandB color coding.

    Args:
        model_index: Integer from 0-3

    Returns:
        Model name string
    """
    model_mapping = {
        0: "resnet-18-perforated-cascor-pretrained",
        1: "resnet-18-perforated-cascor-fc",
        2: "resnet-18-perforated-cascor-pre-fc",
        3: "resnet-34",
    }
    return model_mapping[model_index]


def get_sweep_config(dataset_name):
    """Get WandB sweep configuration for a specific dataset."""
    base_config = {
        "method": "random",  # Random sampling instead of exhaustive grid
        "metric": {"name": "Final Max Val", "goal": "maximize"},
    }

    if dataset_name == "flowers102":
        # Small dataset - use all 4 models, higher regularization
        base_config["parameters"] = {
            "dataset": {"value": "flowers102"},
            "model_index": {
                "values": [0, 1, 2, 3]
            },  # Maps to model names via get_model_name_from_index
            "lr": {"values": [0.0001, 0.0003, 0.001, 0.003]},
            "weight_decay": {"values": [1e-5, 1e-4, 1e-3]},
            "label_smoothing": {"values": [0.05, 0.1, 0.15]},
            # Dendritic hyperparameters (only used by fc/pre-fc perforated models)
            "improvement_threshold": {"values": [[0.01, 0.001, 0.0001, 0], [0]]},
            "pai_forward_function": {"values": ["sigmoid", "relu", "tanh"]},
        }

    elif dataset_name == "pets":
        # Medium dataset - all 4 models, broader LR range
        base_config["parameters"] = {
            "dataset": {"value": "pets"},
            "model_index": {
                "values": [0, 1, 2, 3]
            },  # Maps to model names via get_model_name_from_index
            "lr": {"values": [0.0003, 0.001, 0.003, 0.01]},
            "weight_decay": {"values": [0.0, 1e-5, 1e-4]},
            "label_smoothing": {"values": [0.0, 0.05, 0.1]},
            # Dendritic hyperparameters (only used by fc/pre-fc perforated models)
            "improvement_threshold": {"values": [[0.01, 0.001, 0.0001, 0], [0]]},
            "pai_forward_function": {"values": ["sigmoid", "relu", "tanh"]},
        }

    elif dataset_name == "food101":
        # Larger dataset - all 4 models, can handle higher LR
        base_config["parameters"] = {
            "dataset": {"value": "food101"},
            "model_index": {
                "values": [0, 1, 2, 3]
            },  # Maps to model names via get_model_name_from_index
            "lr": {"values": [0.001, 0.003, 0.01, 0.03]},
            "weight_decay": {"values": [0.0, 1e-5, 1e-4]},
            "label_smoothing": {"values": [0.0, 0.05]},
            # Dendritic hyperparameters (only used by fc/pre-fc perforated models)
            "improvement_threshold": {"values": [[0.01, 0.001, 0.0001, 0], [0]]},
            "pai_forward_function": {"values": ["sigmoid", "relu", "tanh"]},
        }

    else:
        raise ValueError(f"No sweep config defined for dataset: {dataset_name}")

    return base_config


def train_with_wandb():
    """Training function for WandB sweep - gets config from wandb.config."""
    # Parse base arguments (non-swept parameters)
    import argparse

    parser = argparse.ArgumentParser(description="Training run within WandB sweep")
    parser.add_argument("--data-path", default="./data", type=str, help="Dataset path")
    parser.add_argument(
        "--workers", default=16, type=int, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device (cuda or cpu)"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="Print frequency")
    args, unknown = parser.parse_known_args()  # Ignore unknown args from main script

    # Initialize wandb (project is inherited from sweep context)
    wandb.init()
    config = wandb.config

    # Override wandb sweep's silent mode (wandb.agent sets this to True)
    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA
    GPA.pc.set_silent(False)

    # Map model_index to model name
    model_name = get_model_name_from_index(config.model_index)

    # Set run name to match save_name pattern (from wandb.md recommendation)
    # Use all config keys (sorted for consistency)
    config_keys = sorted(config.keys())
    # Put any key containing 'model' first
    model_keys = [k for k in config_keys if 'model' in k.lower()]
    other_keys = [k for k in config_keys if 'model' not in k.lower()]
    config_keys = model_keys + other_keys
    name_parts = [f"{k}_{config[k]}" for k in config_keys]
    if name_parts:
        wandb.run.name = "_".join(name_parts)

    # Set args from sweep config
    args.model = model_name
    args.dataset = config.dataset
    args.lr = config.lr
    args.weight_decay = config.weight_decay
    args.label_smoothing = config.label_smoothing

    # Apply dataset-specific defaults for non-swept parameters
    dataset_config = get_dataset_config(args.dataset)
    args.batch_size = dataset_config.get("batch_size", 32)
    args.epochs = dataset_config.get("epochs", 50)
    args.lr_warmup_epochs = dataset_config.get("lr_warmup_epochs", 5)

    print(f"\n{'='*80}")
    print(f"WandB Sweep Run Configuration")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"{'='*80}\n")

    # Load dataset
    train_loader, test_loader, num_classes = load_dataset(
        args.dataset, args.data_path, args.batch_size, args.workers
    )

    # Train
    best_acc1, best_epoch, model = train_single_run(
        args, train_loader, test_loader, num_classes
    )

    # Measure inference latency
    device = torch.device(args.device)
    latency_results = measure_inference_latency(model, test_loader, device)

    # Count parameters
    param_count = UPA.count_params(model)

    # Log final results - using wandb.md recommended naming
    # Skip for perforated models since Final metrics are already logged inside training loop with dendrite count
    perforate = args.model in [
        "resnet-18-perforated-cascor-fc",
        "resnet-18-perforated-cascor-pre-fc",
    ]
    if not perforate:
        # Non-perforated models: log Final scores here
        wandb.log(
            {
                "Final Max Val": best_acc1,
                "Final Max Test": best_acc1,  # For transfer learning, val=test
                "Final Param Count": param_count,
                "best_epoch": best_epoch,
                "fps": latency_results["fps"],
                "mean_latency_ms": latency_results["mean_latency_ms"],
                "p95_latency_ms": latency_results["p95_latency_ms"],
            }
        )
    else:
        # Perforated models: just log latency (Final scores already logged)
        wandb.log(
            {
                "best_epoch": best_epoch,
                "fps": latency_results["fps"],
                "mean_latency_ms": latency_results["mean_latency_ms"],
                "p95_latency_ms": latency_results["p95_latency_ms"],
            }
        )

    print(f"\n{'='*80}")
    print(f"Run complete - Best Acc@1: {best_acc1:.3f}% at epoch {best_epoch}")
    print(f"{'='*80}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train ResNet models with WandB sweep support"
    )
    parser.add_argument(
        "--sweep-dataset",
        type=str,
        choices=["flowers102", "pets", "food101"],
        help="Initialize and run WandB sweep for this dataset",
    )
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Join an existing sweep by ID (use 'main' to initialize new sweep with --sweep-dataset)",
    )
    parser.add_argument(
        "--sweep-count",
        type=int,
        default=300,
        help="Number of runs for sweep agent (default: 300)",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "resnet-18-perforated-cascor-pretrained",
            "resnet-18-perforated-cascor-fc",
            "resnet-18-perforated-cascor-pre-fc",
            "resnet-34",
        ],
        help="Model to train (required for single runs)",
    )
    parser.add_argument(
        "--dataset",
        default="flowers102",
        type=str,
        choices=["flowers102", "pets", "food101", "cifar100", "stl10"],
        help="Dataset to train on (default: flowers102)",
    )
    parser.add_argument("--data-path", default="./data", type=str, help="Dataset path")
    parser.add_argument(
        "--batch-size",
        default=None,
        type=int,
        help="Batch size (default: dataset-specific)",
    )
    parser.add_argument(
        "--epochs",
        default=None,
        type=int,
        help="Number of epochs (default: dataset-specific)",
    )
    parser.add_argument(
        "--lr",
        default=None,
        type=float,
        help="Learning rate (default: dataset-specific)",
    )
    parser.add_argument(
        "--lr-warmup-epochs",
        default=None,
        type=int,
        help="Number of warmup epochs (default: dataset-specific)",
    )
    parser.add_argument(
        "--label-smoothing",
        default=None,
        type=float,
        help="Label smoothing (default: dataset-specific)",
    )
    parser.add_argument(
        "--weight-decay",
        default=None,
        type=float,
        help="Weight decay (default: dataset-specific)",
    )
    parser.add_argument(
        "--workers", default=16, type=int, help="Number of data loading workers"
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Device (cuda or cpu)"
    )
    parser.add_argument("--print-freq", default=10, type=int, help="Print frequency")
    parser.add_argument(
        "--use-wandb", action="store_true", help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb-project",
        default="resnet-transfer-learning",
        type=str,
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        type=str,
        help="WandB entity name (defaults to your personal account)",
    )

    args = parser.parse_args()

    # Sweep mode: Initialize new sweep or join existing sweep
    if args.sweep_dataset or args.sweep_id:
        if args.sweep_id and args.sweep_id != "main":
            # Join existing sweep
            print(f"\n{'='*80}")
            print(f"Joining existing WandB sweep: {args.sweep_id}")
            print(f"{'='*80}\n")

            wandb.agent(
                args.sweep_id,
                function=train_with_wandb,
                count=args.sweep_count,
                entity=args.wandb_entity,
                project=args.wandb_project,
            )

            print(f"\n{'='*80}")
            print(f"Sweep agent complete! View results at: https://wandb.ai")
            print(f"{'='*80}\n")
            return
        elif args.sweep_dataset:
            # Initialize new sweep
            print(f"\n{'='*80}")
            print(f"Initializing WandB sweep for {args.sweep_dataset}")
            print(f"{'='*80}\n")

            sweep_config = get_sweep_config(args.sweep_dataset)
            project_name = args.sweep_dataset  # Use dataset name as project
            sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=project_name)

            print(f"Sweep initialized: {sweep_id}")
            print(f"Project: {project_name}")
            print(f"View at: https://wandb.ai")
            print(f"\nTo join this sweep from other machines, run:")
            entity_arg = f" --wandb-entity {args.wandb_entity}" if args.wandb_entity else ""
            print(
                f"  python train_from_hf_wandb_sweep.py --sweep-id {sweep_id} --wandb-project {project_name}{entity_arg}"
            )
            print(f"\nStarting sweep agent (count={args.sweep_count})...\n")

            wandb.agent(
                sweep_id,
                function=train_with_wandb,
                count=args.sweep_count,
                entity=args.wandb_entity,
                project=project_name,
            )

            print(f"\n{'='*80}")
            print(f"Sweep complete! View results at: https://wandb.ai")
            print(f"{'='*80}\n")
            return
        else:
            parser.error("--sweep-dataset is required when using --sweep-id main")

    # Single run mode: Require --model argument
    if not args.model:
        parser.error(
            "--model is required for single training runs (or use --sweep-dataset for sweeps)"
        )

    # Single run mode continues here
    # Apply dataset-specific defaults if not explicitly set
    config = get_dataset_config(args.dataset)

    if args.batch_size is None:
        args.batch_size = config.get("batch_size", 32)
    if args.epochs is None:
        args.epochs = config.get("epochs", 50)
    if args.lr is None:
        args.lr = config.get("lr", 0.001)
    if args.lr_warmup_epochs is None:
        args.lr_warmup_epochs = config.get("lr_warmup_epochs", 5)
    if args.label_smoothing is None:
        args.label_smoothing = config.get("label_smoothing", 0.1)
    if args.weight_decay is None:
        args.weight_decay = config.get("weight_decay", 1e-4)

    # Initialize WandB if requested or if running in sweep
    wandb_active = hasattr(wandb, "run") and wandb.run is not None
    if args.use_wandb or wandb_active:
        if not wandb_active:
            wandb.init(
                entity=args.wandb_entity,
                project=args.wandb_project,
                config={
                    "model": args.model,
                    "dataset": args.dataset,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "lr_warmup_epochs": args.lr_warmup_epochs,
                    "label_smoothing": args.label_smoothing,
                    "weight_decay": args.weight_decay,
                },
            )
        # Update args from wandb config if in sweep
        if wandb.config:
            for key in ["model", "lr", "weight_decay", "label_smoothing"]:
                if key in wandb.config:
                    setattr(args, key, wandb.config[key])

    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"LR warmup epochs: {args.lr_warmup_epochs}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")

    # Load dataset
    train_loader, test_loader, num_classes = load_dataset(
        args.dataset, args.data_path, args.batch_size, args.workers
    )

    # Run single training trial
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")

    best_acc1, best_epoch, model = train_single_run(
        args, train_loader, test_loader, num_classes
    )

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE - Best Acc@1: {best_acc1:.3f}, Best Epoch: {best_epoch}")
    print(f"{'='*80}\n")

    # Measure inference latency
    device = torch.device(args.device)
    latency_results = measure_inference_latency(model, test_loader, device)

    # Log results to WandB (Final Max scores already logged inside training loop for perforated models)
    if hasattr(wandb, "run") and wandb.run is not None:
        from perforatedai import utils_perforatedai as UPA
        param_count = UPA.count_params(model)
        
        # Check if this is a perforated model that logs inside training loop
        perforate = args.model in [
            "resnet-18-perforated-cascor-fc",
            "resnet-18-perforated-cascor-pre-fc",
        ]
        
        if not perforate:
            # Non-perforated models: log Final scores here
            wandb.log(
                {
                    "Final Max Val": best_acc1,
                    "Final Max Test": best_acc1,  # For transfer learning, val=test
                    "Final Param Count": param_count,
                    "final_best_epoch": best_epoch,
                    "fps": latency_results["fps"],
                    "mean_latency_ms": latency_results["mean_latency_ms"],
                    "p95_latency_ms": latency_results["p95_latency_ms"],
                }
            )
        else:
            # Perforated models: just log latency (Final scores already logged)
            wandb.log(
                {
                    "final_best_epoch": best_epoch,
                    "fps": latency_results["fps"],
                    "mean_latency_ms": latency_results["mean_latency_ms"],
                    "p95_latency_ms": latency_results["p95_latency_ms"],
                }
            )
        wandb.finish()

    # Print final results
    print(f"\n{'='*80}")
    print(f"FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Best Acc@1: {best_acc1:.3f}%")
    print(f"Best Epoch: {best_epoch}")
    print(f"FPS: {latency_results['fps']:.2f}")
    print(f"Mean Latency: {latency_results['mean_latency_ms']:.2f}ms")
    print(f"P95 Latency: {latency_results['p95_latency_ms']:.2f}ms")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
