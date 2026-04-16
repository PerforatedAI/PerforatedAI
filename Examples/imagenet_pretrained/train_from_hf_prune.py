'''
Training script that loads models from HuggingFace and applies iterative pruning during training.

Usage:
python train_from_hf_sweep_prune.py --hf-repo-id microsoft/resnet-18 --dataset flowers102 --target-params 5000000
python train_from_hf_sweep_prune.py --hf-repo-id perforated-ai/resnet-18-perforated --dataset pets --target-params 3000000 --prune-scope local


#Current experiments
ython train_from_hf_prune.py --hf-repo-id tv/mobilenet_v3_large --dataset food101 --target-params 2500000 --model-type mobilenet --prune-method ln --prune-scope local --device cuda:0 --data-path ./data

python train_from_hf_prune.py --hf-repo-id tv/mnasnet0_75 --dataset food101 --target-params 2200000 --model-type mobilenet --prune-method ln --prune-scope local --device cuda:0 --data-path ./data &

python train_from_hf_prune.py --hf-repo-id tv/efficientnet_b1 --dataset food101 --target-params 5300000 --model-type mobilenet --prune-method ln --prune-scope local --device cuda:1 --data-path ./data &
'''

import datetime
import os
import time
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torch import nn
from torchvision.transforms.functional import InterpolationMode
import torch.nn.utils.prune as prune


def get_dataset_config(dataset_name):
    """Get recommended hyperparameters for each dataset
    
    NOTE: Smaller datasets (flowers102, pets, food101) are designed for 
    transfer learning with pretrained ImageNet weights.
    """
    configs = {
        'cifar100': {
            'num_classes': 100,
            'image_size': 32,
            'epochs': 200,
            'batch_size': 128,
            'lr': 0.1,
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 5e-4,
            'lr_warmup_epochs': 0,
            'label_smoothing': 0.1,
            'use_pretrained': False,  # Train from scratch
        },
        'stl10': {
            'num_classes': 10,
            'image_size': 96,
            'epochs': 100,
            'batch_size': 64,
            'lr': 0.05,
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 0,
            'label_smoothing': 0.0,
            'use_pretrained': False,  # Train from scratch
        },
        'flowers102': {
            'num_classes': 102,
            'image_size': 224,
            'epochs': 200,
            'batch_size': 32,
            'lr': 0.001,  # Lower LR for fine-tuning
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 5,
            'label_smoothing': 0.1,
            'use_pretrained': True,  # Use pretrained weights
        },
        'pets': {
            'num_classes': 37,
            'image_size': 224,
            'epochs': 50,
            'batch_size': 32,
            'lr': 0.001,  # Lower LR for fine-tuning
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 5,
            'label_smoothing': 0.0,
            'use_pretrained': True,  # Use pretrained weights
        },
        'food101': {
            'num_classes': 101,
            'image_size': 224,
            'epochs': 30,
            'batch_size': 64,
            'lr': 0.001,  # Lower LR for fine-tuning
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 5,
            'label_smoothing': 0.0,
            'use_pretrained': True,  # Use pretrained weights
        },
    }
    return configs.get(dataset_name.lower(), configs['flowers102'])


def get_mobilenet_config():
    """Return pruning-specific overrides for MobileNet architectures.
    
    MobileNet uses depthwise separable convolutions with very few weights per
    filter (9 per depthwise channel), so there is almost no redundancy to absorb
    pruning gracefully.  These settings are chosen to:
      - Prune sooner (lower patience)
      - Skip the LR-restart safety cycle so degradation shows up immediately
      - Apply larger prune steps so the accuracy drop is clearly visible
    return {
        'patience': 10,           # Trigger pruning after only 10 stagnant epochs
        'skip_lr_restart': True,  # Go straight to pruning; no LR-restart cycle
        'prune_min': 0.10,        # At least 10% pruned per step (vs 5% for ResNet)
        'prune_max': 0.30,        # Up to 30% per step  (vs 20% for ResNet)
        'prune_factor': 0.5,      # Prune 50% of remaining gap each step (vs 30%)
    }
    """
    return {
        'patience': 10,          # Original patience
        'skip_lr_restart': False, # Use the LR-restart cycle before pruning
        'prune_min': 0.05,
        'prune_max': 0.20,
        'prune_factor': 0.3,
    }

def get_resnet_config():
    """Return pruning-specific settings for ResNet / skip-connection architectures.
    
    ResNets are highly redundant and the skip connections compensate for pruned
    weights, so we give them more time and prune more gradually.
    """
    return {
        'patience': 100,          # Original patience
        'skip_lr_restart': False, # Use the LR-restart cycle before pruning
        'prune_min': 0.05,
        'prune_max': 0.20,
        'prune_factor': 0.3,
    }


def get_model_size(model):
    """Return number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_effective_size(model):
    """Return number of non-zero (effective) parameters.
    
    For pruned models, this checks the pruning mask to count active parameters.
    """
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Check if module has been pruned (has weight_mask attribute)
            if hasattr(module, 'weight_mask'):
                # Count non-zero entries in the mask
                total += (module.weight_mask != 0).sum().item()
            else:
                # No pruning applied yet, count non-zero weights
                if module.weight.requires_grad:
                    total += (module.weight != 0).sum().item()
        # For other parameter types (biases, etc.)
        elif hasattr(module, '_parameters'):
            for param_name, param in module._parameters.items():
                if param is not None and param.requires_grad and param_name != 'weight':
                    total += param.numel()
    return total


def get_model_sparsity(model):
    """Return percentage of zero parameters.
    
    For pruned models, this checks the pruning mask to count pruned parameters.
    """
    total_params = 0
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Check if module has been pruned (has weight_mask attribute)
            if hasattr(module, 'weight_mask'):
                # Count based on mask
                total_params += module.weight_mask.numel()
                zero_params += (module.weight_mask == 0).sum().item()
            else:
                # No pruning applied yet, count zeros in weights
                if module.weight.requires_grad:
                    total_params += module.weight.numel()
                    zero_params += (module.weight == 0).sum().item()
    return 100.0 * zero_params / total_params if total_params > 0 else 0


def apply_pruning(model, amount, method='l1', scope='global'):
    """Apply pruning to model.
    
    Args:
        model: The model to prune
        amount: Fraction of parameters to prune (0-1)
        method: Pruning method ('l1', 'random', 'ln')
        scope: 'global' (prune across all layers) or 'local' (per-layer)
    """
    parameters_to_prune = []
    
    # Collect all Linear and Conv2d layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            parameters_to_prune.append((module, 'weight'))
    
    if len(parameters_to_prune) == 0:
        print("Warning: No layers found to prune")
        return model
    
    # Apply pruning based on scope
    if scope == 'global':
        # Global pruning - prune across all layers to reach target amount
        if method == 'l1':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=amount,
            )
        elif method == 'random':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=amount,
            )
        elif method == 'ln':
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.LnStructured,
                amount=amount,
                n=2,
                dim=0,
            )
    else:  # local/per-layer pruning
        # Apply same amount to each layer independently
        for module, param_name in parameters_to_prune:
            if method == 'l1':
                prune.l1_unstructured(module, name=param_name, amount=amount)
            elif method == 'random':
                prune.random_unstructured(module, name=param_name, amount=amount)
            elif method == 'ln':
                prune.ln_structured(module, name=param_name, amount=amount, n=2, dim=0)
    
    return model


def make_pruning_permanent(model):
    """Remove pruning reparametrization and make it permanent."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            try:
                prune.remove(module, 'weight')
            except:
                pass
    return model


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        
        output = model(image)
        if hasattr(output, 'logits'):
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
    
    return metric_logger.acc1.global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")

    with torch.inference_mode():
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            if hasattr(output, 'logits'):
                output = output.logits
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def measure_inference_latency(model, data_loader, device, warmup_batches=10):
    """Measure inference latency and throughput (FPS) of the model."""
    model.eval()
    
    print("\n" + "="*80)
    print("MEASURING INFERENCE LATENCY")
    print("="*80)
    
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
            if hasattr(output, 'logits'):
                output = output.logits
            if device.type == 'cuda':
                torch.cuda.synchronize()
        
        # Timing phase
        print("Measuring latency...")
        for i, (image, _) in enumerate(data_loader):
            image = image.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            output = model(image)
            if hasattr(output, 'logits'):
                output = output.logits
            
            if device.type == 'cuda':
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
    p50_ms = sorted_times[len(sorted_times)//2] * 1000
    p95_ms = sorted_times[int(len(sorted_times)*0.95)] * 1000
    p99_ms = sorted_times[int(len(sorted_times)*0.99)] * 1000
    
    results = {
        'fps': fps,
        'mean_latency_ms': mean_latency_ms,
        'p50_latency_ms': p50_ms,
        'p95_latency_ms': p95_ms,
        'p99_latency_ms': p99_ms,
        'total_images': total_images,
        'total_batches': len(batch_times),
        'total_time_s': total_time,
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
    print("="*80 + "\n")
    
    return results


def load_dataset(dataset_name, data_path, batch_size, workers):
    """Load dataset with standard preprocessing."""
    print(f"Loading {dataset_name} dataset from {data_path}")
    
    # Dataset-specific configurations
    dataset_configs = {
        'flowers102': {
            'num_classes': 102,
            'img_size': 224,
            'train_split': 'train',
            'test_split': 'test',
            'dataset_class': torchvision.datasets.Flowers102,
        },
        'pets': {
            'num_classes': 37,
            'img_size': 224,
            'train_split': 'trainval',
            'test_split': 'test',
            'dataset_class': torchvision.datasets.OxfordIIITPet,
        },
        'food101': {
            'num_classes': 101,
            'img_size': 224,
            'train_split': 'train',
            'test_split': 'test',
            'dataset_class': torchvision.datasets.Food101,
        },
        'cifar100': {
            'num_classes': 100,
            'img_size': 32,
            'train_split': True,  # CIFAR uses True/False
            'test_split': False,
            'dataset_class': torchvision.datasets.CIFAR100,
        },
        'stl10': {
            'num_classes': 10,
            'img_size': 96,
            'train_split': 'train',
            'test_split': 'test',
            'dataset_class': torchvision.datasets.STL10,
        },
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    img_size = config['img_size']
    interpolation = InterpolationMode.BILINEAR
    
    # Training transforms
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(img_size, interpolation=interpolation),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transforms
    val_resize_size = img_size if img_size <= 32 else int(img_size * 256 / 224)
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(val_resize_size, interpolation=interpolation),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets based on type
    if dataset_name == 'cifar100':
        dataset_train = config['dataset_class'](
            root=data_path, train=config['train_split'], download=True, transform=train_transform
        )
        dataset_test = config['dataset_class'](
            root=data_path, train=config['test_split'], download=True, transform=val_transform
        )
    elif dataset_name == 'stl10':
        dataset_train = config['dataset_class'](
            root=data_path, split=config['train_split'], download=True, transform=train_transform
        )
        dataset_test = config['dataset_class'](
            root=data_path, split=config['test_split'], download=True, transform=val_transform
        )
    else:
        dataset_train = config['dataset_class'](
            root=data_path, split=config['train_split'], download=True, transform=train_transform
        )
        dataset_test = config['dataset_class'](
            root=data_path, split=config['test_split'], download=True, transform=val_transform
        )
    
    print(f"Train dataset size: {len(dataset_train)}")
    print(f"Test dataset size: {len(dataset_test)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )
    
    return train_loader, test_loader, config['num_classes']


def load_model_from_hf(hf_repo_id, num_classes, model_type='resnet'):
    """Load model from HuggingFace and adapt for target number of classes."""
    
    # For mobilenet, load directly from torchvision to avoid HuggingFace wrapper issues
    if model_type == 'mobilenet':
        model_name = hf_repo_id.split('/')[-1].replace('-', '_')
        print(f"\nLoading model directly from torchvision: {model_name}")
        model = torchvision.models.get_model(model_name, weights='DEFAULT')
        print(f"Successfully loaded torchvision model")
        # Replace final classifier layer
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        print(f"Replaced classifier layer for {num_classes} classes")
        return model

    print(f"\nLoading model from HuggingFace: {hf_repo_id}")
    
    # Check if it's a perforated model
    if 'perforated' in hf_repo_id.lower():
        from perforatedai import utils_perforatedai as UPA
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import library_perforatedai as LPA
        
        # Create base model architecture
        base_model = torchvision.models.get_model('resnet18', weights=None, num_classes=1000)
        model = LPA.ResNetPAIPreFC(base_model)
        # Load from HuggingFace
        model = UPA.from_hf_pretrained(model, hf_repo_id)
        print(f"Successfully loaded perforated model from HuggingFace")
    else:
        # Try loading as transformers model
        try:
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(hf_repo_id)
            print(f"Successfully loaded transformers model from HuggingFace")
        except Exception as e:
            print(f"Failed to load as transformers model: {e}")
            # Fallback: try loading as torchvision model
            model_name = hf_repo_id.split('/')[-1].replace('-', '')
            print(f"Attempting to load as torchvision model: {model_name}")
            model = torchvision.models.get_model(model_name, weights='IMAGENET1K_V1')
            print(f"Successfully loaded torchvision model")
    
    # Replace final layer for target number of classes
    if hasattr(model, 'fc'):
        # Check if it's a TrackedNeuronModule (from HuggingFace PAI model) or regular Linear
        if hasattr(model.fc, 'main_module'):
            in_features = model.fc.main_module.in_features
        else:
            in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"Replaced fc layer for {num_classes} classes")
    elif hasattr(model, 'classifier'):
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
    
    # Load model from HuggingFace
    model = load_model_from_hf(args.hf_repo_id, num_classes, model_type=args.model_type)
    model = model.to(device)
    
    # Print initial model size
    initial_size = get_model_size(model)
    print(f"\nInitial model size: {initial_size:,} parameters")
    
    if args.target_params:
        print(f"Target size: {args.target_params:,} parameters")
        if initial_size <= args.target_params:
            print(f"Model already at or below target size. No pruning needed.")
    
    # Setup training
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    # Use CosineAnnealingLR as main scheduler
    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=0.0
    )
    
    # Add warmup if specified
    if args.lr_warmup_epochs > 0:
        warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=0.01, total_iters=args.lr_warmup_epochs
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
            milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
    
    # Training loop with iterative pruning (IMP with patience)
    print("\nStarting training with Iterative Magnitude Pruning (IMP)...")
    print("Will run until target parameter count is reached")
    print("Strategy: LR restart first, then prune only if restart doesn't help\n")
    print("Epoch | Train Acc@1 | Test Acc@1 |     LR     | Prune Count | Effective Params | Phase")
    print("-" * 105)
    
    # Load model-type-specific pruning config
    if args.model_type == 'mobilenet':
        prune_cfg = get_mobilenet_config()
    else:
        prune_cfg = get_resnet_config()

    start_time = time.time()
    best_acc1 = 0.0  # Best test accuracy
    best_train_acc1 = 0.0  # Best train accuracy
    best_epoch = 0
    prune_step = 0
    epochs_without_improvement = 0
    patience = prune_cfg['patience']  # Wait epochs without improvement before action
    epoch = 0
    
    # Track parameter count and accuracy before each pruning step
    pruning_history = []
    
    # Track LR restart state
    lr_restart_done = False  # Have we done LR restart since last prune?
    best_acc_before_restart = 0.0  # Best accuracy before the restart
    phase = "training"  # Current phase: "training", "lr_restart", or "post_restart"
    
    # Run until target size is reached (or forever if no target set)
    while True:
        train_acc1 = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        test_acc1, test_loss = evaluate(model, criterion, test_loader, device)
        
        # Track best accuracies and patience
        if test_acc1 > best_acc1:
            best_acc1 = test_acc1
            best_epoch = epoch + 1
            epochs_without_improvement = 0  # Reset patience counter
        else:
            epochs_without_improvement += 1
        
        if train_acc1 > best_train_acc1:
            best_train_acc1 = train_acc1
        
        current_effective_size = get_effective_size(model)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Single consolidated print per epoch
        print(f"{epoch+1:5d} | {train_acc1:11.3f} | {test_acc1:10.3f} | {current_lr:10.6f} | {prune_step:11d} | {current_effective_size:,} | {phase}")
        
        # Check if we've reached target size
        if args.target_params and current_effective_size <= args.target_params:
            print(f"\nTarget parameter count reached: {current_effective_size:,} <= {args.target_params:,}")
            break
        
        # Strategy: First try LR restart, then prune only if restart doesn't help
        if (args.target_params and 
            current_effective_size > args.target_params and 
            epochs_without_improvement >= patience):
            
            if not lr_restart_done and not prune_cfg['skip_lr_restart']:
                # First, try LR restart without pruning
                print(f"\n{'='*80}")
                print(f"Patience exceeded. Attempting LR RESTART (no pruning yet)...")
                print(f"Best test accuracy before restart: {best_acc1:.3f}%")
                print(f"Best train accuracy: {best_train_acc1:.3f}%")
                print(f"{'='*80}")
                
                # Save best accuracy before restart
                best_acc_before_restart = best_acc1
                
                # Recreate optimizer with initial LR (restart)
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
                
                # Recreate LR scheduler
                main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=0.0
                )
                if args.lr_warmup_epochs > 0:
                    warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                        optimizer, factor=0.01, total_iters=args.lr_warmup_epochs
                    )
                    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                        optimizer, 
                        schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
                        milestones=[args.lr_warmup_epochs]
                    )
                else:
                    lr_scheduler = main_lr_scheduler
                
                # Reset counters but keep best accuracies to compare
                epochs_without_improvement = 0
                lr_restart_done = True
                phase = "lr_restart"
                
                print(f"Epoch | Train Acc@1 | Test Acc@1 |     LR     | Prune Count | Effective Params | Phase")
                print("-" * 105)
            
            else:
                # LR restart was already done (or skipped for this model type),
                # now check if it helped (skip_lr_restart always goes straight to pruning)
                if not prune_cfg['skip_lr_restart'] and best_acc1 > best_acc_before_restart:
                    print(f"\n{'='*80}")
                    print(f"LR restart helped! Improved from {best_acc_before_restart:.3f}% to {best_acc1:.3f}%")
                    print(f"Continuing training without pruning...")
                    print(f"{'='*80}\n")
                    
                    # Reset for next cycle
                    epochs_without_improvement = 0
                    lr_restart_done = False
                    phase = "post_restart"
                else:
                    # LR restart didn't help, now we prune
                    print(f"\n{'='*80}")
                    print(f"LR restart did NOT help. Best remained at {best_acc1:.3f}%")
                    print(f"Now applying PRUNING...")
                    print(f"{'='*80}")
                    
                    # Record parameter count and accuracies before pruning
                    pruning_history.append((current_effective_size, best_train_acc1, best_acc1))
                    if pruning_history:
                        print(f"\nPruning History so far (Effective Params, Best Train Acc@1, Best Test Acc@1):")
                        for params, train_acc, test_acc in pruning_history:
                            print(f"{params},{train_acc:.3f},{test_acc:.3f}")
                    
                    # Calculate how much to prune to get closer to target
                    remaining_to_prune = (current_effective_size - args.target_params) / current_effective_size
                    # Clamp between model-type-specific min/max, scaled by prune_factor
                    prune_amount = max(
                        prune_cfg['prune_min'],
                        min(prune_cfg['prune_max'], remaining_to_prune * prune_cfg['prune_factor'])
                    )
                    
                    print(f"Pruning {prune_amount*100:.1f}% of parameters...")
                    
                    prune_step += 1
                    
                    model = apply_pruning(model, prune_amount, method=args.prune_method, scope=args.prune_scope)
                    
                    # Reset everything after pruning
                    epochs_without_improvement = 0
                    best_acc1 = 0.0  # Reset to track pruned model's best
                    best_train_acc1 = 0.0
                    lr_restart_done = False
                    phase = "training"
                    
                    # Recreate optimizer for the pruned model with initial LR
                    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
                    
                    # Recreate LR scheduler
                    main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                        optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=0.0
                    )
                    if args.lr_warmup_epochs > 0:
                        warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                            optimizer, factor=0.01, total_iters=args.lr_warmup_epochs
                        )
                        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                            optimizer, 
                            schedulers=[warmup_lr_scheduler, main_lr_scheduler], 
                            milestones=[args.lr_warmup_epochs]
                        )
                    else:
                        lr_scheduler = main_lr_scheduler
                    
                    print(f"Epoch | Train Acc@1 | Test Acc@1 |     LR     | Prune Count | Effective Params | Phase")
                    print("-" * 105)
        
        epoch += 1
    
    # Record final parameter count and best accuracies
    final_effective_size = get_effective_size(model)
    pruning_history.append((final_effective_size, best_train_acc1, best_acc1))
    
    # Make pruning permanent before returning
    if args.target_params:
        final_size = get_model_size(model)
        final_sparsity = get_model_sparsity(model)
        print(f"\nMaking pruning permanent...")
        model = make_pruning_permanent(model)
        print(f"Final total parameters: {final_size:,}")
        print(f"Final effective (non-zero) parameters: {final_effective_size:,}")
        print(f"Final sparsity: {final_sparsity:.2f}%")
        print(f"Total pruning steps: {prune_step}")
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"\nTraining complete! Total time: {total_time_str}")
    print(f"Best Train Accuracy: {best_train_acc1:.3f}%")
    print(f"Best Test Accuracy: {best_acc1:.3f}% (achieved at epoch {best_epoch})")
    
    # Print pruning history
    if pruning_history:
        print(f"\nPruning History (Effective Params, Best Train Acc@1, Best Test Acc@1):")
        for params, train_acc, test_acc in pruning_history:
            print(f"{params},{train_acc:.3f},{test_acc:.3f}")
    
    return best_acc1, best_epoch, model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model from HuggingFace with multiple runs and statistics")
    parser.add_argument("--hf-repo-id", required=True, type=str, 
                       help="HuggingFace repository ID (e.g., 'perforated-ai/resnet-18-perforated' or 'microsoft/resnet-18')")
    parser.add_argument("--dataset", default="flowers102", type=str,
                       choices=['flowers102', 'pets', 'food101', 'cifar100', 'stl10'],
                       help="Dataset to train on (default: flowers102)")
    parser.add_argument("--data-path", default="./data", type=str, help="Dataset path")
    parser.add_argument("--batch-size", default=None, type=int, help="Batch size (default: dataset-specific)")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: dataset-specific)")
    parser.add_argument("--lr", default=None, type=float, help="Learning rate (default: dataset-specific)")
    parser.add_argument("--lr-warmup-epochs", default=None, type=int, help="Number of warmup epochs (default: dataset-specific)")
    parser.add_argument("--label-smoothing", default=None, type=float, help="Label smoothing (default: dataset-specific)")
    parser.add_argument("--workers", default=16, type=int, help="Number of data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cuda or cpu)")
    parser.add_argument("--print-freq", default=10, type=int, help="Print frequency")
    
    # Pruning parameters
    parser.add_argument("--target-params", default=None, type=int, 
                       help="Target number of parameters (model will be pruned during training to reach this)")
    parser.add_argument("--prune-method", default="l1", type=str,
                       choices=['l1', 'random', 'ln'],
                       help="Pruning method: l1 (magnitude), random, or ln (structured)")
    parser.add_argument("--prune-scope", default="global", type=str,
                       choices=['global', 'local'],
                       help="Pruning scope: global (across all layers) or local (per-layer uniform)")
    parser.add_argument("--model-type", default="resnet", type=str,
                       choices=['resnet', 'mobilenet'],
                       help=("Architecture family being pruned. "
                             "'mobilenet': loads directly from torchvision using the last component of "
                             "--hf-repo-id as the model name (e.g. tv/mobilenet_v3_large, tv/mnasnet1_0, "
                             "tv/efficientnet_b0). Uses conservative pruning settings. "
                             "'resnet': loads from HuggingFace, patience=100, LR-restart first, 5-20%% steps."))
    
    args = parser.parse_args()
    
    # Apply dataset-specific defaults if not explicitly set
    config = get_dataset_config(args.dataset)
    
    if args.batch_size is None:
        args.batch_size = config.get('batch_size', 32)
    if args.epochs is None:
        args.epochs = config.get('epochs', 50)
    if args.lr is None:
        args.lr = config.get('lr', 0.001)
    if args.lr_warmup_epochs is None:
        args.lr_warmup_epochs = config.get('lr_warmup_epochs', 5)
    if args.label_smoothing is None:
        args.label_smoothing = config.get('label_smoothing', 0.1)
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"HuggingFace Repo: {args.hf_repo_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LR warmup epochs: {args.lr_warmup_epochs}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Device: {args.device}")
    if args.target_params:
        print(f"Target params: {args.target_params:,}")
        print(f"Pruning method: {args.prune_method}")
        print(f"Pruning scope: {args.prune_scope}")
        print(f"Model type: {args.model_type}")
    print(f"{'='*80}\n")
    
    # Load dataset
    train_loader, test_loader, num_classes = load_dataset(
        args.dataset, args.data_path, args.batch_size, args.workers
    )
    
    # Run training
    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")
    
    best_acc1, best_epoch, model = train_single_run(args, train_loader, test_loader, num_classes)
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Acc@1: {best_acc1:.3f}%")
    print(f"Best Epoch: {best_epoch}")
    print(f"{'='*80}\n")
    
    # Measure inference latency
    print(f"\n{'='*80}")
    print(f"MEASURING LATENCY")
    print(f"{'='*80}\n")
    device = torch.device(args.device)
    latency_results = measure_inference_latency(model, test_loader, device)
    
    # Print results in CSV format
    print(f"\n{'='*80}")
    print(f"RESULTS - CSV FORMAT")
    print(f"{'='*80}\n")
    
    # Extract model name from repo ID
    model_name = args.hf_repo_id.split('/')[-1]
    
    # CSV Header
    print("Model,Dataset,BestAcc1,BestEpoch,FPS,MeanLatency_ms,P95Latency_ms")
    
    # Results
    print(f"{model_name},{args.dataset},{best_acc1:.3f},{best_epoch},"
          f"{latency_results['fps']:.2f},{latency_results['mean_latency_ms']:.2f},"
          f"{latency_results['p95_latency_ms']:.2f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
