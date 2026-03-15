'''
Training script that loads models from HuggingFace for knowledge distillation.

Usage:
python train_from_hf_kd.py --teacher-hf-repo-id microsoft/resnet-34 --student-hf-repo-id microsoft/resnet-18 --dataset flowers102
python train_from_hf_kd.py --teacher-hf-repo-id perforated-ai/resnet-34-perforated --student-hf-repo-id perforated-ai/resnet-18-perforated --dataset pets --temperature 5.0 --alpha 0.95
'''

import datetime
import os
import time
import random
import numpy as np
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torch import nn
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode

from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For reproducibility (may reduce performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


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


def get_model_size(model):
    """Return number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_checkpoint_path(hf_repo_id, dataset_name, checkpoint_dir="./checkpoints"):
    """Generate checkpoint path from HF repo ID and dataset name."""
    # Sanitize HF repo ID (replace / with _)
    model_name = hf_repo_id.replace('/', '_').replace('-', '_')
    checkpoint_name = f"{model_name}_{dataset_name}.pt"
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, checkpoint_name)


def save_checkpoint(model, hf_repo_id, dataset_name, best_acc1, best_epoch, checkpoint_dir="./checkpoints"):
    """Save model checkpoint."""
    checkpoint_path = get_checkpoint_path(hf_repo_id, dataset_name, checkpoint_dir)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hf_repo_id': hf_repo_id,
        'dataset_name': dataset_name,
        'best_acc1': best_acc1,
        'best_epoch': best_epoch,
    }, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(model, hf_repo_id, dataset_name, checkpoint_dir="./checkpoints"):
    """Load model checkpoint if it exists."""
    checkpoint_path = get_checkpoint_path(hf_repo_id, dataset_name, checkpoint_dir)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"  Best Acc@1: {checkpoint['best_acc1']:.3f}% (epoch {checkpoint['best_epoch']})")
        return True
    return False


def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha, criterion):
    """Compute knowledge distillation loss.
    
    Args:
        student_logits: Logits from student model
        teacher_logits: Logits from teacher model
        labels: Ground truth labels
        temperature: Temperature for softening probability distributions
        alpha: Weight for distillation loss vs task loss (0-1)
        criterion: Loss function for hard labels (e.g., CrossEntropyLoss)
    
    Returns:
        Combined distillation loss
    """
    # Soft targets loss (KL divergence between teacher and student)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    distill_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Hard targets loss (regular classification loss)
    task_loss = criterion(student_logits, labels)
    
    # Combine losses
    return alpha * distill_loss + (1 - alpha) * task_loss


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=10, teacher_model=None, temperature=4.0, alpha=0.9):
    model.train()
    if teacher_model is not None:
        teacher_model.eval()  # Teacher always in eval mode
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        
        output = model(image)
        if hasattr(output, 'logits'):
            output = output.logits
        
        # Compute loss (with or without distillation)
        if teacher_model is not None:
            with torch.no_grad():
                teacher_output = teacher_model(image)
                if hasattr(teacher_output, 'logits'):
                    teacher_output = teacher_output.logits
            loss = distillation_loss(output, teacher_output, target, temperature, alpha, criterion)
        else:
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


def load_model_from_hf(hf_repo_id, num_classes):
    """Load model from HuggingFace and adapt for target number of classes."""
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
    elif hf_repo_id.startswith('torchvision/'):
        tv_name = hf_repo_id[len('torchvision/'):]
        model = torchvision.models.get_model(tv_name, weights='DEFAULT')
        print(f"Successfully loaded torchvision model: {tv_name}")
    elif hf_repo_id.startswith('timm/') or ':' not in hf_repo_id and '/' not in hf_repo_id:
        # Load directly via timm using the model name after 'timm/'
        import timm
        timm_name = hf_repo_id[len('timm/'):] if hf_repo_id.startswith('timm/') else hf_repo_id
        model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
        print(f"Successfully loaded timm model: {timm_name}")
        return model  # timm handles num_classes directly, no layer replacement needed
    else:
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(hf_repo_id)
        print(f"Successfully loaded transformers model from HuggingFace")
    
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
    """Perform a single training run with optional knowledge distillation.
    
    If only teacher is specified: Train teacher and save checkpoint
    If both teacher and student specified: Load/train teacher, then distill to student
    """
    device = torch.device(args.device)
    
    # Determine mode: teacher-only training or distillation
    teacher_only = args.teacher_hf_repo_id and not args.student_hf_repo_id
    
    if teacher_only:
        # Mode 1: Train teacher only
        print(f"\n{'='*80}")
        print("MODE: Training Teacher Model")
        print(f"{'='*80}\n")
        
        print(f"Loading teacher model from HuggingFace: {args.teacher_hf_repo_id}")
        model = load_model_from_hf(args.teacher_hf_repo_id, num_classes)
        model = model.to(device)
        model_size = get_model_size(model)
        print(f"Model size: {model_size:,} parameters")
        
        # Check if checkpoint exists
        checkpoint_path = get_checkpoint_path(args.teacher_hf_repo_id, args.dataset, args.checkpoint_dir)
        if os.path.exists(checkpoint_path) and not args.force_retrain:
            print(f"\nCheckpoint already exists: {checkpoint_path}")
            print("Use --force-retrain to train from scratch")
            load_checkpoint(model, args.teacher_hf_repo_id, args.dataset, args.checkpoint_dir)
            return 0.0, 0, model  # Return dummy values since we're not training
        
        # Train the teacher
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        
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
        
        print("\nStarting teacher training...")
        print(f"Will run for {args.epochs} epochs\n")
        print("Epoch | Train Acc@1 | Test Acc@1 | Parameters")
        print("-" * 60)
        
        start_time = time.time()
        best_acc1 = 0.0
        best_epoch = 0
        
        for epoch in range(args.epochs):
            train_acc1 = train_one_epoch(
                model, criterion, optimizer, train_loader, device, epoch, args.print_freq
            )
            lr_scheduler.step()
            test_acc1, test_loss = evaluate(model, criterion, test_loader, device)
            
            if test_acc1 > best_acc1:
                best_acc1 = test_acc1
                best_epoch = epoch + 1
            
            current_size = get_model_size(model)
            print(f"{epoch+1:5d} | {train_acc1:11.3f} | {test_acc1:10.3f} | {current_size:,}")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"\nTeacher training complete! Total time: {total_time_str}")
        print(f"Best Test Accuracy: {best_acc1:.3f}% (achieved at epoch {best_epoch})")
        UPA.count_params(model)
        
        # Save teacher checkpoint
        save_checkpoint(model, args.teacher_hf_repo_id, args.dataset, best_acc1, best_epoch, args.checkpoint_dir)
        
        return best_acc1, best_epoch, model
    
    else:
        # Mode 2: Knowledge distillation (teacher -> student)
        print(f"\n{'='*80}")
        print("MODE: Knowledge Distillation")
        print(f"{'='*80}\n")
        
        # Load or train teacher model
        teacher_model = None
        if args.teacher_hf_repo_id:
            print(f"Loading teacher model from HuggingFace: {args.teacher_hf_repo_id}")
            teacher_model = load_model_from_hf(args.teacher_hf_repo_id, num_classes)
            teacher_model = teacher_model.to(device)
            teacher_size = get_model_size(teacher_model)  # Get size before freezing
            
            # Try to load checkpoint
            checkpoint_loaded = load_checkpoint(teacher_model, args.teacher_hf_repo_id, args.dataset, args.checkpoint_dir)
            
            if not checkpoint_loaded:
                print(f"\nNo checkpoint found for teacher. Training teacher first...")
                
                # Train teacher
                criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
                optimizer = torch.optim.SGD(teacher_model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
                
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
                
                print("\nTraining teacher...")
                print(f"Will run for {args.epochs} epochs\n")
                print("Epoch | Train Acc@1 | Test Acc@1 | Parameters")
                print("-" * 60)
                
                best_teacher_acc1 = 0.0
                best_teacher_epoch = 0
                
                for epoch in range(args.epochs):
                    train_acc1 = train_one_epoch(
                        teacher_model, criterion, optimizer, train_loader, device, epoch, args.print_freq
                    )
                    lr_scheduler.step()
                    test_acc1, test_loss = evaluate(teacher_model, criterion, test_loader, device)
                    
                    if test_acc1 > best_teacher_acc1:
                        best_teacher_acc1 = test_acc1
                        best_teacher_epoch = epoch + 1
                    
                    current_size = get_model_size(teacher_model)
                    print(f"{epoch+1:5d} | {train_acc1:11.3f} | {test_acc1:10.3f} | {current_size:,}")
                
                print(f"\nTeacher training complete!")
                print(f"Best Teacher Accuracy: {best_teacher_acc1:.3f}% (epoch {best_teacher_epoch})")
                UPA.count_params(teacher_model)
                
                # Save teacher checkpoint
                save_checkpoint(teacher_model, args.teacher_hf_repo_id, args.dataset, 
                              best_teacher_acc1, best_teacher_epoch, args.checkpoint_dir)
            
            # Freeze teacher
            teacher_model.eval()
            for param in teacher_model.parameters():
                param.requires_grad = False
            print(f"\nTeacher model size: {teacher_size:,} parameters (frozen)")
        
        # Load student model
        print(f"\nLoading student model from HuggingFace: {args.student_hf_repo_id}")
        model = load_model_from_hf(args.student_hf_repo_id, num_classes)
        model = model.to(device)
    
        # Print model sizes
        student_size = get_model_size(model)
        print(f"Student model size: {student_size:,} parameters")
        
        if teacher_model is not None:
            print(f"\nKnowledge Distillation Settings:")
            print(f"  Temperature: {args.temperature}")
            print(f"  Alpha (distillation weight): {args.alpha}")
            print(f"  Teacher size / Student size: {teacher_size / student_size:.2f}x")
        
        # Setup PAI for student — settings depend on model architecture
        is_mobilenet = 'mobilenet' in args.student_hf_repo_id.lower()
        if is_mobilenet:
            GPA.pc.append_module_names_to_convert(["InvertedResidual"])
            if hasattr(model, 'blocks'):
                # timm mobilenetv3 — uses .blocks and .conv_head
                num_features = sum(1 for _ in model.blocks)
                for i in range(num_features):
                    GPA.pc.append_module_ids_to_track([f'.blocks.{i}'])
                GPA.pc.append_module_ids_to_track(['.conv_head'])
            else:
                # torchvision mobilenetv3 — uses .features
                num_features = sum(1 for _ in model.features)
                for i in range(num_features):
                    GPA.pc.append_module_ids_to_track([f'.features.{i}'])
            GPA.pc.append_module_ids_to_convert(['.classifier'])
            GPA.pc.set_testing_dendrite_capacity(False)
        else:
            # ResNet defaults
            GPA.pc.append_module_names_to_convert(["BasicBlock", "Bottleneck"])
            for i in range(4):
                GPA.pc.append_module_ids_to_track(['.layer' + str(i + 1)])
            GPA.pc.append_module_ids_to_track(['.b1'])
            GPA.pc.append_module_ids_to_track(['.fc', '.conv1', '.bn1'])
            GPA.pc.set_testing_dendrite_capacity(False)
        # Setup PAI for student — settings depend on model architecture
        GPA.pc.set_n_epochs_to_switch(25)
        GPA.pc.set_weight_decay_accepted(True)
        model = UPA.initialize_pai(model, save_name="PAI-" + args.student_hf_repo_id.split('/')[-1])
        model = model.to(device)
        if is_mobilenet:
            model.classifier.set_this_output_dimensions([-1,  0])


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
        GPA.pai_tracker.set_optimizer_instance(optimizer)

        # Training loop
        print("\nStarting student training...")
        if teacher_model is not None:
            print(f"Training student with knowledge distillation from teacher\n")
        print(f"Will run for {args.epochs} epochs\n")
        print("Epoch | Train Acc@1 | Test Acc@1 | Parameters")
        print("-" * 60)
        
        start_time = time.time()
        best_acc1 = 0.0
        best_epoch = 0
        
        epoch = -1
        while True:
            epoch += 1
            train_acc1 = train_one_epoch(
                model, criterion, optimizer, train_loader, device, epoch, args.print_freq,
                teacher_model=teacher_model, temperature=args.temperature, alpha=args.alpha
            )
            lr_scheduler.step()
            test_acc1, test_loss = evaluate(model, criterion, test_loader, device)
            
            # Track best accuracy
            if test_acc1 > best_acc1:
                best_acc1 = test_acc1
                best_epoch = epoch + 1
            
            current_size = get_model_size(model)
            
            # Single consolidated print per epoch
            print(f"{epoch+1:5d} | {train_acc1:11.3f} | {test_acc1:10.3f} | {current_size:,}")

            GPA.pai_tracker.add_extra_score(train_acc1, "train_acc1")
            model, restructured, training_complete = GPA.pai_tracker.add_validation_score(test_acc1, model)
            model.to(device)
            if training_complete:
                break
            elif restructured:
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
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
                GPA.pai_tracker.set_optimizer_instance(optimizer)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"\nStudent training complete! Total time: {total_time_str}")
        print(f"Best Test Accuracy: {best_acc1:.3f}% (achieved at epoch {best_epoch})")
        UPA.count_params(model)
        
        return best_acc1, best_epoch, model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train model from HuggingFace with knowledge distillation")
    parser.add_argument("--teacher-hf-repo-id", default=None, type=str, 
                       help="HuggingFace repository ID for teacher model (e.g., 'microsoft/resnet-34')")
    parser.add_argument("--student-hf-repo-id", default=None, type=str, 
                       help="HuggingFace repository ID for student model (optional, e.g., 'microsoft/resnet-18')")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", type=str, 
                       help="Directory to save/load checkpoints (default: ./checkpoints)")
    parser.add_argument("--force-retrain", action="store_true",
                       help="Force retraining even if checkpoint exists")
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
    parser.add_argument("--seed", default=None, type=int, help="Random seed for reproducibility (default: None for random behavior)")
    
    # Knowledge Distillation parameters
    parser.add_argument("--temperature", default=4.0, type=float, 
                       help="Temperature for distillation (default: 4.0, higher = softer distributions)")
    parser.add_argument("--alpha", default=0.9, type=float, 
                       help="Weight for distillation loss vs task loss (default: 0.9, range: 0-1)")
    
    args = parser.parse_args()
    
    # Set random seed if specified
    if args.seed is not None:
        print(f"Setting random seed to {args.seed}")
        set_seed(args.seed)
    else:
        print("No seed specified - using random initialization")
    
    # Validate arguments
    if not args.teacher_hf_repo_id and not args.student_hf_repo_id:
        parser.error("At least one of --teacher-hf-repo-id or --student-hf-repo-id must be specified")
    
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
    if args.teacher_hf_repo_id:
        print(f"Teacher HF Repo: {args.teacher_hf_repo_id}")
    print(f"Student HF Repo: {args.student_hf_repo_id}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LR warmup epochs: {args.lr_warmup_epochs}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Device: {args.device}")
    if args.teacher_hf_repo_id:
        print(f"\nKnowledge Distillation:")
        print(f"  Temperature: {args.temperature}")
        print(f"  Alpha: {args.alpha}")
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
    
    # Extract model names from repo IDs
    student_name = args.student_hf_repo_id.split('/')[-1] if args.student_hf_repo_id else "None"
    teacher_name = args.teacher_hf_repo_id.split('/')[-1] if args.teacher_hf_repo_id else "None"
    
    # CSV Header
    print("Teacher,Student,Dataset,BestAcc1,BestEpoch,FPS,MeanLatency_ms,P95Latency_ms")
    
    # Results
    print(f"{teacher_name},{student_name},{args.dataset},{best_acc1:.3f},{best_epoch},"
          f"{latency_results['fps']:.2f},{latency_results['mean_latency_ms']:.2f},"
          f"{latency_results['p95_latency_ms']:.2f}")
    
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
