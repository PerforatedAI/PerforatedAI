'''
Simplified training script for Flowers102 dataset using ResNet18 loaded from HuggingFace.

Usage:
python train_flowers_from_hf.py --hf-repo-id perforated-ai/resnet-18-perforated --epochs 10
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


def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
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


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
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

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def load_flowers102_data(data_path, batch_size, workers):
    """Load Flowers102 dataset with standard preprocessing."""
    print(f"Loading Flowers102 dataset from {data_path}")
    
    # Flowers102 config
    img_size = 224
    interpolation = InterpolationMode.BILINEAR
    
    # Training transforms
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(img_size, interpolation=interpolation),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Validation transforms
    val_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256, interpolation=interpolation),
        torchvision.transforms.CenterCrop(img_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load datasets
    dataset_train = torchvision.datasets.Flowers102(
        root=data_path,
        split='train',
        download=True,
        transform=train_transform
    )
    
    dataset_test = torchvision.datasets.Flowers102(
        root=data_path,
        split='test',
        download=True,
        transform=val_transform
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
        pin_memory=True
    )
    
    return train_loader, test_loader


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Flowers102 with ResNet18 from HuggingFace")
    parser.add_argument("--hf-repo-id", required=True, type=str, help="HuggingFace repository ID (e.g., 'perforated-ai/resnet-18-perforated')")
    parser.add_argument("--data-path", default="./data", type=str, help="Dataset path")
    parser.add_argument("--batch-size", default=32, type=int, help="Batch size")
    parser.add_argument("--epochs", default=50, type=int, help="Number of epochs (default: 50 for Flowers102)")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--lr-warmup-epochs", default=5, type=int, help="Number of warmup epochs")
    parser.add_argument("--label-smoothing", default=0.1, type=float, help="Label smoothing")
    parser.add_argument("--workers", default=16, type=int, help="Number of data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cuda or cpu)")
    parser.add_argument("--print-freq", default=10, type=int, help="Print frequency")
    parser.add_argument("--latency-only", action="store_true", help="Skip training and only measure latency")
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Training Configuration")
    print(f"{'='*80}")
    print(f"HuggingFace Repo: {args.hf_repo_id}")
    print(f"Dataset: Flowers102")
    print(f"Model: ResNet18 (from HuggingFace)")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LR warmup epochs: {args.lr_warmup_epochs}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"Device: {args.device}")
    print(f"{'='*80}\n")
    
    device = torch.device(args.device)
    
    # Load Flowers102 data (102 classes)
    train_loader, test_loader = load_flowers102_data(args.data_path, args.batch_size, args.workers)
    num_classes = 102
    
    # Load model
    if 'perforated' not in args.hf_repo_id:
        # For latency-only mode with non-perforated models, load from torchvision for fair comparison
        # Extract model name from repo ID (e.g., "microsoft/resnet-18" -> "resnet18")
        model_name = args.hf_repo_id.split('/')[-1].replace('-', '')
        print(f"\nLoading torchvision model: {model_name}")
        model = torchvision.models.get_model(model_name, weights='IMAGENET1K_V1')
        print(f"Successfully loaded torchvision model")
    else:
        # Load from HuggingFace
        print(f"\nLoading model from HuggingFace: {args.hf_repo_id}")
        from perforatedai import utils_perforatedai as UPA
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import library_perforatedai as LPA
            
        # Create base model architecture
        if('perforated' in args.hf_repo_id):
            base_model = torchvision.models.get_model('resnet18', weights=None, num_classes=1000)
            model = LPA.ResNetPAIPreFC(base_model)
            # Load from HuggingFace
            model = UPA.from_hf_pretrained(model, args.hf_repo_id)
        else:
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(args.hf_repo_id)
        print(f"Successfully loaded model from HuggingFace")
    
    # Replace final layer for Flowers102 (102 classes)
    if hasattr(model, 'fc'):
        # Check if it's a TrackedNeuronModule (from HuggingFace PAI model) or regular Linear (from torchvision)
        if hasattr(model.fc, 'main_module'):
            in_features = model.fc.main_module.in_features
        else:
            in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"Replaced final layer for {num_classes} classes")
    elif hasattr(model, 'classifier'):
        # Transformers models use 'classifier' instead of 'fc'
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        print(f"Replaced classifier layer for {num_classes} classes")
    
    model = model.to(device)
    
    # Setup training with Flowers102-specific hyperparameters
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
    
    # Training loop
    if not args.latency_only:
        print("\nStarting training...")
        start_time = time.time()
        best_acc1 = 0.0
        best_epoch = 0
        
        for epoch in range(args.epochs):
            train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.print_freq)
            lr_scheduler.step()
            test_acc1, test_loss = evaluate(model, criterion, test_loader, device)
            
            # Track best accuracy
            if test_acc1 > best_acc1:
                best_acc1 = test_acc1
                best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1}/{args.epochs} - Test Acc@1: {test_acc1:.3f}, Loss: {test_loss:.4f}")
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"\nTraining complete! Total time: {total_time_str}")
        print(f"Final Test Accuracy: {test_acc1:.3f}%")
        print(f"Final Test Loss: {test_loss:.4f}")
        print(f"Best Test Accuracy: {best_acc1:.3f}% (achieved at epoch {best_epoch})")
    else:
        print("\nSkipping training (latency-only mode)")
        test_acc1 = None
        test_loss = None
    
    if args.latency_only:
        # Measure inference latency on CPU, one image at a time
        print(f"\n{'='*80}")
        print("Measuring inference latency on CPU (single images)...")
        print(f"{'='*80}")
        model = model.to('cpu')
        model.eval()
        total_images = 0
        total_time = 0.0
        
        with torch.inference_mode():
            for image, _ in test_loader:
                # Process each image individually
                for single_image in image:
                    single_image = single_image.unsqueeze(0).to('cpu')
                    
                    start_time = time.time()
                    _ = model(single_image)
                    elapsed = time.time() - start_time
                    
                    total_time += elapsed
                    total_images += 1
        
        time_per_image = (total_time / total_images) * 1000  # Convert to ms
        fps = total_images / total_time
        
        print(f"Total images processed: {total_images}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Time per image: {time_per_image:.2f}ms")
        print(f"Throughput: {fps:.2f} FPS")
        print(f"{'='*80}\n")
    
    return test_acc1, test_loss


if __name__ == "__main__":
    main()
