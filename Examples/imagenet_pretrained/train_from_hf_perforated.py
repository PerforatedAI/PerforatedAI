'''
Baseline training script — no pruning, no LR restart, no patience logic.
Loads models the same way as train_from_hf_prune.py and fine-tunes straight
through for args.epochs epochs with a single cosine-annealing schedule.

Usage:
python train_from_hf_baseline.py --hf-repo-id tv/mobilenet_v3_large --dataset food101 --model-type mobilenet --device cuda:0 --data-path ./data
python train_from_hf_baseline.py --hf-repo-id tv/mnasnet0_75       --dataset food101 --model-type mobilenet --device cuda:0 --data-path ./data
python train_from_hf_baseline.py --hf-repo-id tv/efficientnet_b1   --dataset food101 --model-type mobilenet --device cuda:1 --data-path ./data
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


from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

# ---------------------------------------------------------------------------
# Dataset / model-type configs
# ---------------------------------------------------------------------------

def get_dataset_config(dataset_name):
    """Get recommended hyperparameters for each dataset."""
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
            'use_pretrained': False,
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
            'use_pretrained': False,
        },
        'flowers102': {
            'num_classes': 102,
            'image_size': 224,
            'epochs': 200,
            'batch_size': 32,
            'lr': 0.001,
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 5,
            'label_smoothing': 0.1,
            'use_pretrained': True,
        },
        'pets': {
            'num_classes': 37,
            'image_size': 224,
            'epochs': 50,
            'batch_size': 32,
            'lr': 0.001,
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 5,
            'label_smoothing': 0.0,
            'use_pretrained': True,
        },
        'food101': {
            'num_classes': 101,
            'image_size': 224,
            'epochs': 30,
            'batch_size': 64,
            'lr': 0.001,
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 5,
            'label_smoothing': 0.0,
            'use_pretrained': True,
        },
    }
    return configs.get(dataset_name.lower(), configs['flowers102'])


# ---------------------------------------------------------------------------
# Model-size helpers (kept for reporting purposes)
# ---------------------------------------------------------------------------

def get_model_size(model):
    """Return number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    for i, (image, target) in enumerate(data_loader):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        output = model(image)
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

    return metric_logger.meters["acc1"].global_avg


def evaluate(model, criterion, data_loader, device, print_freq=100):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    metric_logger.synchronize_between_processes()
    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f}  Acc@5 {metric_logger.acc5.global_avg:.3f}")
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def measure_inference_latency(model, data_loader, device, num_batches=50):
    """Measure inference FPS and latency."""
    model.eval()
    latencies = []

    # Warm up
    with torch.inference_mode():
        for i, (images, _) in enumerate(data_loader):
            if i >= 3:
                break
            images = images.to(device)
            _ = model(images)

    # Measure
    with torch.inference_mode():
        for i, (images, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            images = images.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

    if not latencies:
        return {'fps': 0, 'mean_latency_ms': 0, 'p95_latency_ms': 0}

    batch_size = next(iter(data_loader))[0].shape[0]
    mean_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(0.95 * len(latencies))]

    return {
        'fps': batch_size / mean_latency,
        'mean_latency_ms': mean_latency * 1000,
        'p95_latency_ms': p95_latency * 1000,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(dataset_name, data_path, batch_size, workers=16):
    """Load dataset with appropriate transforms for transfer learning."""
    config = get_dataset_config(dataset_name)
    image_size = config['image_size']

    # Training transforms with augmentation
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Test transforms (no augmentation)
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(int(image_size * 256 / 224), interpolation=InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(image_size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset_name_lower = dataset_name.lower()

    if dataset_name_lower == 'flowers102':
        dataset_train = torchvision.datasets.Flowers102(root=data_path, split='train',
                                                        transform=train_transform, download=True)
        dataset_test  = torchvision.datasets.Flowers102(root=data_path, split='test',
                                                        transform=test_transform, download=True)
    elif dataset_name_lower == 'pets':
        dataset_train = torchvision.datasets.OxfordIIITPet(root=data_path, split='trainval',
                                                            transform=train_transform, download=True)
        dataset_test  = torchvision.datasets.OxfordIIITPet(root=data_path, split='test',
                                                            transform=test_transform, download=True)
    elif dataset_name_lower == 'food101':
        dataset_train = torchvision.datasets.Food101(root=data_path, split='train',
                                                     transform=train_transform, download=True)
        dataset_test  = torchvision.datasets.Food101(root=data_path, split='test',
                                                     transform=test_transform, download=True)
    elif dataset_name_lower == 'cifar100':
        dataset_train = torchvision.datasets.CIFAR100(root=data_path, train=True,
                                                      transform=train_transform, download=True)
        dataset_test  = torchvision.datasets.CIFAR100(root=data_path, train=False,
                                                      transform=test_transform, download=True)
    elif dataset_name_lower == 'stl10':
        dataset_train = torchvision.datasets.STL10(root=data_path, split='train',
                                                   transform=train_transform, download=True)
        dataset_test  = torchvision.datasets.STL10(root=data_path, split='test',
                                                   transform=test_transform, download=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True
    )

    return train_loader, test_loader, config['num_classes']


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_from_hf(hf_repo_id, num_classes, model_type='resnet'):
    """Load model from HuggingFace (or torchvision) and adapt for num_classes."""

    if model_type == 'mobilenet':
        model_name = hf_repo_id.split('/')[-1].replace('-', '_')
        print(f"\nLoading model directly from torchvision: {model_name}")
        model = torchvision.models.get_model(model_name, weights='DEFAULT')
        print(f"Successfully loaded torchvision model")
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        print(f"Replaced classifier layer for {num_classes} classes")
        return model

    print(f"\nLoading model from HuggingFace: {hf_repo_id}")

    if 'perforated' in hf_repo_id.lower():
        from perforatedai import utils_perforatedai as UPA
        from perforatedai import globals_perforatedai as GPA
        from perforatedai import library_perforatedai as LPA
        base_model = torchvision.models.get_model('resnet18', weights=None, num_classes=1000)
        model = LPA.ResNetPAIPreFC(base_model)
        model = UPA.from_hf_pretrained(model, hf_repo_id)
        print(f"Successfully loaded perforated model from HuggingFace")
    else:
        try:
            from transformers import AutoModelForImageClassification
            model = AutoModelForImageClassification.from_pretrained(hf_repo_id)
            print(f"Successfully loaded transformers model from HuggingFace")
        except Exception as e:
            print(f"Failed to load as transformers model: {e}")
            model_name = hf_repo_id.split('/')[-1].replace('-', '')
            print(f"Attempting to load as torchvision model: {model_name}")
            model = torchvision.models.get_model(model_name, weights='IMAGENET1K_V1')
            print(f"Successfully loaded torchvision model")

    if hasattr(model, 'fc'):
        if hasattr(model.fc, 'main_module'):
            in_features = model.fc.main_module.in_features
        else:
            in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        print(f"Replaced fc layer for {num_classes} classes")
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
        print(f"Replaced classifier layer for {num_classes} classes")
    else:
        raise ValueError(f"Cannot adapt model — unknown classifier layer")

    return model


# ---------------------------------------------------------------------------
# Training run (no pruning)
# ---------------------------------------------------------------------------

def train_single_run(args, train_loader, test_loader, num_classes):
    """Straight fine-tuning for args.epochs with cosine-annealing LR."""
    device = torch.device(args.device)

    model = load_model_from_hf(args.hf_repo_id, num_classes, model_type=args.model_type)
    model = model.to(device)
    GPA.pc.append_module_names_to_track(['Conv2dNormActivation', 'InvertedResidual']) 
    #GPA.pc.append_module_ids_to_track(['.classifier.0'])
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_n_epochs_to_switch(25)
    model = UPA.initialize_pai(model, save_name="PAI-" + (args.hf_repo_id).split('/')[-1])  # Initialize PAI state for potential future pruning steps

    import pdb; pdb.set_trace()

    total_params = get_model_size(model)
    print(f"\nModel size: {total_params:,} parameters")

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
    GPA.pai_tracker.set_optimizer_instance(optimizer)

    print(f"\nStarting baseline training for {args.epochs} epochs (no pruning)...\n")
    print("Epoch | Train Acc@1 | Test Acc@1 |     LR")
    print("-" * 55)

    start_time = time.time()
    best_acc1 = 0.0
    best_epoch = 0

    epoch = -1
    while True:
        epoch += 1
        train_acc1 = train_one_epoch(model, criterion, optimizer, train_loader, device, epoch, args.print_freq)
        lr_scheduler.step()
        test_acc1, _ = evaluate(model, criterion, test_loader, device)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"{epoch+1:5d} | {train_acc1:11.3f} | {test_acc1:10.3f} | {current_lr:10.6f}")

        if test_acc1 > best_acc1:
            best_acc1 = test_acc1
            best_epoch = epoch + 1
        GPA.pai_tracker.add_extra_score(train_acc1, "train_acc1")
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(test_acc1,
        model) # .module if its a dataParallel
        model.to(device)
        if(training_complete):
            break
        elif(restructured): 
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


    total_time_str = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"\nTraining complete! Total time: {total_time_str}")
    print(f"Best Test Accuracy: {best_acc1:.3f}% (epoch {best_epoch})")

    return best_acc1, best_epoch, model


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Baseline fine-tuning (no pruning)")
    parser.add_argument("--hf-repo-id", required=True, type=str,
                        help="HuggingFace repo ID or torchvision model path (e.g. tv/mobilenet_v3_large)")
    parser.add_argument("--dataset", default="food101", type=str,
                        choices=['flowers102', 'pets', 'food101', 'cifar100', 'stl10'],
                        help="Dataset to train on (default: food101)")
    parser.add_argument("--data-path", default="./data", type=str, help="Dataset root path")
    parser.add_argument("--batch-size", default=None, type=int, help="Batch size (default: dataset-specific)")
    parser.add_argument("--epochs", default=None, type=int, help="Number of epochs (default: dataset-specific)")
    parser.add_argument("--lr", default=None, type=float, help="Learning rate (default: dataset-specific)")
    parser.add_argument("--lr-warmup-epochs", default=None, type=int, help="Warmup epochs (default: dataset-specific)")
    parser.add_argument("--label-smoothing", default=None, type=float, help="Label smoothing (default: dataset-specific)")
    parser.add_argument("--workers", default=16, type=int, help="Data loading workers")
    parser.add_argument("--device", default="cuda", type=str, help="Device (cuda or cpu)")
    parser.add_argument("--print-freq", default=10, type=int, help="Print frequency")
    parser.add_argument("--model-type", default="resnet", type=str,
                        choices=['resnet', 'mobilenet'],
                        help=("Architecture family. "
                              "'mobilenet': loads directly from torchvision (e.g. tv/mobilenet_v3_large, "
                              "tv/mnasnet0_75, tv/efficientnet_b1). "
                              "'resnet': loads from HuggingFace."))

    args = parser.parse_args()

    # Apply dataset-specific defaults where not overridden
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
        args.label_smoothing = config.get('label_smoothing', 0.0)

    print(f"\n{'='*80}")
    print(f"Baseline Training Configuration (no pruning)")
    print(f"{'='*80}")
    print(f"HuggingFace Repo : {args.hf_repo_id}")
    print(f"Dataset          : {args.dataset}")
    print(f"Epochs           : {args.epochs}")
    print(f"Batch size       : {args.batch_size}")
    print(f"Learning rate    : {args.lr}")
    print(f"LR warmup epochs : {args.lr_warmup_epochs}")
    print(f"Label smoothing  : {args.label_smoothing}")
    print(f"Device           : {args.device}")
    print(f"Model type       : {args.model_type}")
    print(f"{'='*80}\n")

    train_loader, test_loader, num_classes = load_dataset(
        args.dataset, args.data_path, args.batch_size, args.workers
    )

    print(f"\n{'='*80}")
    print(f"STARTING TRAINING")
    print(f"{'='*80}\n")

    best_acc1, best_epoch, model = train_single_run(args, train_loader, test_loader, num_classes)

    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Acc@1 : {best_acc1:.3f}%")
    print(f"Best Epoch : {best_epoch}")
    print(f"{'='*80}\n")

    # Measure inference latency
    print(f"\n{'='*80}")
    print(f"MEASURING LATENCY")
    print(f"{'='*80}\n")
    device = torch.device(args.device)
    latency_results = measure_inference_latency(model, test_loader, device)

    # CSV summary
    model_name = args.hf_repo_id.split('/')[-1]
    print(f"\n{'='*80}")
    print(f"RESULTS - CSV FORMAT")
    print(f"{'='*80}\n")
    print("Model,Dataset,BestAcc1,BestEpoch,FPS,MeanLatency_ms,P95Latency_ms")
    print(f"{model_name},{args.dataset},{best_acc1:.3f},{best_epoch},"
          f"{latency_results['fps']:.2f},{latency_results['mean_latency_ms']:.2f},"
          f"{latency_results['p95_latency_ms']:.2f}")
    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
