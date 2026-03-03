'''

Usage:
CUDA_VISIBLE_DEVICES=1 python train_perforatedai.py --dataset cifar100 --model resnet18 --perforatedai

'''



import datetime
import os
import time
import warnings

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
import sys

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
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def get_dataset_config(dataset_name):
    """Get recommended hyperparameters for each dataset
    
    NOTE: Smaller datasets (flowers102, pets, food101) are designed for 
    transfer learning with pretrained ImageNet weights. Use --weights to 
    load pretrained models for better results.
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
            'auto_augment': 'ta_wide',
            'random_erase': 0.1,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
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
            'auto_augment': 'rand-m9-mstd0.5-inc1',
            'random_erase': 0.1,
            'mixup_alpha': 0.2,
            'cutmix_alpha': 1.0,
            'use_pretrained': False,  # Train from scratch
        },
        'flowers102': {
            'num_classes': 102,
            'image_size': 224,
            'epochs': 50,
            'batch_size': 32,
            'lr': 0.001,  # Lower LR for fine-tuning
            'lr_scheduler': 'cosineannealinglr',
            'weight_decay': 1e-4,
            'lr_warmup_epochs': 5,
            'auto_augment': None,  # Less aggressive for fine-tuning
            'random_erase': 0.0,
            'mixup_alpha': 0.0,
            'cutmix_alpha': 0.0,
            'label_smoothing': 0.1,
            'use_pretrained': True,  # Use pretrained weights
            'default_weights': 'IMAGENET1K_V1',
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
            'auto_augment': None,
            'random_erase': 0.0,
            'mixup_alpha': 0.0,
            'cutmix_alpha': 0.0,
            'use_pretrained': True,  # Use pretrained weights
            'default_weights': 'IMAGENET1K_V1',
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
            'auto_augment': None,
            'random_erase': 0.0,
            'mixup_alpha': 0.0,
            'cutmix_alpha': 0.0,
            'use_pretrained': True,  # Use pretrained weights
            'default_weights': 'IMAGENET1K_V1',
        },
    }
    return configs.get(dataset_name.lower(), configs['cifar100'])


def load_data(traindir, valdir, args):
    # Data loading code
    print(f"Loading data for dataset: {args.dataset}")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    
    # We need a default value for the variables below because args may come
    # from train_quantization.py which doesn't define them.
    auto_augment_policy = getattr(args, "auto_augment", None)
    random_erase_prob = getattr(args, "random_erase", 0.0)
    ra_magnitude = getattr(args, "ra_magnitude", None)
    augmix_severity = getattr(args, "augmix_severity", None)
    
    train_transform = presets.ClassificationPresetTrain(
        crop_size=train_crop_size,
        interpolation=interpolation,
        auto_augment_policy=auto_augment_policy,
        random_erase_prob=random_erase_prob,
        ra_magnitude=ra_magnitude,
        augmix_severity=augmix_severity,
        backend=args.backend,
        use_v2=args.use_v2,
    )
    
    val_transform = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
        use_v2=args.use_v2,
    )
    
    # Load dataset based on dataset argument
    dataset_name = args.dataset.lower()
    
    if dataset_name == 'cifar100':
        dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=True,
            download=True,
            transform=train_transform,
        )
        dataset_test = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=False,
            download=True,
            transform=val_transform,
        )
    elif dataset_name == 'stl10':
        dataset = torchvision.datasets.STL10(
            root=args.data_path,
            split='train',
            download=True,
            transform=train_transform,
        )
        dataset_test = torchvision.datasets.STL10(
            root=args.data_path,
            split='test',
            download=True,
            transform=val_transform,
        )
    elif dataset_name == 'flowers102':
        dataset = torchvision.datasets.Flowers102(
            root=args.data_path,
            split='train',
            download=True,
            transform=train_transform,
        )
        dataset_test = torchvision.datasets.Flowers102(
            root=args.data_path,
            split='test',
            download=True,
            transform=val_transform,
        )
    elif dataset_name == 'pets':
        dataset = torchvision.datasets.OxfordIIITPet(
            root=args.data_path,
            split='trainval',
            download=True,
            transform=train_transform,
        )
        dataset_test = torchvision.datasets.OxfordIIITPet(
            root=args.data_path,
            split='test',
            download=True,
            transform=val_transform,
        )
    elif dataset_name == 'food101':
        dataset = torchvision.datasets.Food101(
            root=args.data_path,
            split='train',
            download=True,
            transform=train_transform,
        )
        dataset_test = torchvision.datasets.Food101(
            root=args.data_path,
            split='test',
            download=True,
            transform=val_transform,
        )
    elif dataset_name == 'imagenet':
        # Legacy ImageNet support
        cache_path = _get_cache_path(traindir)
        if args.cache_dataset and os.path.exists(cache_path):
            print(f"Loading dataset_train from {cache_path}")
            dataset, _ = torch.load(cache_path, weights_only=False)
        else:
            dataset = torchvision.datasets.ImageFolder(
                traindir,
                train_transform,
            )
            if args.cache_dataset:
                print(f"Saving dataset_train to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset, traindir), cache_path)
        
        cache_path = _get_cache_path(valdir)
        if args.cache_dataset and os.path.exists(cache_path):
            print(f"Loading dataset_test from {cache_path}")
            dataset_test, _ = torch.load(cache_path, weights_only=False)
        else:
            if args.weights and args.test_only:
                weights = torchvision.models.get_weight(args.weights)
                preprocessing = weights.transforms(antialias=True)
                if args.backend == "tensor":
                    preprocessing = torchvision.transforms.Compose([torchvision.transforms.PILToTensor(), preprocessing])
                val_transform = preprocessing
            
            dataset_test = torchvision.datasets.ImageFolder(
                valdir,
                val_transform,
            )
            if args.cache_dataset:
                print(f"Saving dataset_test to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset_test, valdir), cache_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}. Supported: cifar100, stl10, flowers102, pets, food101, imagenet")
    
    print("Took", time.time() - st)

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


def main(args):
    if args.output_dir:
        utils.mkdir(args.output_dir)

    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True

    # For built-in datasets, we don't need train/val subdirectories
    if args.dataset.lower() in ['cifar100', 'stl10', 'flowers102', 'pets', 'food101']:
        train_dir = args.data_path
        val_dir = args.data_path
    else:
        train_dir = os.path.join(args.data_path, "train")
        val_dir = os.path.join(args.data_path, "val")
    
    dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)

    # Get number of classes
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, 'class_to_idx'):
        num_classes = len(dataset.class_to_idx)
    else:
        # Fallback: infer from dataset config
        config = get_dataset_config(args.dataset)
        num_classes = config['num_classes']
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

    if args.hf_mode == 1:
        num_classes = 1000
    print("Creating model")
    if args.perforatedai:
        from perforatedai import utils_perforatedai as UPA
        from perforatedai import globals_perforatedai as GPA
        import resnet as custom_resnet
        from perforatedai import blockwise_perforatedai as BPA
        from perforatedai import clean_perforatedai as CPA
        for i in range(4):
            GPA.pc.append_module_ids_to_track(['.layer'+str(i+1)])
        GPA.pc.append_module_ids_to_track(['.b1'])
        GPA.pc.append_module_ids_to_track(['.fc', '.conv1', '.bn1'])
        GPA.pc.append_module_names_to_convert(["BasicBlock", "Bottleneck"])
        
        if args.hf_mode == 2:
            GPA.pc.append_module_ids_to_track(['.fc'])
            # Load from HuggingFace
            if not args.hf_repo_id:
                raise ValueError("--hf-repo-id is required when using --hf-mode 2")
            print(f"Loading model from HuggingFace: {args.hf_repo_id}")
            # Create base model architecture
            base_model = torchvision.models.get_model(args.model, weights=None, num_classes=1000)
            model = custom_resnet.ResNetPAI(base_model)
            #Dont call initialize_pai when loading from hf
            #model = UPA.initialize_pai(model)
            # Load from HuggingFace
            model = UPA.from_hf_pretrained(model, args.hf_repo_id)
            print(f"Successfully loaded model from HuggingFace")
        else:
            # Normal loading (mode 0 or 1)
            # Create model with 1000 classes if keeping FC, otherwise use target classes
            model = torchvision.models.get_model(args.model, weights='IMAGENET1K_V1', num_classes=1000)
            print(f"Loaded ImageNet model with 1000 classes for FC feature extraction")
        
            model = custom_resnet.ResNetPAI(model)
            model = UPA.initialize_pai(model)
            pretrained_path = args.pretrained_path
            if not pretrained_path.endswith(".pt"):
                raise ValueError(f"--pretrained-path must end with .pt (got: {pretrained_path})")
            pretrained_dir, pretrained_file = os.path.split(pretrained_path)
            if not pretrained_dir:
                pretrained_dir = "."
            pretrained_name = os.path.splitext(pretrained_file)[0]
            model = UPA.load_system(model, pretrained_dir, pretrained_name, True)
            model = BPA.blockwise_network(model)
            model  = CPA.refresh_net(model)
            
            

        print("model has total params: ", sum([p.numel() for p in model.parameters()]))

        # Apply keep-fc-as-features if requested
        if args.keep_fc_as_features:
            if hasattr(model, 'fc'):
                original_fc = model.fc
                fc_out_features = original_fc.out_features if hasattr(original_fc, 'out_features') else 1000
                model.fc = nn.Sequential(
                    original_fc,
                    nn.ReLU(),
                    nn.Linear(fc_out_features, num_classes)
                )
                # All parameters are trainable
                print(f"Kept custom FC as trainable feature extractor ({fc_out_features}D), added adapter layer for {num_classes} classes")
            else:
                print("Warning: --keep-fc-as-features specified but model has no 'fc' layer")
        elif(args.hf_mode != 1):
            # Replace final layer with correct number of classes
            if hasattr(model, 'fc'):
                # ResNet, RegNet, etc.
                if(args.hf_mode == 2):
                    in_features = model.fc.main_module.in_features
                else:
                    in_features = model.fc.layer_array[0].in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'classifier'):
                # VGG, MobileNet, etc.
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'head'):
                # Vision Transformer, etc.
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Cannot adapt model {args.model} - unknown classifier layer")
            print(f"Replaced final layer for {num_classes} classes")
    # Handle pretrained weights: load with default classes, then replace final layer
    elif args.weights:
        model = torchvision.models.get_model(args.model, weights=args.weights)
        
        if args.keep_fc_as_features:
            # Keep pretrained FC as feature extractor
            if hasattr(model, 'fc'):
                original_fc = model.fc
                fc_out_features = original_fc.out_features
                model.fc = nn.Sequential(
                    original_fc,
                    nn.ReLU(),
                    nn.Linear(fc_out_features, num_classes)
                )
                # All parameters are trainable
                print(f"Kept pretrained FC as trainable feature extractor ({fc_out_features}D), added adapter layer for {num_classes} classes")
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    original_classifier = model.classifier
                    out_features = original_classifier[-1].out_features
                    model.classifier = nn.Sequential(
                        original_classifier,
                        nn.ReLU(),
                        nn.Linear(out_features, num_classes)
                    )
                    # All parameters are trainable
                    print(f"Kept pretrained classifier as trainable feature extractor ({out_features}D), added adapter layer for {num_classes} classes")
            elif hasattr(model, 'head'):
                original_head = model.head
                out_features = original_head.out_features
                model.head = nn.Sequential(
                    original_head,
                    nn.ReLU(),
                    nn.Linear(out_features, num_classes)
                )
                # All parameters are trainable
                print(f"Kept pretrained head as trainable feature extractor ({out_features}D), added adapter layer for {num_classes} classes")
            else:
                raise ValueError(f"Cannot adapt model {args.model} - unknown classifier layer")
        else:
            # Replace final layer with correct number of classes
            if hasattr(model, 'fc'):
                # ResNet, RegNet, etc.
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'classifier'):
                # VGG, MobileNet, etc.
                if isinstance(model.classifier, nn.Sequential):
                    in_features = model.classifier[-1].in_features
                    model.classifier[-1] = nn.Linear(in_features, num_classes)
                else:
                    in_features = model.classifier.in_features
                    model.classifier = nn.Linear(in_features, num_classes)
            elif hasattr(model, 'head'):
                # Vision Transformer, etc.
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes)
            else:
                raise ValueError(f"Cannot adapt model {args.model} - unknown classifier layer")
            print(f"Replaced final layer for {num_classes} classes")
    else:
        # No pretrained weights, create model from scratch with correct num_classes
        model = torchvision.models.get_model(args.model, weights=None, num_classes=num_classes)
    



    # If mode 1, upload to HuggingFace after loading
    if args.hf_mode == 1:
        GPA.pc.append_module_ids_to_track(['.fc', '.conv1', '.bn1'])
        GPA.pc.set_verbose(False)
        GPA.pc.set_silent(True)
        GPA.pc.set_unwrapped_modules_confirmed(True)
        if not args.hf_repo_id:
            raise ValueError("--hf-repo-id is required when using --hf-mode 1")
        print(f"\nUploading model to HuggingFace: {args.hf_repo_id}")
        from perforatedai import modules_perforatedai as MPA

        correctingb1 = True
        if correctingb1:
            from collections import OrderedDict
            model.conv1 = MPA.TrackedNeuronModule(model.b1.model[0], "model.conv1")
            model.bn1 = MPA.TrackedNeuronModule(model.b1.model[1], "model.bn1")
            del model.b1
            priority = ['conv1', 'bn1']
            reordered = priority + [k for k in model._modules if k not in priority]
            model._modules = OrderedDict((k, model._modules[k]) for k in reordered)

        UPA.upload_to_huggingface(
            model, 
            args.hf_repo_id,
            license="apache-2.0",
            pipeline_tag="image-classification",
            tags=["perforated-ai", args.dataset, args.model]
        )
        print(f"Successfully uploaded model to HuggingFace")
        print(model)
        import pdb; pdb.set_trace()
        sys.exit(0)

    print("After fc replacement model has total params: ", sum([p.numel() for p in model.parameters()]))
    
    model.to(device)
    
    # Check parameter requires_grad status
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    frozen_layers = []
    
    print("\n" + "="*80)
    print("PARAMETER REQUIRES_GRAD STATUS CHECK")
    print("="*80)
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        else:
            frozen_params += param.numel()
            frozen_layers.append(name)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
    if frozen_layers:
        print(f"\nFrozen layers ({len(frozen_layers)}):")
        for layer in frozen_layers[:10]:  # Show first 10
            print(f"  - {layer}")
        if len(frozen_layers) > 10:
            print(f"  ... and {len(frozen_layers) - 10} more")
    else:
        print("\n✓ All parameters are trainable (requires_grad=True)")
    
    print("="*80 + "\n")

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )

    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

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
            acc1, loss = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            acc1, loss = evaluate(model, criterion, data_loader_test, device=device)
        return acc1, loss

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler)
        lr_scheduler.step()
        test_acc1, test_loss = evaluate(model, criterion, data_loader_test, device=device)
        if model_ema:
            test_acc1, test_loss = evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
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

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    
    # Return final results
    return test_acc1, test_loss


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument(
        "--dataset",
        default="cifar100",
        type=str,
        choices=['cifar100', 'stl10', 'flowers102', 'pets', 'food101', 'imagenet'],
        help="dataset to train on (default: cifar100)"
    )
    parser.add_argument(
        "--data-path",
        default="./data",
        type=str,
        help="dataset path (default: ./data)"
    )
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=None, type=int, help="images per gpu, the total batch size is $NGPU x batch_size (default: dataset-specific)"
    )
    parser.add_argument("--epochs", default=None, type=int, metavar="N", help="number of total epochs to run (default: dataset-specific)")
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=None, type=float, help="initial learning rate (default: dataset-specific)")
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
        "--label-smoothing", default=None, type=float, help="label smoothing (default: dataset-specific)", dest="label_smoothing"
    )
    parser.add_argument("--mixup-alpha", default=None, type=float, help="mixup alpha (default: dataset-specific)")
    parser.add_argument("--cutmix-alpha", default=None, type=float, help="cutmix alpha (default: dataset-specific)")
    parser.add_argument("--lr-scheduler", default=None, type=str, help="the lr scheduler (default: dataset-specific)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument(
        "--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)"
    )
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("--lr-gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
    parser.add_argument("--lr-min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")
    parser.add_argument("--print-freq", default=10, type=int, help="print frequency")
    parser.add_argument("--output-dir", default=None, type=str, help="path to save outputs (default: None, no checkpoints saved)")
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
    parser.add_argument("--auto-augment", default=None, type=str, help="auto augment policy (default: dataset-specific)")
    parser.add_argument("--ra-magnitude", default=9, type=int, help="magnitude of auto augment policy")
    parser.add_argument("--augmix-severity", default=3, type=int, help="severity of augmix policy")
    parser.add_argument("--random-erase", default=None, type=float, help="random erasing probability (default: dataset-specific)")

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
    parser.add_argument(
        "--val-resize-size", default=None, type=int, help="the resize size used for validation (default: dataset-specific)"
    )
    parser.add_argument(
        "--val-crop-size", default=None, type=int, help="the central crop size used for validation (default: dataset-specific)"
    )
    parser.add_argument(
        "--train-crop-size", default=None, type=int, help="the random crop size used for training (default: dataset-specific)"
    )
    parser.add_argument("--clip-grad-norm", default=None, type=float, help="the maximum gradient norm (default None)")
    parser.add_argument("--ra-sampler", action="store_true", help="whether to use Repeated Augmentation in training")
    parser.add_argument(
        "--ra-reps", default=3, type=int, help="number of repetitions for Repeated Augmentation (default: 3)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--perforatedai", action="store_true", help="Use PerforatedAI model")
    parser.add_argument("--keep-fc-as-features", default=0, type=int, help="Keep FC layer as feature extractor and add adapter layer on top (default: True)")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    parser.add_argument("--use-v2", action="store_true", help="Use V2 transforms")
    parser.add_argument("--hf-mode", default=0, type=int, choices=[0, 1, 2], help="HuggingFace mode: 0=normal (default), 1=upload after loading, 2=load from HuggingFace")
    parser.add_argument("--hf-repo-id", default=None, type=str, help="HuggingFace repository ID (e.g., 'username/model-name') for upload (mode 1) or download (mode 2)")
    parser.add_argument(
        "--pretrained-path",
        default="./pretrained/best_model.pt",
        type=str,
        help="Path to a local .pt file used by load_system (default: ./pretrained/best_model.pt)",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    # Apply dataset-specific defaults if not explicitly set
    config = get_dataset_config(args.dataset)
    
    # Auto-load pretrained weights if recommended for dataset
    if args.weights is None and config.get('use_pretrained', False):
        model_name_upper = args.model.upper()
        default_weights_name = config.get('default_weights', 'IMAGENET1K_V1')
        args.weights = f"{model_name_upper.replace('RESNET', 'ResNet')}_Weights.{default_weights_name}"
        print(f"\n⚠️  Auto-loading pretrained weights: {args.weights}")
        print(f"   Dataset '{args.dataset}' requires transfer learning for good results.")
        print(f"   To train from scratch, use: --weights=None\n")
    
    if args.batch_size is None:
        args.batch_size = config.get('batch_size', 32)
    if args.epochs is None:
        args.epochs = config.get('epochs', 90)
    if args.lr is None:
        args.lr = config.get('lr', 0.1)
    if args.lr_scheduler is None:
        args.lr_scheduler = config.get('lr_scheduler', 'steplr')
    if args.lr_warmup_epochs == 0:
        # Allow config to override warmup epochs
        args.lr_warmup_epochs = config.get('lr_warmup_epochs', 0)
    if args.mixup_alpha is None:
        args.mixup_alpha = config.get('mixup_alpha', 0.0)
    if args.cutmix_alpha is None:
        args.cutmix_alpha = config.get('cutmix_alpha', 0.0)
    if args.random_erase is None:
        args.random_erase = config.get('random_erase', 0.0)
    if args.auto_augment is None:
        args.auto_augment = config.get('auto_augment', None)
    if args.label_smoothing is None:
        args.label_smoothing = config.get('label_smoothing', 0.0)
    
    # Set image sizes based on dataset
    img_size = config.get('image_size', 224)
    if args.train_crop_size is None:
        args.train_crop_size = img_size
    if args.val_crop_size is None:
        args.val_crop_size = img_size
    if args.val_resize_size is None:
        # For small images (CIFAR), no resize needed
        if img_size <= 32:
            args.val_resize_size = img_size
        else:
            args.val_resize_size = int(img_size * 256 / 224)
    
    print(f"\n=== Dataset Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Pretrained weights: {args.weights if args.weights else 'None (training from scratch)'}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"LR scheduler: {args.lr_scheduler}")
    print(f"LR warmup epochs: {args.lr_warmup_epochs}")
    print(f"Image size: {img_size}")
    print(f"Mixup alpha: {args.mixup_alpha}")
    print(f"Cutmix alpha: {args.cutmix_alpha}")
    print(f"Auto augment: {args.auto_augment}")
    print(f"Random erase: {args.random_erase}")
    print(f"Label smoothing: {args.label_smoothing}")
    print(f"============================\n")
    
    # Run training 7 times and collect results
    all_results = []
    for run_num in range(1, 8):
        print(f"\n{'='*80}")
        print(f"STARTING RUN {run_num}/7")
        print(f"{'='*80}\n")
        
        acc1, loss = main(args)
        all_results.append((acc1, loss))
        
        print(f"\n{'='*80}")
        print(f"COMPLETED RUN {run_num}/7 - Acc@1: {acc1:.3f}, Loss: {loss:.4f}")
        print(f"{'='*80}\n")
    
    # Print results in CSV format
    print(f"\n\n{'='*80}")
    print(f"ALL 7 RUNS COMPLETE - CSV OUTPUT")
    print(f"{'='*80}\n")
    
    # Calculate statistics
    accs = [acc1 for acc1, _ in all_results]
    losses = [loss for _, loss in all_results]
    mean_acc = sum(accs)/len(accs)
    std_acc = (sum((x - mean_acc)**2 for x in accs) / len(accs))**0.5
    mean_loss = sum(losses)/len(losses)
    std_loss = (sum((x - mean_loss)**2 for x in losses) / len(losses))**0.5
    
    # CSV Header
    print("Model,Dataset,Run,Acc1,Loss")
    
    # Add "pai-" prefix to model name
    model_name = f"pai-{args.model}"
    
    # Individual runs
    for i, (acc1, loss) in enumerate(all_results, 1):
        print(f"{model_name},{args.dataset},{i},{acc1:.3f},{loss:.4f}")
    
    # Statistics rows
    print(f"{model_name},{args.dataset},Mean,{mean_acc:.3f},{mean_loss:.4f}")
    print(f"{model_name},{args.dataset},Std,{std_acc:.3f},{std_loss:.4f}")
    print(f"{model_name},{args.dataset},Min,{min(accs):.3f},{min(losses):.4f}")
    print(f"{model_name},{args.dataset},Max,{max(accs):.3f},{max(losses):.4f}")
    print(f"{model_name},{args.dataset},Range,{max(accs) - min(accs):.3f},{max(losses) - min(losses):.4f}")
    
    print(f"\n{'='*80}\n")