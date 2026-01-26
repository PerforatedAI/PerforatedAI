from __future__ import print_function

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


# ────────────────────────────────────────────────────────────────
# Utils
# ────────────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def accuracy_top1(logits, targets) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ────────────────────────────────────────────────────────────────
# Train / Eval
# ────────────────────────────────────────────────────────────────
def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    seen = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        bs = data.size(0)
        seen += bs
        running_loss += loss.item() * bs
        running_acc += accuracy_top1(output, target) * bs

        if batch_idx % args.log_interval == 0:
            pct = 100.0 * batch_idx / len(train_loader)
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {:.2f}%".format(
                    epoch,
                    batch_idx * bs,
                    len(train_loader.dataset),
                    pct,
                    loss.item(),
                    100.0 * accuracy_top1(output, target),
                )
            )
            if args.dry_run:
                break

    epoch_loss = running_loss / max(1, seen)
    epoch_acc = 100.0 * (running_acc / max(1, seen))
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, device, loader, split_name="val"):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0

    for data, target in loader:
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(data)
        loss = nn.functional.cross_entropy(output, target, reduction="sum")
        loss_sum += loss.item()

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

    avg_loss = loss_sum / max(1, total)
    acc = 100.0 * correct / max(1, total)

    print(
        "\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            split_name.capitalize(), avg_loss, correct, total, acc
        )
    )
    return avg_loss, acc


# ────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────
def build_model(num_classes: int):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_f = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_f, num_classes)
    return model, weights


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fruit/Veg Baseline (MobileNetV3-Small) — No Dendrites"
    )

    # Kaggle dataset structure (already split)
    parser.add_argument(
        "--train-dir",
        type=str,
        default="/kaggle/input/fruit-and-vegetable-image-recognition/train",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        default="/kaggle/input/fruit-and-vegetable-image-recognition/validation",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        default="/kaggle/input/fruit-and-vegetable-image-recognition/test",
    )

    # Training hyperparams
    parser.add_argument("--batch-size", type=int, default=64, metavar="N")
    parser.add_argument("--test-batch-size", type=int, default=128, metavar="N")
    parser.add_argument("--epochs", type=int, default=8, metavar="N")
    parser.add_argument("--lr", type=float, default=3e-4, metavar="LR")
    parser.add_argument("--gamma", type=float, default=0.9, metavar="M")
    parser.add_argument("--weight-decay", type=float, default=1e-4, metavar="WD")

    # Runtime
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42, metavar="S")
    parser.add_argument("--log-interval", type=int, default=20, metavar="N")
    parser.add_argument("--num-workers", type=int, default=2)

    # Outputs
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default="./outputs")
    parser.add_argument("--model-name", type=str, default="fruitveg_mnv3s_baseline.pt")
    parser.add_argument("--metrics-name", type=str, default="baseline_metrics.json")

    args = parser.parse_args()

    # Validate dataset dirs
    for p in [args.train_dir, args.val_dir, args.test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Directory not found: {p}")

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[INFO] device:", device)

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Transforms
    _, weights = build_model(num_classes=2)

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    eval_transform = weights.transforms()

    train_ds = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(args.val_dir, transform=eval_transform)
    test_ds = datasets.ImageFolder(args.test_dir, transform=eval_transform)

    # Ensure class mapping consistency
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError(
            "Class mapping mismatch train vs validation.\n"
            f"train: {train_ds.class_to_idx}\nval: {val_ds.class_to_idx}"
        )
    if test_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError(
            "Class mapping mismatch train vs test.\n"
            f"train: {train_ds.class_to_idx}\ntest: {test_ds.class_to_idx}"
        )

    num_classes = len(train_ds.classes)
    print("[INFO] num_classes:", num_classes)
    print("[INFO] classes:", train_ds.classes)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    # Build model
    model, _ = build_model(num_classes=num_classes)
    model = model.to(device)

    params = count_params(model)
    print(f"[INFO] trainable params: {params:,} ({params/1e6:.2f}M)")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    best_val_acc = -1.0
    best_state = None
    history = []

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(args, model, device, train_loader, optimizer, epoch)
        val_loss, val_acc = evaluate(model, device, val_loader, split_name="validation")
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "class_to_idx": train_ds.class_to_idx,
                "best_val_acc": best_val_acc,
                "params": params,
                "args": vars(args),
            }
            if args.save_model:
                torch.save(best_state, os.path.join(args.out_dir, args.model_name))
                print(f"[INFO] Saved best model to {os.path.join(args.out_dir, args.model_name)}")

        if args.dry_run:
            break

    # Evaluate best checkpoint on test
    if best_state is not None and args.save_model:
        ckpt_path = os.path.join(args.out_dir, args.model_name)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

    test_loss, test_acc = evaluate(model, device, test_loader, split_name="test")

    elapsed = time.time() - start_time

    metrics = {
        "best_val_acc": best_val_acc,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "params": params,
        "params_million": params / 1e6,
        "elapsed_seconds": elapsed,
        "history": history,
    }

    metrics_path = os.path.join(args.out_dir, args.metrics_name)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] Metrics saved to: {metrics_path}")
    print(f"[DONE] Best Val Acc: {best_val_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    print(f"[DONE] Time: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    import sys
    # Kaggle notebooks inject args; wipe them so argparse doesn't crash
    sys.argv = [""]
    main()
