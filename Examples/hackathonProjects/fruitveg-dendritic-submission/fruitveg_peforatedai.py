from __future__ import print_function

import argparse
import importlib
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# PerforatedAI (repo-required functions)
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


# ────────────────────────────────────────────────────────────────
# Kaggle wheel robustness: ensure UPA.UPB exists (convert_network uses it)
# ────────────────────────────────────────────────────────────────
def _bind_upb_if_missing():
    """
    Some environments expose UPB differently.
    convert_network() can reference UPA.UPB; if absent it can crash.
    We try common module names and bind UPA.UPB if we find one.
    """
    if hasattr(UPA, "UPB"):
        return True

    candidates = [
        "perforatedai.utils_perforatedai_backpropagation",
        "perforatedai.backpropagation_perforatedai",
        "perforatedai.pb_perforatedai",
        "perforatedai.perforated_backpropagation",
        "perforatedai._utils_perforatedai_backpropagation",
    ]

    for name in candidates:
        try:
            mod = importlib.import_module(name)
            # Heuristic: module should have init function(s) used by perforated backprop
            if hasattr(mod, "initialize_pb") or hasattr(mod, "initialize"):
                UPA.UPB = mod
                print("[INFO] Bound UPA.UPB ->", name)
                return True
        except Exception:
            continue

    # If we can't bind, fail early with a clear error (better than NameError deep inside)
    raise ImportError(
        "PerforatedAI UPB backprop module could not be located in this environment. "
        "If you're on Kaggle, make sure you're using the PerforatedAI repo installation "
        "or the same setup as the mnist-example-submission."
    )


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
# PerforatedAI config (Path A style)
# ────────────────────────────────────────────────────────────────
def configure_perforatedai(args):
    # Crucial: avoid interactive confirmation / pdb stops
    # (matches your ipdb screenshot cause)
    if hasattr(GPA.pc, "set_unwrapped_modules_confirmed"):
        GPA.pc.set_unwrapped_modules_confirmed(True)

    # Keep it stable + comparable
    # (Your run log showed "try perforated backpropagation next time" so we keep PB off)
    if hasattr(GPA.pc, "set_perforated_backpropagation"):
        GPA.pc.set_perforated_backpropagation(False)

    # Dendrite policy
    if hasattr(GPA.pc, "set_max_dendrites"):
        GPA.pc.set_max_dendrites(args.max_dendrites)

    # Threshold schedule
    if hasattr(GPA.pc, "set_improvement_threshold"):
        if args.improvement_threshold == 0:
            GPA.pc.set_improvement_threshold([0.01, 0.001, 0.0001, 0])
        elif args.improvement_threshold == 1:
            GPA.pc.set_improvement_threshold([0.001, 0.0001, 0])
        else:
            GPA.pc.set_improvement_threshold([0])

    # Candidate init multiplier
    if hasattr(GPA.pc, "set_candidate_weight_initialization_multiplier"):
        GPA.pc.set_candidate_weight_initialization_multiplier(args.candidate_weight_init_mult)

    # Dendrite forward function
    if hasattr(GPA.pc, "set_pai_forward_function"):
        if args.pai_forward_function == "relu":
            GPA.pc.set_pai_forward_function(torch.relu)
        elif args.pai_forward_function == "tanh":
            GPA.pc.set_pai_forward_function(torch.tanh)
        else:
            GPA.pc.set_pai_forward_function(torch.sigmoid)

    # Noise reduction
    if hasattr(GPA.pc, "set_verbose"):
        GPA.pc.set_verbose(False)


# ────────────────────────────────────────────────────────────────
# Train / Eval (PerforatedAI-aware)
# ────────────────────────────────────────────────────────────────
def train_one_epoch(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        bs = data.size(0)
        loss_sum += loss.item() * bs
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += bs

        if batch_idx % args.log_interval == 0:
            pct = 100.0 * batch_idx / len(train_loader)
            acc = 100.0 * correct / max(1, total)
            print(
                f"Train Epoch: {epoch} [{batch_idx*bs}/{len(train_loader.dataset)} ({pct:.0f}%)] "
                f"Loss: {loss.item():.6f}  Acc: {acc:.2f}%"
            )
            if args.dry_run:
                break

    train_loss = loss_sum / max(1, total)
    train_acc = 100.0 * correct / max(1, total)

    # Optional: keep tracker aware of extra score streams
    if hasattr(GPA, "pai_tracker") and hasattr(GPA.pai_tracker, "add_extra_score"):
        GPA.pai_tracker.add_extra_score(train_acc, "train")

    return train_loss, train_acc


@torch.no_grad()
def evaluate_and_maybe_restructure(args, model, device, loader, optimizer, scheduler, split_name="validation"):
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

    print(f"\n{split_name.capitalize()} set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)\n")

    # Must call add_validation_score() per rules
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(acc, model)
    model.to(device)

    # If restructured, tracker may want to rebuild optimizer/scheduler
    if restructured and not training_complete:
        optim_args = {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
        sched_args = {"step_size": 1, "gamma": args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)
        print("[INFO] Model restructured by PerforatedAI → optimizer/scheduler reset.")

    return avg_loss, acc, model, optimizer, scheduler, training_complete, restructured


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Fruit/Veg — Dendritic Optimization (PerforatedAI Path A) on MobileNetV3-Small"
    )

    # Dataset dirs (Kaggle)
    parser.add_argument("--train-dir", type=str, default="/kaggle/input/fruit-and-vegetable-image-recognition/train")
    parser.add_argument("--val-dir", type=str, default="/kaggle/input/fruit-and-vegetable-image-recognition/validation")
    parser.add_argument("--test-dir", type=str, default="/kaggle/input/fruit-and-vegetable-image-recognition/test")

    # Training hyperparams
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)  # PerforatedAI may stop early
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # Runtime
    parser.add_argument("--no-cuda", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)

    # Outputs
    parser.add_argument("--save-model", action="store_true", default=True)
    parser.add_argument("--out-dir", type=str, default="./outputs")
    parser.add_argument("--model-name", type=str, default="fruitveg_mnv3s_pathA_gd_dendritic.pt")
    parser.add_argument("--metrics-name", type=str, default="pathA_gd_dendritic_metrics.json")
    parser.add_argument("--save-name", type=str, default="PAI")  # Graph output folder name

    # PerforatedAI knobs (keep simple; sweep later if you want)
    parser.add_argument("--max-dendrites", type=int, default=5)
    parser.add_argument("--improvement-threshold", type=int, default=1, choices=[0, 1, 2])
    parser.add_argument("--candidate-weight-init-mult", type=float, default=0.01)
    parser.add_argument("--pai-forward-function", type=str, default="sigmoid", choices=["sigmoid", "relu", "tanh"])

    # Baseline val to compute Remaining Error Reduction (RER)
    parser.add_argument("--baseline-val-acc", type=float, default=95.16)

    args = parser.parse_args()

    # Validate dirs
    for p in [args.train_dir, args.val_dir, args.test_dir]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Directory not found: {p}")

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("[INFO] device:", device)

    set_seed(args.seed)
    ensure_dir(args.out_dir)

    # Bind UPB if needed (avoid NameError inside convert_network)
    _bind_upb_if_missing()

    # Transforms (same as baseline = fair comparison)
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

    # Ensure mapping consistency
    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError("Class mapping mismatch train vs val.")
    if test_ds.class_to_idx != train_ds.class_to_idx:
        raise ValueError("Class mapping mismatch train vs test.")

    num_classes = len(train_ds.classes)
    print("[INFO] num_classes:", num_classes)
    print("[INFO] example classes:", train_ds.classes[:10], "...")

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

    # Build base model
    model, _ = build_model(num_classes=num_classes)
    model = model.to(device)

    base_params = count_params(model)
    print(f"[INFO] base trainable params: {base_params:,} ({base_params/1e6:.2f}M)")

    # Configure PerforatedAI
    configure_perforatedai(args)

    # Required by rules: initialize_pai()
    model = UPA.initialize_pai(model, save_name=args.save_name)
    model.to(device)

    # Required pattern: optimizer/scheduler via tracker
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.set_scheduler(StepLR)

    optim_args = {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
    sched_args = {"step_size": 1, "gamma": args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optim_args, sched_args)

    best_val_acc = -1.0
    history = []
    start_time = time.time()

    training_complete = False

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(args, model, device, train_loader, optimizer, epoch)

        val_loss, val_acc, model, optimizer, scheduler, training_complete, restructured = (
            evaluate_and_maybe_restructure(args, model, device, val_loader, optimizer, scheduler, "validation")
        )

        scheduler.step()

        # Track
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "restructured": bool(restructured),
                "pai_mode": str(getattr(GPA.pai_tracker, "mode", getattr(GPA.pai_tracker, "member_vars", {}).get("mode", ""))),
            }
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if args.save_model:
                out_path = os.path.join(args.out_dir, args.model_name)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "classes": train_ds.classes,
                        "class_to_idx": train_ds.class_to_idx,
                        "best_val_acc": best_val_acc,
                        "base_params": base_params,
                        "args": vars(args),
                    },
                    out_path,
                )
                print(f"[INFO] Saved best model to {out_path}")

        if args.dry_run:
            break

        if training_complete:
            print("[INFO] PerforatedAI reports training complete. Stopping early.")
            break

    # Load best ckpt
    if args.save_model:
        ckpt_path = os.path.join(args.out_dir, args.model_name)
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])

    # Final test eval
    model.eval()
    test_loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = nn.functional.cross_entropy(output, target, reduction="sum")
            test_loss_sum += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    test_loss = test_loss_sum / max(1, total)
    test_acc = 100.0 * correct / max(1, total)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({test_acc:.2f}%)\n")

    elapsed = time.time() - start_time

    # Remaining Error Reduction vs baseline val
    baseline_val_acc = float(args.baseline_val_acc)
    baseline_err = 100.0 - baseline_val_acc
    dendritic_err = 100.0 - best_val_acc
    rer = ((baseline_err - dendritic_err) / max(1e-9, baseline_err)) * 100.0

    # Save metrics
    metrics = {
        "baseline_val_acc_for_rer": baseline_val_acc,
        "best_val_acc": best_val_acc,
        "final_test_acc": test_acc,
        "final_test_loss": test_loss,
        "base_params": base_params,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60.0,
        "remaining_error_reduction_percent": rer,
        "history": history,
        "expected_pai_graph_path": f"./{args.save_name}/{args.save_name}.png",
        "required_submission_graph": "PAI/PAI.png",
    }

    metrics_path = os.path.join(args.out_dir, args.metrics_name)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[DONE] Metrics saved to: {metrics_path}")
    print(f"[DONE] Best Val Acc: {best_val_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    print(f"[DONE] Remaining Error Reduction (vs {baseline_val_acc:.2f}% val): {rer:.2f}%")
    print(f"[DONE] Time: {elapsed/60:.1f} minutes")

    print("\n[IMPORTANT] For judging, include the PerforatedAI graph image:")
    print(f"  -> ./{args.save_name}/{args.save_name}.png  (should be PAI/PAI.png)\n")


if __name__ == "__main__":
    import sys
    # Kaggle notebooks inject args; wipe them so argparse doesn't crash
    sys.argv = [""]
    main()
