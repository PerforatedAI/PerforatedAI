import os
import sys
import torch
import wandb

# ========================
# PATH SETUP
# ========================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

import bootstrap

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, Activations, AsDiscrete

from src.data.dataset_loader import get_dataloaders
from src.models.unet_baseline import get_unet

# ========================
# CONFIG
# ========================
PROJECT_NAME = "Perforated-MONAI"
RUN_NAME = "baseline_unet_converged"

DATA_DIR = "datasets/monai"

LR = 2e-4

PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 3
DEVICE = "cuda"

# Convergence control
MAX_EPOCHS = 150
PATIENCE = 12
MIN_DELTA = 0.002


def flatten_if_needed(x):
    # [B,S,C,D,H,W] → [B*S,C,D,H,W]
    if x.ndim == 6:
        return x.flatten(0, 1)
    return x


def main():
    os.makedirs("checkpoints/baseline", exist_ok=True)

    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)

    # ========================
    # MODEL
    # ========================
    model = get_unet(out_channels=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    # ========================
    # DATA
    # ========================
    train_loader, val_loader = get_dataloaders(DATA_DIR)
    steps_per_epoch = len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=MAX_EPOCHS * steps_per_epoch
    )

    # ========================
    # LOSS & METRICS
    # ========================
    loss_fn = DiceCELoss(
        sigmoid=True,
        to_onehot_y=False,
        include_background=False,
        lambda_dice=1.0,
        lambda_ce=0.0,   # 🔥 IMPORTANT
    )




    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=False,
    )

    scaler = torch.amp.GradScaler("cuda")

    # ========================
    # TRAINING STATE
    # ========================
    best_dice = -1.0
    epochs_no_improve = 0

    # ========================
    # TRAIN LOOP
    # ========================
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            inputs = flatten_if_needed(batch["image"]).to(DEVICE, non_blocking=True)
            labels = flatten_if_needed(batch["label"]).to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()  # ✅ correct: per-step

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # ========================
        # VALIDATION
        # ========================
        model.eval()
        dice_metric.reset()
        post_pred = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5),
        ])

        post_label = AsDiscrete(threshold=0.5)
        with torch.no_grad():
            for batch in val_loader:
                inputs = flatten_if_needed(batch["image"]).to(DEVICE)
                labels = flatten_if_needed(batch["label"]).to(DEVICE)

                outputs = sliding_window_inference(
                    inputs,
                    PATCH_SIZE,
                    sw_batch_size=2,
                    predictor=model,
                )

                outputs = post_pred(outputs)
                labels = post_label(labels)

                dice_metric(y_pred=outputs, y=labels)


        dice_vals = dice_metric.aggregate()
        mean_dice = dice_vals.mean().item()

        log_dict = {
            "epoch": epoch,
            "train/loss": epoch_loss,
            "val/dice_mean": mean_dice,
            "lr": scheduler.get_last_lr()[0],
        }

        # Optional per-class logging (safe)
        if dice_vals.numel() > 0:
            log_dict["dice/class_0"] = dice_vals[0].item()
        if dice_vals.numel() > 1:
            log_dict["dice/class_1"] = dice_vals[1].item()
        if dice_vals.numel() > 2:
            log_dict["dice/class_2"] = dice_vals[2].item()

        wandb.log(log_dict)


        print(
            f"Epoch {epoch:03d} | "
            f"Loss {epoch_loss:.4f} | "
            f"Val Dice {mean_dice:.4f}"
        )

        # ========================
        # EARLY STOPPING
        # ========================
        if mean_dice > best_dice + MIN_DELTA:
            best_dice = mean_dice
            epochs_no_improve = 0
            torch.save(
                model.state_dict(),
                "checkpoints/baseline/unet_baseline_best.pt"
            )
            print(f"✅ New best baseline Dice: {best_dice:.4f}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            print(
                f"🛑 Early stopping: no Dice improvement ≥ {MIN_DELTA} "
                f"for {PATIENCE} epochs"
            )
            break

    wandb.finish()


if __name__ == "__main__":
    main()
