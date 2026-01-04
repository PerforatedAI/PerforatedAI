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

from src.data.dataset_loader import get_dataloaders
from src.models.unet_baseline import get_unet

# ========================
# CONFIG
# ========================
PROJECT_NAME = "Perforated-MONAI"
RUN_NAME = "baseline_unet"

DATA_DIR = "datasets/monai"
NUM_EPOCHS = 30               # 🔴 match dendritic
LR = 5e-5                     # 🔴 match dendritic
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 4
DEVICE = "cuda"


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
    # MODEL (MATCHES DENDRITIC)
    # ========================
    model = get_unet(out_channels=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    # ========================
    # LOSS & METRICS (MATCH)
    # ========================
    loss_fn = DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        include_background=False,
        lambda_dice=1.0,
        lambda_ce=0.5,
    )

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",
        get_not_nans=False,
    )

    scaler = torch.amp.GradScaler("cuda")

    # ========================
    # DATA
    # ========================
    train_loader, val_loader = get_dataloaders(DATA_DIR)

    # ========================
    # TRAIN LOOP
    # ========================
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            inputs = flatten_if_needed(batch["image"].cuda())
            labels = flatten_if_needed(batch["label"].cuda())

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # ✅ correct scheduler usage (once per epoch)
        scheduler.step()

        # ========================
        # VALIDATION
        # ========================
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                val_inputs = flatten_if_needed(batch["image"].cuda())
                val_labels = flatten_if_needed(batch["label"].cuda())

                if val_labels.ndim == 5:
                    val_labels = val_labels.squeeze(1)
                val_labels = val_labels.long()

                with torch.amp.autocast("cuda"):
                    val_outputs = sliding_window_inference(
                        val_inputs,
                        PATCH_SIZE,
                        sw_batch_size=1,
                        predictor=model,
                    )

                # 🔴 RAW LOGITS (same as dendritic)
                dice_metric(y_pred=val_outputs, y=val_labels)

        dice_vals = dice_metric.aggregate()
        mean_dice = dice_vals.mean().item()


        wandb.log({
            "epoch": epoch,
            "train/loss": epoch_loss,
            "val/dice": mean_dice,
            "dice/WT": dice_vals[0].item(),
            "dice/TC": dice_vals[1].item(),
            "dice/ET": dice_vals[2].item(),
            "lr": scheduler.get_last_lr()[0],
        })


        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Loss {epoch_loss:.4f} | "
            f"Dice {mean_dice:.4f}"
        )

    # ========================
    # SAVE
    # ========================
    torch.save(
        model.state_dict(),
        "checkpoints/baseline/unet_baseline.pt"
    )

    wandb.finish()


if __name__ == "__main__":
    main()
