import os
os.environ["PYTHONBREAKPOINT"] = "0"

import sys
import torch
import wandb
import builtins

# hard-disable breakpoint everywhere
builtins.breakpoint = lambda *args, **kwargs: None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

import bootstrap

from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from src.data.dataset_loader import get_dataloaders

# ========================
# CONFIG
# ========================
PROJECT_NAME = "Perforated-MONAI"
RUN_NAME = "dendritic_unet"

DATA_DIR = "datasets/monai"
NUM_EPOCHS = 20
LR = 3e-5
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 4
DEVICE = "cuda"


def flatten_if_needed(x):
    # [B,S,C,D,H,W] → [B*S,C,D,H,W]
    if x.ndim == 6:
        return x.flatten(0, 1)
    return x


def main():
    os.makedirs("checkpoints/dendritic", exist_ok=True)
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    torch.backends.cudnn.benchmark = False
    torch.cuda.set_device(0)

    # ============================================================
    # 🔴 PAI CONFIG — MUST BE SET *BEFORE* initialize_pai
    # ============================================================
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_n(4)
    GPA.pc.set_max_dendrites(10)

    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_module_names_to_convert(["Conv3d"])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)


    # ========================
    # MODEL
    # ========================
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=NUM_CLASSES,
        channels = (24, 40, 80, 160),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)

    # 🔴 tracker is created HERE
    model = UPA.initialize_pai(
        model,
        save_name="PAI_MONAI",
        maximizing_score=True,
    )
    GPA.pc.set_switch_mode("DOING_HISTORY")   
    # Required for Conv3D dendrites
    for m in model.modules():
        if hasattr(m, "set_this_output_dimensions"):
            m.set_this_output_dimensions([-1, 0, -1, -1, -1])

    # ========================
    # OPTIMIZER + SCHEDULER
    # ========================
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    optimizer = GPA.pai_tracker.setup_optimizer(
        model,
        {"params": model.parameters(), "lr": LR},
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS
    )

    # ========================
    # LOSS & METRICS
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

    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)

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

            loss.backward()
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # ========================
        # VALIDATION
        # ========================
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                inputs = flatten_if_needed(batch["image"].cuda())
                labels = flatten_if_needed(batch["label"].cuda())

                if labels.ndim == 5:
                    labels = labels.squeeze(1)
                labels = labels.long()

                with torch.amp.autocast("cuda"):
                    outputs = sliding_window_inference(
                        inputs,
                        PATCH_SIZE,
                        sw_batch_size=1,
                        predictor=model,
                    )

                outputs = post_pred(outputs)
                dice_metric(y_pred=outputs, y=labels)

        dice_vals = dice_metric.aggregate()
        mean_dice = dice_vals.mean().item()

        # ========================
        # DENDRITE TRIGGER
        # ========================
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            mean_dice, model
        )
        model = model.to(DEVICE)

        if restructured and not training_complete:
            GPA.pai_tracker.set_optimizer(torch.optim.Adam)
            optimizer = GPA.pai_tracker.setup_optimizer(
                model, {"params": model.parameters(), "lr": LR * 2}
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - epoch
            )
            print("🧠 Dendrites grown → optimizer & scheduler reset")

        wandb.log({
            "epoch": epoch,
            "train/loss": epoch_loss,
            "val/dice": mean_dice,
            "dice/WT": dice_vals[0].item(),
            "dice/TC": dice_vals[1].item(),
            "dice/ET": dice_vals[2].item(),
            "dendrites": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
            "lr": scheduler.get_last_lr()[0],
        })

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Loss {epoch_loss:.4f} | "
            f"Dice {mean_dice:.4f}"
        )

        if training_complete:
            print("✅ Training complete according to PAI tracker")
            break

    torch.save(
        model.state_dict(),
        "checkpoints/dendritic/unet_dendritic_compressed.pt",
    )
    wandb.finish()


if __name__ == "__main__":
    main()
