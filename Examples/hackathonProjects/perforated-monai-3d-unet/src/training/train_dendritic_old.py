import os
os.environ["PYTHONBREAKPOINT"] = "0"

import builtins
builtins.breakpoint = lambda *args, **kwargs: None

import sys
import torch
import wandb

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, ROOT_DIR)

import bootstrap  # noqa

from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA

from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet

from src.data.dataset_loader import get_dataloaders

# ========================
# CONFIG
# ========================
PROJECT_NAME = "Perforated-MONAI"
RUN_NAME = "dendritic_unet_from_baseline"

DATA_DIR = "datasets/monai"
DEVICE = "cuda"

LR = 2e-4
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 3
MAX_EPOCHS = 60

BASELINE_CKPT = "checkpoints/baseline/unet_baseline_best.pt"


def flatten_if_needed(x):
    if x.ndim == 6:
        return x.flatten(0, 1)
    return x


def main():
    os.makedirs("checkpoints/dendritic", exist_ok=True)
    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    # ============================================================
    # 🔴 PAI CONFIG — MUST BE SET *BEFORE* INITIALIZE
    # ============================================================
    GPA.pc.set_switch_mode("DOING_HISTORY")
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_max_dendrites(6)
    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_module_names_to_convert(["Conv3d"])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)
    GPA.pc.set_improvement_threshold(0.01)

    # 🔴 THIS LINE PREVENTS PDB DROPS (YOU WERE MISSING THIS)
    GPA.pc.set_disable_interactive_debugging(True)

    # ========================
    # MODEL
    # ========================
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=NUM_CLASSES,
        channels=(32, 64, 128, 256, 512),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    ).to(DEVICE)

    print("✅ Loading baseline checkpoint")
    model.load_state_dict(torch.load(BASELINE_CKPT, map_location=DEVICE))

    # ========================
    # INIT PAI (AFTER LOADING WEIGHTS)
    # ========================
    model = UPA.initialize_pai(
        model,
        save_name="PAI_MONAI",
        maximizing_score=True,
    )

    # 🔴 MUST BE SET AGAIN (initialize_pai resets internals)
    GPA.pc.set_switch_mode("DOING_HISTORY")
    GPA.pc.set_disable_interactive_debugging(True)

    # Conv3D dendrite shape fix
    for m in model.modules():
        if hasattr(m, "set_this_output_dimensions"):
            m.set_this_output_dimensions([-1, 0, -1, -1, -1])

    # ========================
    # OPTIMIZER
    # ========================
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    optimizer = GPA.pai_tracker.setup_optimizer(
        model,
        {"params": model.parameters(), "lr": LR, "weight_decay": 1e-5},
    )

    train_loader, val_loader = get_dataloaders(DATA_DIR)
    steps_per_epoch = len(train_loader)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=MAX_EPOCHS * steps_per_epoch
    )

    # ========================
    # LOSS & METRIC
    # ========================
    loss_fn = DiceCELoss(
        sigmoid=True,
        include_background=False,
        lambda_dice=1.0,
        lambda_ce=0.0,
    )

    dice_metric = DiceMetric(
        include_background=False,
        reduction="mean_batch",
    )

    # ========================
    # TRAIN LOOP
    # ========================
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x = flatten_if_needed(batch["image"]).to(DEVICE)
            y = flatten_if_needed(batch["label"]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                out = model(x)
                loss = loss_fn(out, y)

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
                x = flatten_if_needed(batch["image"]).to(DEVICE)
                y = flatten_if_needed(batch["label"]).to(DEVICE)

                with torch.amp.autocast("cuda"):
                    out = sliding_window_inference(x, PATCH_SIZE, 2, model)

                dice_metric(y_pred=out, y=y)

        dice = dice_metric.aggregate().mean().item()

        model, restructured, done = GPA.pai_tracker.add_validation_score(
            dice, model
        )
        model = model.to(DEVICE)

        if restructured:
            print("🧠 Dendrites grown")

        wandb.log({
            "epoch": epoch + 1,
            "loss": epoch_loss,
            "dice": dice,
            "dendrites": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
        })

        print(f"Epoch {epoch+1:03d} | Loss {epoch_loss:.4f} | Dice {dice:.4f}")

        if done:
            print("✅ PAI training complete")
            break

    torch.save(
        model.state_dict(),
        "checkpoints/dendritic/unet_dendritic_final.pt"
    )
    wandb.finish()


if __name__ == "__main__":
    main()
