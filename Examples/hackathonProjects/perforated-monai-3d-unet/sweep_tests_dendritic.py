import os
import sys
import torch
import wandb
import builtins
import os
os.environ["PYTHONBREAKPOINT"] = "0"
os.environ["WANDB_DISABLE_CONSOLE_CAPTURE"] = "true"


# ------------------------
# ENV / SAFETY
# ------------------------
os.environ["PYTHONBREAKPOINT"] = "0"
builtins.breakpoint = lambda *args, **kwargs: None
import pdb
pdb.set_trace = lambda *args, **kwargs: None

ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT_DIR)

import bootstrap

from perforatedai import utils_perforatedai as UPA
from perforatedai import globals_perforatedai as GPA

from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.inferers import sliding_window_inference

from src.data.dataset_loader import get_dataloaders

# ------------------------
# CONSTANTS
# ------------------------
PROJECT_NAME = "Perforated-MONAI"
DATA_DIR = "datasets/monai"
PATCH_SIZE = (96, 96, 96)
NUM_CLASSES = 4
DEVICE = "cuda"
MAX_EPOCHS = 50


def flatten_if_needed(x):
    return x.flatten(0, 1) if x.ndim == 6 else x


# ------------------------
# TRAINING FUNCTION
# ------------------------
def train():
      # 🔴 MOVE THIS HERE
    GPA.pc.set_switch_mode("DOING_HISTORY")
    best_dice = -1.0
    patience = 4
    patience_ctr = 0

    wandb.init(project=PROJECT_NAME)
    cfg = wandb.config

    wandb.run.name = (
        f"{cfg.model_variant}_d{cfg.max_dendrites}_t{cfg.improvement_threshold}"
    )

    # ------------------------
    # MODEL CAPACITY AXIS
    # ------------------------
    if cfg.model_variant == "full":
        channels = (32, 64, 128, 256)
    else:
        channels = (24, 40, 80, 160)

    # ------------------------
    # PAI CONFIG (SWEEPED)
    # ------------------------
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_max_dendrites(cfg.max_dendrites)
    GPA.pc.set_improvement_threshold(cfg.improvement_threshold)

    GPA.pc.set_perforated_backpropagation(False)
    GPA.pc.set_module_names_to_convert(["Conv3d"])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(False)
  

    # ------------------------
    # MODEL
    # ------------------------
    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=NUM_CLASSES,
        channels=channels,
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(DEVICE)

    model = UPA.initialize_pai(
        model,
        save_name=f"PAI_MONAI_{cfg.model_variant}",
        maximizing_score=True,
    )
    
  
    for m in model.modules():
        if hasattr(m, "set_this_output_dimensions"):
            m.set_this_output_dimensions([-1, 0, -1, -1, -1])

    # ------------------------
    # OPTIMIZER + AMP
    # ------------------------
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    optimizer = GPA.pai_tracker.setup_optimizer(
        model, {"params": model.parameters(), "lr": cfg.lr}
    )

    scaler = torch.cuda.amp.GradScaler()

    # ------------------------
    # DATA
    # ------------------------
    train_loader, val_loader = get_dataloaders(DATA_DIR)

    # ------------------------
    # LOSS & METRICS
    # ------------------------
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
    )

    post_pred = AsDiscrete(argmax=True, to_onehot=NUM_CLASSES)

    # ------------------------
    # TRAIN LOOP
    # ------------------------
    epoch = 0
    while True:
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            x = flatten_if_needed(batch["image"]).to(DEVICE)
            y = flatten_if_needed(batch["label"]).to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                out = model(x)
                loss = loss_fn(out, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # ------------------------
        # VALIDATION
        # ------------------------
        model.eval()
        dice_metric.reset()

        with torch.no_grad():
            for batch in val_loader:
                x = flatten_if_needed(batch["image"]).to(DEVICE)
                y = flatten_if_needed(batch["label"]).to(DEVICE).long()

                with torch.cuda.amp.autocast():
                    out = sliding_window_inference(
                        x, PATCH_SIZE, 2, model
                    )

                out = post_pred(out)
                dice_metric(y_pred=out, y=y)

        mean_dice = dice_metric.aggregate().mean().item()

        # ------------------------
        # EARLY STOPPING
        # ------------------------
        if mean_dice > best_dice + 1e-4:
            best_dice = mean_dice
            patience_ctr = 0
        else:
            patience_ctr += 1

        # ------------------------
        # PAI TRACKER
        # ------------------------
        model, _, done = GPA.pai_tracker.add_validation_score(
            mean_dice, model
        )
        model = model.to(DEVICE)

        # ------------------------
        # LOGGING
        # ------------------------
        wandb.log({
            "epoch": epoch,
            "val/dice": mean_dice,
            "best_val_dice": best_dice,
            "train/loss": epoch_loss,
            "params": sum(p.numel() for p in model.parameters()),
            "model_variant": cfg.model_variant,
            "max_dendrites": cfg.max_dendrites,
            "improvement_threshold": cfg.improvement_threshold,
            "lr": cfg.lr,
            "num_dendrites_added": GPA.pai_tracker.member_vars.get(
                "num_dendrites_added", 0
            ),
        })
        print(
            f"[{cfg.model_variant} | d={cfg.max_dendrites}] "
            f"Epoch {epoch} | Dice {mean_dice:.4f} | Best {best_dice:.4f}"
        )

        # ------------------------
        # STOP CONDITIONS
        # ------------------------
        if done:
            wandb.log({"stop_reason": "pai_converged"})
            break

        if patience_ctr >= patience:
            wandb.log({"stop_reason": "early_stopping"})
            break

        if epoch >= MAX_EPOCHS:
            wandb.log({"stop_reason": "max_epoch_cap"})
            break

        epoch += 1

    wandb.finish()


# ------------------------
# SWEEP CONFIG
# ------------------------
SWEEP_CONFIG = {
    "method": "grid",
    "metric": {
        "name": "best_val_dice",
        "goal": "maximize",
    },
    "parameters": {
        "model_variant": {
            "values": ["full", "compressed"]
        },
        "max_dendrites": {
            "values": [0, 1, 2, 3]
        },
        "improvement_threshold": {
            "values": [0.0, 0.005, 0.01]
        },
        "lr": {
            "value": 5e-5
        },
    },
}



# ------------------------
# ENTRYPOINT
# ------------------------
if __name__ == "__main__":
    sweep_id = wandb.sweep(
        sweep=SWEEP_CONFIG,
        project=PROJECT_NAME,
    )
    wandb.agent(sweep_id, function=train)
