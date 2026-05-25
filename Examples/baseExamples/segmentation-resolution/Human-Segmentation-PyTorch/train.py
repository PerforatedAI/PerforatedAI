# ------------------------------------------------------------------------------
#   Libraries
# ------------------------------------------------------------------------------
import os, json, argparse, torch, warnings

warnings.filterwarnings("ignore")

import models as module_arch
import evaluation.losses as module_loss
import evaluation.metrics as module_metric
import dataloaders.dataloader as module_data

from utils.logger import Logger
from trainer.trainer import Trainer

# PerforatedAI imports
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


# ------------------------------------------------------------------------------
#   Get instance
# ------------------------------------------------------------------------------
def get_instance(module, name, config, *args):
    return getattr(module, config[name]["type"])(*args, **config[name]["args"])


# ------------------------------------------------------------------------------
#   Main function
# ------------------------------------------------------------------------------
def main(config, resume):
    train_logger = Logger()

    # PAI Configuration for Segmentation (UNet/CNNs)
    GPA.pc.set_testing_dendrite_capacity(False)  # Full training mode
    GPA.pc.set_module_names_to_perforate(
        ["InvertedResidual", "DecoderBlock", "Linear", "Conv2d"]
    )
    GPA.pc.set_output_dimensions([-1, 0, -1, -1])  # [batch, channels, height, width]
    # Skip only stem and transition layers (keep output layer for dendrites)
    GPA.pc.append_module_ids_to_track(
        [".backbone.features.0", ".backbone.features.18.1"]
    )
    # Use fixed-epoch switch mode: add dendrites every 80 epochs
    GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
    GPA.pc.set_fixed_switch_num(80)
    GPA.pc.set_first_fixed_switch_num(80)

    # Build model architecture
    model = get_instance(module_arch, "arch", config)
    img_sz = config["train_loader"]["args"]["resize"]
    model.summary(input_shape=(3, img_sz, img_sz))

    # Initialize PAI after summary (summary doesn't work with wrapped model)
    model = UPA.perforate_model(
        model, save_name="HumanSeg_dendritic", maximizing_score=False
    )

    # Setup data_loader instances
    train_loader = get_instance(module_data, "train_loader", config).loader
    valid_loader = get_instance(module_data, "valid_loader", config).loader

    # Get function handles of loss and metrics
    loss = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # Build optimizer, learning rate scheduler.
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = get_instance(torch.optim, "optimizer", config, trainable_params)
    lr_scheduler = get_instance(
        torch.optim.lr_scheduler, "lr_scheduler", config, optimizer
    )

    # Create trainer and start training
    trainer = Trainer(
        model,
        loss,
        metrics,
        optimizer,
        resume=resume,
        config=config,
        data_loader=train_loader,
        valid_data_loader=valid_loader,
        lr_scheduler=lr_scheduler,
        train_logger=train_logger,
    )
    trainer.train()


# ------------------------------------------------------------------------------
#   Main execution
# ------------------------------------------------------------------------------
if __name__ == "__main__":

    # Argument parsing
    parser = argparse.ArgumentParser(description="Train model")

    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )

    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )

    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    parser.add_argument(
        "--resolution-scale",
        default="full",
        type=str,
        choices=["full", "half", "quarter"],
        help="image resolution scale: full (1x), half (0.5x), or quarter (0.25x) (default: full)",
    )

    args = parser.parse_args()

    # Load config file
    if args.config:
        config = json.load(open(args.config))
        path = os.path.join(config["trainer"]["save_dir"], config["name"])

    # Load config file from checkpoint, in case new config file is not given.
    # Use '--config' and '--resume' arguments together to load trained model and train more with changed config.
    elif args.resume:
        config = torch.load(args.resume)["config"]

    # AssertionError
    else:
        raise AssertionError(
            "Configuration file need to be specified. Add '-c config.json', for example."
        )

    # Apply resolution scaling
    if args.resolution_scale != "full":
        scale_factor = 0.5 if args.resolution_scale == "half" else 0.25
        base_resolution = config["train_loader"]["args"]["resize"]
        new_resolution = int(base_resolution * scale_factor)

        # Ensure resolution is divisible by 32 for UNet compatibility
        new_resolution = (new_resolution // 32) * 32

        print(
            f"Scaling resolution from {base_resolution} to {new_resolution} ({args.resolution_scale})"
        )

        config["train_loader"]["args"]["resize"] = new_resolution
        config["valid_loader"]["args"]["resize"] = new_resolution

    # Set visible devices
    if args.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Run the main function
    main(config, args.resume)
