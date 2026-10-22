"""
The 1D ResNet from resnet1d.py, trained with PerforatedAI dendrites.

The data, model and training step are imported unchanged from resnet1d.py. The
PerforatedAI changes are marked "PAI:" below:
1. Choose which modules get dendrites and add the dendrite scaffolding with
   UPA.perforate_model.
2. Build the optimizer and scheduler through GPA.pai_tracker, which steps the
   scheduler itself.
3. Report validation balanced accuracy to GPA.pai_tracker.add_validation_score
   after every epoch. After --switch-epochs epochs without improvement,
   PerforatedAI restores the best network, adds a set of dendrites and restarts
   the learning rate. It keeps sets that improve the score, and ends training
   with the best network when new sets stop helping.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from perforatedai import blockwise_perforatedai as BPA
from perforatedai import clean_perforatedai as CPA
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

from resnet1d import (
    DILATIONS,
    BrainStormData,
    ResNet1d,
    balanced_accuracy,
    common_args,
    set_seed,
    train_epoch,
)

# PAI: modules that get dendrites. Every other module with parameters is tracked.
PERFORATION_TARGETS = {
    # The 1x1 output convolution only
    "head": {"names": [], "ids": [".head"]},
    # Whole residual blocks plus the head
    "blocks": {"names": ["ResidualBlock"], "ids": [".head"]},
}


def configure_pai(args):
    """PAI: global settings, made before UPA.perforate_model."""
    targets = PERFORATION_TARGETS[args.perforate]
    GPA.pc.set_testing_dendrite_capacity(False)
    # Gradient-descent dendrites from the open-source package
    GPA.pc.set_perforated_backpropagation(False)
    # Skip the interactive confirmation screens so the script runs unattended
    GPA.pc.set_configuration_confirmed(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_module_names_to_perforate(targets["names"])
    GPA.pc.set_module_ids_to_perforate(targets["ids"])
    GPA.pc.set_module_names_to_track(["CausalConv1d", "BatchNorm1d"])
    # Perforated modules output (batch, channels, time)
    GPA.pc.set_output_dimensions([-1, 0, -1])
    GPA.pc.set_n_epochs_to_switch(args.switch_epochs)
    GPA.pc.set_max_dendrites(args.max_dendrites)


def main():
    parser = common_args("Train the 1D ResNet on BrainStorm with dendrites.")
    parser.add_argument(
        "--perforate",
        choices=sorted(PERFORATION_TARGETS),
        default="head",
        help="modules that get dendrites",
    )
    parser.add_argument(
        "--max-dendrites", type=int, default=5, help="most dendrite sets per module"
    )
    parser.add_argument(
        "--switch-epochs",
        type=int,
        default=10,
        help="epochs without improvement before dendrites are added",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1500,
        help="safety cap; PerforatedAI normally ends training first",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = BrainStormData(args.data_dir, device)
    net = ResNet1d(
        len(data.classes), width=args.width, dilations=DILATIONS[args.memory]
    )
    rf = net.receptive_field
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"{args.memory} model ({rf} ms of history), width {args.width}: "
        f"{n_params:,} parameters, dendrites on the {args.perforate}, on {device}"
    )

    # PAI: configure, then add the dendrite scaffolding
    configure_pai(args)
    GPA.pc.set_device(device)
    net = UPA.perforate_model(net, save_name="PAI", maximizing_score=True)
    net.to(device)

    # PAI: the tracker builds the optimizer and scheduler, and steps the
    # scheduler with each validation score
    GPA.pai_tracker.set_optimizer(torch.optim.AdamW)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    def build_optimizer(net):
        optim_args = {
            "params": net.parameters(),
            "lr": args.lr,
            "weight_decay": args.weight_decay,
        }
        sched_args = {"mode": "max", "factor": 0.5, "patience": args.lr_patience}
        optimizer, _ = GPA.pai_tracker.setup_optimizer(net, optim_args, sched_args)
        return optimizer

    optimizer = build_optimizer(net)
    criterion = nn.CrossEntropyLoss(weight=data.class_weights)

    history = []
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(net, data, optimizer, criterion, rf, args)
        lr = optimizer.param_groups[0]["lr"]
        train_acc = balanced_accuracy(net, data.x_train_seq, data.y_train)
        val_acc = balanced_accuracy(net, data.x_val, data.y_val)
        dendrites = GPA.pai_tracker.member_vars["num_dendrites_added"]
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "loss": loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
                "dendrites": dendrites,
            }
        )
        print(
            f"epoch {epoch:4d}  {dendrites} dendrite sets  lr {lr:.1e}  "
            f"loss {loss:.4f}  train {train_acc:.2f}%  val {val_acc:.2f}%"
        )

        # PAI: report the scores. PerforatedAI may restore the best network and
        # add dendrites, in which case the optimizer has to be rebuilt.
        GPA.pai_tracker.add_extra_score(train_acc, "Train")
        net, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            val_acc, net
        )
        net.to(device)
        if training_complete:
            break
        if restructured:
            optimizer = build_optimizer(net)
    else:
        print(f"PerforatedAI didn't finish within --max-epochs {args.max_epochs}")

    # PAI: once training is complete, the tracker has restored the best network
    val_acc = balanced_accuracy(net, data.x_val, data.y_val)
    before = max(h["val_accuracy"] for h in history if h["dendrites"] == 0)
    # PAI: the inference form keeps the dendrites and drops training-only copies
    net = CPA.refresh_net(BPA.blockwise_network(net.eval()))
    final_params = sum(p.numel() for p in net.parameters())
    pai_modules = [m for m in net.modules() if isinstance(m, CPA.PAIModulePyThread)]
    sets = max(len(m.layer_array) - 1 for m in pai_modules)

    print(f"Before dendrites: {before:.2f}% validation, {n_params:,} parameters")
    print(
        f"After PerforatedAI: {val_acc:.2f}% validation, {final_params:,} "
        f"parameters, {sets} dendrite sets, {len(history)} epochs"
    )
    if args.out:
        result = {
            "val_accuracy": val_acc,
            "val_accuracy_before_dendrites": before,
            "dendrite_sets": sets,
            "params_before_dendrites": n_params,
            "params": final_params,
            "epochs": len(history),
            "args": vars(args),
            "history": history,
        }
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
