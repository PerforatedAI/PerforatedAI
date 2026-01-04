"""
YOLOv8n with Dendritic Optimization for Object Detection
Hackathon Submission - Dendritic YOLOv8
"""
import argparse
import torch
import torch.nn as nn
from pathlib import Path
from types import SimpleNamespace
import wandb
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import LOGGER

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


def train_yolo(args, config, run=None):
    """Train YOLOv8 with dendritic optimization"""

    # Set PAI configuration from config
    # Decode improvement_threshold
    if config.improvement_threshold == 0:
        thresh = [0.01, 0.001, 0.0001, 0]
    elif config.improvement_threshold == 1:
        thresh = [0.001, 0.0001, 0]
    elif config.improvement_threshold == 2:
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)

    GPA.pc.set_candidate_weight_initialization_multiplier(
        config.candidate_weight_initialization_multiplier
    )

    # Decode pai_forward_function
    if config.pai_forward_function == 0:
        pai_forward_function = torch.sigmoid
    elif config.pai_forward_function == 1:
        pai_forward_function = torch.relu
    elif config.pai_forward_function == 2:
        pai_forward_function = torch.tanh
    else:
        pai_forward_function = torch.sigmoid

    GPA.pc.set_pai_forward_function(pai_forward_function)

    # Set dendrite mode
    if config.dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
    else:
        GPA.pc.set_max_dendrites(5)

    if config.dendrite_mode < 2:
        GPA.pc.set_perforated_backpropagation(False)
    else:
        GPA.pc.set_perforated_backpropagation(True)

    # Set wandb run name
    if run is not None:
        excluded = ['method', 'metric', 'parameters', 'dendrite_mode']
        keys = [k for k in get_parameters_dict().keys() if k not in excluded]
        name_str = "Dendrites-" + str(config.dendrite_mode) + "_" + "_".join(
            str(getattr(config, k)) for k in keys if hasattr(config, k)
        )
        run.name = name_str

    # Set up PAI global parameters
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(True)

    # Load YOLOv8n model
    model = YOLO('yolov8n.yaml')

    # Get the underlying PyTorch model
    if hasattr(model, 'model'):
        pytorch_model = model.model
    else:
        pytorch_model = model

    # Initialize with PerforatedAI
    pytorch_model = UPA.initialize_pai(pytorch_model, save_name=args.save_name)

    # Setup custom training with PAI integration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pytorch_model = pytorch_model.to(device)

    # Setup optimizer through PAI tracker
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    optimArgs = {
        'params': pytorch_model.parameters(),
        'lr': config.learning_rate,
        'weight_decay': config.weight_decay
    }
    schedArgs = {'mode': 'max', 'patience': 5}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(pytorch_model, optimArgs, schedArgs)

    # Train using Ultralytics with custom callbacks
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=args.project,
        name=run.name if run else args.name,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
    )

    # Get final validation metrics
    metrics = model.val()

    if run is not None:
        run.log({
            "final_mAP50": metrics.box.map50,
            "final_mAP50-95": metrics.box.map,
            "final_params": UPA.count_params(pytorch_model),
            "final_dendrites": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
        })

    return metrics


def train_yolo_custom_loop(args, config, run=None):
    """
    Custom training loop with full PAI integration.
    This is the recommended approach for hackathon to show dendritic optimization clearly.
    """

    # Set PAI configuration
    if config.improvement_threshold == 0:
        thresh = [0.01, 0.001, 0.0001, 0]
    elif config.improvement_threshold == 1:
        thresh = [0.001, 0.0001, 0]
    elif config.improvement_threshold == 2:
        thresh = [0]
    GPA.pc.set_improvement_threshold(thresh)

    GPA.pc.set_candidate_weight_initialization_multiplier(
        config.candidate_weight_initialization_multiplier
    )

    if config.pai_forward_function == 0:
        pai_forward_function = torch.sigmoid
    elif config.pai_forward_function == 1:
        pai_forward_function = torch.relu
    elif config.pai_forward_function == 2:
        pai_forward_function = torch.tanh
    else:
        pai_forward_function = torch.sigmoid

    GPA.pc.set_pai_forward_function(pai_forward_function)

    if config.dendrite_mode == 0:
        GPA.pc.set_max_dendrites(0)
    else:
        GPA.pc.set_max_dendrites(5)

    if config.dendrite_mode < 2:
        GPA.pc.set_perforated_backpropagation(False)
    else:
        GPA.pc.set_perforated_backpropagation(True)

    if run is not None:
        excluded = ['method', 'metric', 'parameters', 'dendrite_mode']
        keys = [k for k in get_parameters_dict().keys() if k not in excluded]
        name_str = "Dendrites-" + str(config.dendrite_mode) + "_" + "_".join(
            str(getattr(config, k)) for k in keys if hasattr(config, k)
        )
        run.name = name_str

    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_verbose(True)

    # Load YOLOv8n
    yolo = YOLO('yolov8n.pt')

    # Extract the model
    model = yolo.model

    # Initialize PAI
    model = UPA.initialize_pai(model, save_name=args.save_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Setup optimizer
    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

    optimArgs = {
        'params': model.parameters(),
        'lr': config.learning_rate,
        'weight_decay': config.weight_decay
    }
    schedArgs = {'mode': 'max', 'patience': 5}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    # Tracking variables
    best_map = 0
    global_best_map = 0
    global_best_params = 0

    # Training loop using YOLO's built-in trainer but with PAI callbacks
    for epoch in range(args.epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*50}")

        # Train for one epoch using ultralytics
        results = yolo.train(
            data=args.data,
            epochs=1,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=args.project,
            name=run.name if run else args.name,
            resume=epoch > 0,
            verbose=False,
        )

        # Validate
        metrics = yolo.val()
        map50_95 = float(metrics.box.map)
        map50 = float(metrics.box.map50)

        print(f"Validation - mAP@0.5: {map50:.4f}, mAP@0.5:0.95: {map50_95:.4f}")

        # Add score to PAI tracker
        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
            map50_95 * 100,  # Convert to percentage
            model
        )
        model = model.to(device)

        # Update tracking
        if map50_95 > best_map:
            best_map = map50_95

        if map50_95 > global_best_map:
            global_best_map = map50_95
            global_best_params = UPA.count_params(model)

        # Log to wandb
        if run is not None:
            run.log({
                "epoch": epoch,
                "mAP50": map50,
                "mAP50-95": map50_95,
                "params": UPA.count_params(model),
                "dendrites": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
            })

        # If restructured, reset optimizer
        if restructured and not training_complete:
            print(f"\n🌳 Model restructured! Dendrites added: {GPA.pai_tracker.member_vars.get('num_dendrites_added', 0)}")
            optimArgs = {
                'params': model.parameters(),
                'lr': config.learning_rate,
                'weight_decay': config.weight_decay
            }
            schedArgs = {'mode': 'max', 'patience': 5}
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

            # Update YOLO model reference
            yolo.model = model

        if training_complete:
            print(f"\n✅ Training complete!")
            print(f"Best mAP@0.5:0.95: {global_best_map:.4f}")
            print(f"Final params: {global_best_params:,}")

            if run is not None:
                run.log({
                    "final_best_mAP": global_best_map,
                    "final_params": global_best_params,
                    "final_dendrites": GPA.pai_tracker.member_vars.get("num_dendrites_added", 0),
                })
            break

    return metrics


def get_parameters_dict():
    """Return the parameters dictionary for the sweep."""
    parameters_dict = {
        "learning_rate": {"values": [0.01, 0.001, 0.0001]},
        "weight_decay": {"values": [0, 0.0001, 0.0005]},
        "improvement_threshold": {"values": [0, 1, 2]},
        "candidate_weight_initialization_multiplier": {"values": [0.1, 0.01]},
        "pai_forward_function": {"values": [0, 1, 2]},
        "dendrite_mode": {"values": [0, 1]},  # 0=no dendrites, 1=GD dendrites
    }
    return parameters_dict


def run_sweep():
    """Wrapper function for wandb sweep."""
    try:
        with wandb.init() as wandb_run:
            parser = argparse.ArgumentParser()
            parser.add_argument('--data', type=str, default='coco128.yaml')
            parser.add_argument('--epochs', type=int, default=100)
            parser.add_argument('--imgsz', type=int, default=640)
            parser.add_argument('--batch', type=int, default=16)
            parser.add_argument('--project', type=str, default='dendritic-yolov8')
            parser.add_argument('--name', type=str, default='train')
            parser.add_argument('--patience', type=int, default=50)
            parser.add_argument('--save-name', type=str, default='PAI')
            args = parser.parse_args()

            train_yolo_custom_loop(args, wandb_run.config, wandb_run)
    except Exception as e:
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='YOLOv8n with Dendritic Optimization')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--project', type=str, default='dendritic-yolov8', help='Project name')
    parser.add_argument('--name', type=str, default='train', help='Run name')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--save-name', type=str, default='PAI', help='PAI save name')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    parser.add_argument('--sweep-id', type=str, default='main', help='Sweep ID or "main"')
    parser.add_argument('--count', type=int, default=50, help='Number of sweep runs')
    args = parser.parse_args()

    if args.use_wandb:
        wandb.login()
        project = "Dendritic YOLOv8"

        sweep_config = {"method": "random"}
        metric = {"name": "mAP50-95", "goal": "maximize"}
        sweep_config["metric"] = metric
        sweep_config["parameters"] = get_parameters_dict()

        if args.sweep_id == "main":
            sweep_id = wandb.sweep(sweep_config, project=project)
            print(f"\n✨ Initialized sweep: {sweep_id}")
            print(f"Use --sweep-id {sweep_id} to join on other machines.\n")
            wandb.agent(sweep_id, run_sweep, count=args.count)
        else:
            wandb.agent(args.sweep_id, run_sweep, count=args.count, project=project)
    else:
        # Run single training without wandb
        config = SimpleNamespace(
            learning_rate=0.001,
            weight_decay=0.0005,
            improvement_threshold=1,
            candidate_weight_initialization_multiplier=0.01,
            pai_forward_function=0,
            dendrite_mode=1,
        )
        train_yolo_custom_loop(args, config, None)


if __name__ == '__main__':
    main()
