"""
Baseline YOLOv8n Training for Object Detection (without dendrites)
Used for comparison with dendritic version
"""
import argparse
from ultralytics import YOLO
import wandb


def main():
    parser = argparse.ArgumentParser(description='Baseline YOLOv8n Training')
    parser.add_argument('--data', type=str, default='coco128.yaml', help='Dataset YAML path')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--project', type=str, default='yolov8-baseline', help='Project name')
    parser.add_argument('--name', type=str, default='baseline', help='Run name')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    args = parser.parse_args()

    if args.use_wandb:
        wandb.login()
        wandb.init(project="YOLOv8 Baseline", name=args.name)

    # Load YOLOv8n model
    model = YOLO('yolov8n.pt')

    # Train the model
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=args.patience,
        save=True,
        plots=True,
        verbose=True,
    )

    # Validate the model
    metrics = model.val()

    print(f"\n{'='*50}")
    print(f"Training Complete!")
    print(f"{'='*50}")
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

    if args.use_wandb:
        wandb.log({
            "final_mAP50": metrics.box.map50,
            "final_mAP50-95": metrics.box.map,
        })
        wandb.finish()

    return metrics


if __name__ == '__main__':
    main()
