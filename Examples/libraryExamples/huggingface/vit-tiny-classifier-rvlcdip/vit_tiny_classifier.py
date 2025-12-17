from datasets import load_dataset
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch
import argparse
import random

# PAI imports (optional - only used when --use-dendrites is set)
GPA = None
UPA = None


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def init_pai():
    """Initialize PAI imports."""
    global GPA, UPA
    from perforatedai import globals_perforatedai as _GPA
    from perforatedai import utils_perforatedai as _UPA
    GPA = _GPA
    UPA = _UPA
    GPA.pc.set_weight_decay_accepted(True)
    GPA.pc.set_unwrapped_modules_confirmed(True)


def load_processor():
    """Load the image processor for the ViT model."""
    return AutoImageProcessor.from_pretrained("HAMMALE/vit-tiny-classifier-rvlcdip")


def load_model():
    """Load a ViT model with random weights using the config from a pretrained checkpoint."""
    config = AutoConfig.from_pretrained("HAMMALE/vit-tiny-classifier-rvlcdip")
    return AutoModelForImageClassification.from_config(config)


def convert_label_to_class_name(label_id):
    """Convert a label id to its class name string."""
    class_names = [
        "letter", "form", "email", "handwritten", "advertisement",
        "scientific_report", "scientific_publication", "specification",
        "file_folder", "news_article", "budget", "invoice",
        "presentation", "questionnaire", "resume", "memo"
    ]
    return class_names[label_id]


def example_transform(example, processor):
    """Transform a dataset example into model-ready pixel values and label."""
    img = example["image"]
    pixel = processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0].cpu().numpy()
    return {"pixel_values": pixel, "label": example["label"]}


def batch_iterator(iterable_ds, batch_size, device, processor, max_samples=None):
    """Yield batches of (pixel_values, labels) from a streaming dataset, handling errors."""
    batch_pixels = []
    batch_labels = []
    seen = 0
    failed = 0
    iterator = iter(iterable_ds)

    while True:
        try:
            ex = next(iterator)
        except StopIteration:
            break
        except Exception:
            failed += 1
            continue

        try:
            transformed = example_transform(ex, processor)
            batch_pixels.append(transformed["pixel_values"])
            batch_labels.append(transformed["label"])
            seen += 1
        except Exception:
            failed += 1
            continue

        if len(batch_pixels) >= batch_size:
            pixel_batch = torch.from_numpy(np.stack(batch_pixels)).to(device)
            label_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)
            yield pixel_batch, label_batch
            batch_pixels = []
            batch_labels = []

        if max_samples is not None and seen >= max_samples:
            break

    # Yield remaining samples
    if batch_pixels:
        pixel_batch = torch.from_numpy(np.stack(batch_pixels)).to(device)
        label_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)
        yield pixel_batch, label_batch

    if failed > 0:
        print(f"Skipped {failed} examples due to processing errors.")


def create_optimizer_and_scheduler(model, lr, weight_decay, warmup_ratio, steps_per_epoch, epochs, use_dendrites=False):
    """Create AdamW optimizer and cosine scheduler with warmup."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name.lower() for nd in ["bias", "layernorm", "ln", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    num_training_steps = steps_per_epoch * epochs
    num_warmup_steps = int(num_training_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    print(f"Scheduler: total_steps={num_training_steps}, warmup_steps={num_warmup_steps}")

    # PAI: Register optimizer for learning rate tracking
    if use_dendrites and GPA is not None:
        GPA.pai_tracker.set_optimizer_instance(optimizer)

    return optimizer, scheduler


def evaluate(model, test_dataset, batch_size, device, processor, max_samples=None):
    """Evaluate the model on a test dataset and return accuracy."""
    correct = 0
    total = 0
    batch_num = 0
    model.eval()

    with torch.no_grad():
        for pixel_batch, label_batch in batch_iterator(test_dataset, batch_size, device, processor, max_samples):
            outputs = model(pixel_batch)
            preds = outputs.logits.argmax(-1)
            correct += int((preds == label_batch).sum().item())
            total += label_batch.size(0)
            batch_num += 1
            print(f"Eval batch {batch_num}: samples={total}, accuracy={correct/total*100:.2f}%")

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy * 100:.2f}% on {total} samples")
    return accuracy


def configure_pai_dimensions(model):
    """Configure PAI input dimensions for ViT layers."""
    try:
        patch_proj = model.vit.embeddings.patch_embeddings.projection
        if hasattr(patch_proj, "set_this_input_dimensions"):
            patch_proj.set_this_input_dimensions([-1, 0, -1, -1])
    except AttributeError:
        pass

    try:
        clf = model.classifier
        if hasattr(clf, "set_this_input_dimensions"):
            clf.set_this_input_dimensions([-1, 0])
    except AttributeError:
        pass


def train(
    model,
    batch_size,
    device,
    processor,
    epochs=1,
    lr=3e-4,
    max_samples=None,
    weight_decay=0.05,
    warmup_ratio=0.1,
    dataset_name="aharley/rvl_cdip",
    use_dendrites=False,
    save_name="vit_rvlcdip",
    streaming=False,
):
    """Train the model, optionally with PAI dendrites."""
    criterion = CrossEntropyLoss()

    # Load datasets once if not streaming
    if not streaming:
        print("Loading full train dataset (this may take a while on first run)...")
        train_dataset = load_dataset(dataset_name, split="train", trust_remote_code=True)
        print("Loading full validation dataset...")
        val_dataset = load_dataset(dataset_name, split="test", trust_remote_code=True)
        dataset_size = min(len(train_dataset), max_samples) if max_samples else len(train_dataset)
        steps_per_epoch = (dataset_size + batch_size - 1) // batch_size
    else:
        train_dataset = None
        val_dataset = None
        # Estimate steps per epoch for streaming
        if max_samples is not None:
            steps_per_epoch = (max_samples + batch_size - 1) // batch_size
        else:
            steps_per_epoch = 1000 // batch_size
            print(f"Using approximate steps_per_epoch={steps_per_epoch}")

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        use_dendrites=use_dendrites,
    )

    global_step = 0

    for epoch in range(epochs):
        # PAI: Signal start of epoch
        if use_dendrites and GPA is not None:
            GPA.pai_tracker.start_epoch()

        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        total = 0
        batch_num = 0
        model.train()

        # Reload dataset for streaming (exhausted after one pass), reuse for non-streaming
        if streaming:
            epoch_train_dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
        else:
            epoch_train_dataset = train_dataset

        for pixel_batch, label_batch in batch_iterator(epoch_train_dataset, batch_size, device, processor, max_samples):
            optimizer.zero_grad()
            outputs = model(pixel_batch)
            loss = criterion(outputs.logits, label_batch)
            loss.backward()

            clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            global_step += 1

            batch_size_actual = label_batch.size(0)
            total_loss += loss.item() * batch_size_actual
            total += batch_size_actual
            batch_num += 1

            if global_step % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}: batch={batch_num}, loss={loss.item():.4f}, lr={current_lr:.6f}")

        avg_loss = total_loss / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}, samples: {total}")

        # Validation after each epoch
        print(f"Running validation after epoch {epoch+1}...")
        if streaming:
            epoch_val_dataset = load_dataset(dataset_name, split="test", streaming=True, trust_remote_code=True)
        else:
            epoch_val_dataset = val_dataset
        accuracy = evaluate(model, epoch_val_dataset, batch_size, device, processor, max_samples)

        # PAI: Record validation score
        if use_dendrites and GPA is not None:
            GPA.pai_tracker.set_optimizer_instance(optimizer)
            model, training_complete, restructured = GPA.pai_tracker.add_validation_score(accuracy, model)

            if restructured:
                print("Model restructured by PAI, recreating optimizer...")
                optimizer, scheduler = create_optimizer_and_scheduler(
                    model=model,
                    lr=lr,
                    weight_decay=weight_decay,
                    warmup_ratio=warmup_ratio,
                    steps_per_epoch=steps_per_epoch,
                    epochs=epochs - epoch - 1,
                    use_dendrites=use_dendrites,
                )

    # PAI: Save graphs at end of training
    if use_dendrites and GPA is not None:
        GPA.pai_tracker.save_graphs()
        print(f"PAI graphs saved to {save_name}/ folder")

    return model


def main():
    """Parse arguments and run training/evaluation."""
    parser = argparse.ArgumentParser(description="ViT tiny classifier on RVL-CDIP dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--dataset", type=str, default="aharley/rvl_cdip", help="HF dataset identifier")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation")
    parser.add_argument("--training-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    # PAI option
    parser.add_argument("--use-dendrites", action="store_true", help="Enable PerforatedAI dendrites")
    parser.add_argument("--save-name", type=str, default="vit_rvlcdip", help="Save name for PAI outputs")
    # Reproducibility
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    print("Loading processor and model...")
    processor = load_processor()
    model = load_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize PAI if dendrites are enabled
    if args.use_dendrites:
        print("Initializing PerforatedAI...")
        init_pai()
        GPA.pc.set_input_dimensions([-1, -1, 0])
        model = UPA.initialize_pai(
            model,
            doing_pai=True,
            save_name=args.save_name,
            making_graphs=True,
            maximizing_score=True,
        )
        configure_pai_dimensions(model)

    model.to(device)

    if args.train:
        print(f"Starting training on '{args.dataset}'...")
        model = train(
            model,
            args.batch_size,
            device,
            processor,
            epochs=args.training_epochs,
            lr=args.lr,
            max_samples=args.max_samples,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
            dataset_name=args.dataset,
            use_dendrites=args.use_dendrites,
            save_name=args.save_name,
            streaming=args.streaming,
        )

    if args.eval and not args.train:
        # Standalone eval (training already includes validation)
        print(f"Loading test dataset '{args.dataset}'...")
        test_dataset = load_dataset(args.dataset, split="test", streaming=args.streaming, trust_remote_code=True)
        evaluate(model, test_dataset, args.batch_size, device, processor, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
