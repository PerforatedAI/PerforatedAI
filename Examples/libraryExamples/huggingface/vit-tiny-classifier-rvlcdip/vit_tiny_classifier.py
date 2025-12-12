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
from PIL import Image
import numpy as np
import torch
import argparse


def load_processor():
    """Load the image processor for the ViT model."""
    return AutoImageProcessor.from_pretrained("HAMMALE/vit-tiny-classifier-rvlcdip")

def load_model():
    """Load a ViT model with random weights using the config from a pretrained checkpoint."""
    config = AutoConfig.from_pretrained("HAMMALE/vit-tiny-classifier-rvlcdip")
    return AutoModelForImageClassification.from_config(config)
    #return AutoModelForImageClassification.from_pretrained("HAMMALE/vit-tiny-classifier-rvlcdip")

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
        except Exception as e:
            failed += 1
            continue
        try:
            transformed = example_transform(ex, processor)
            batch_pixels.append(transformed["pixel_values"])
            batch_labels.append(transformed["label"])
            seen += 1
        except Exception as e:
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

    if batch_pixels:
        pixel_batch = torch.from_numpy(np.stack(batch_pixels)).to(device)
        label_batch = torch.tensor(batch_labels, dtype=torch.long).to(device)
        batch_num += 1
        label_counts = Counter(batch_labels)
        print(f"Batch {batch_num}: Label distribution: " +
              ", ".join(f"{lbl}({label_counts[lbl]})" for lbl in sorted(label_counts)))
        yield pixel_batch, label_batch

    if failed > 0:
        print(f"Skipped {failed} examples due to processing errors.")

def create_optimizer_and_scheduler(
    model,
    lr,
    weight_decay,
    warmup_ratio,
    steps_per_epoch,
    epochs,
):
    """Create AdamW optimizer and cosine scheduler with warmup."""
    # Parameter groups: decay for weights, no decay for LayerNorm/bias
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

    print(
        f"Scheduler: total_steps={num_training_steps}, "
        f"warmup_steps={num_warmup_steps} ({warmup_ratio*100:.1f}% of training)"
    )

    return optimizer, scheduler

def evaluate(model, test_dataset, batch_size, device, processor, max_samples=None):
    """Evaluate the model on a test dataset and print progress and accuracy."""
    correct = 0
    total = 0
    batch_num = 0

    with torch.no_grad():
        for pixel_batch, label_batch in batch_iterator(test_dataset, batch_size, device, processor, max_samples):
            outputs = model(pixel_batch)
            preds = outputs.logits.argmax(-1)
            correct += int((preds == label_batch).sum().item())
            total += label_batch.size(0)
            batch_num += 1
            print(f"Processed batch {batch_num}: Total samples so far = {total}, Accuracy so far = {(correct / total) * 100:.2f}%")
    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Total samples evaluated: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Test Dataset Accuracy: {accuracy * 100:.2f}%")
    return accuracy

def train(
    model,
    train_dataset,
    batch_size,
    device,
    processor,
    epochs=1,
    lr=3e-4,          
    max_samples=None,
    weight_decay=0.05,
    warmup_ratio=0.1,
):
    """Train the model on the training dataset for a given number of epochs."""
    criterion = CrossEntropyLoss()
    model.train()

    # Estimate steps per epoch (needed for scheduler)
    if max_samples is not None:
        steps_per_epoch = (max_samples + batch_size - 1) // batch_size
    else:
        # Non-streaming dataset: we can use len(...)
        try:
            steps_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size
        except TypeError:
            # Streaming & no max_samples: fall back to a guess or make this a CLI arg
            steps_per_epoch = 1000 // batch_size
            print(
                "Warning: could not determine steps_per_epoch from dataset; "
                f"using approximate value {steps_per_epoch}."
            )

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
    )

    global_step = 0

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        total = 0
        batch_num = 0

        for pixel_batch, label_batch in batch_iterator(
            train_dataset,
            batch_size,
            device,
            processor,
            max_samples,
        ):
            optimizer.zero_grad()
            outputs = model(pixel_batch)
            loss = criterion(outputs.logits, label_batch)
            loss.backward()

            # Gradient clipping for stability with transformers
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
                print(
                    f"  Step {global_step}: "
                    f"Batch {batch_num}, loss={loss.item():.4f}, "
                    f"lr={current_lr:.6f}, total_examples={total}"
                )

        avg_loss = total_loss / total if total > 0 else 0.0
        print(
            f"Epoch {epoch+1} finished. "
            f"Average loss: {avg_loss:.4f}, Total examples processed: {total}"
        )

def main():
    """Parse arguments, load data/model, and run training/evaluation as requested."""
    parser = argparse.ArgumentParser(description="ViT tiny classifier on RVL-CDIP dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for inference/training")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after this many samples (useful for testing)")
    parser.add_argument("--dataset", type=str, default="aharley/rvl_cdip", help="HF dataset identifier to use")
    parser.add_argument("--streaming", action="store_true", help="Load dataset in streaming mode")
    parser.add_argument("--train", action="store_true",  help="Run training")
    parser.add_argument("--eval", action="store_true",  help="Run evaluation")
    parser.add_argument("--training-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    args = parser.parse_args()

    print("Loading processor and model...")
    processor = load_processor()
    model = load_model()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if args.train:
        print(f"Loading training dataset '{args.dataset}' (streaming={args.streaming})...")
        train_dataset = load_dataset(args.dataset, split="train", streaming=args.streaming, trust_remote_code=True)
        train(
            model,
            train_dataset,
            args.batch_size,
            device,
            processor,
            epochs=args.training_epochs,
            lr=args.lr,
            max_samples=args.max_samples,
            weight_decay=args.weight_decay,
            warmup_ratio=args.warmup_ratio,
        )
    if args.eval:
        print(f"Loading test dataset '{args.dataset}' (streaming={args.streaming})...")
        test_dataset = load_dataset(args.dataset, split="test", streaming=args.streaming, trust_remote_code=True)
        evaluate(model, test_dataset, args.batch_size, device, processor, max_samples=args.max_samples)

if __name__ == "__main__":
    main()