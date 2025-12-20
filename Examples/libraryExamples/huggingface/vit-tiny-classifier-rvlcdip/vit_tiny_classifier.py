from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torch
import argparse
import random
import numpy as np
import time
import os

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


def _prefetch_worker(dataset_name, split, stream_batch_size, train_batch_size, max_samples, processor, queue, worker_id, num_workers):
    """Separate process that fetches and processes batches (module-level for pickling)."""
    from datasets import load_dataset as hf_load_dataset

    dataset = hf_load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        trust_remote_code=True
    )
    batched_dataset = dataset.batch(batch_size=stream_batch_size)

    samples_seen = 0
    batch_idx = 0

    # Use manual iteration to catch HuggingFace-level decoding errors
    batch_iter = iter(batched_dataset)
    while True:
        # Try to get next batch - this is where HF decodes images and can fail
        try:
            batch = next(batch_iter)
        except StopIteration:
            break
        except Exception as e:
            print(f"[Worker {worker_id}] Skipping corrupt batch from HF iterator: {type(e).__name__}: {e}")
            continue

        # Simple sharding: each worker handles every nth batch
        if batch_idx % num_workers != worker_id:
            batch_idx += 1
            continue
        batch_idx += 1

        if max_samples and samples_seen >= max_samples // num_workers:
            break

        try:
            images = batch["image"]
            labels = batch["label"]

            # Convert images to RGB and process as a batch
            rgb_images = []
            valid_labels = []
            for idx, (img, label) in enumerate(zip(images, labels)):
                try:
                    rgb_images.append(img.convert("RGB"))
                    valid_labels.append(label)
                except Exception as e:
                    print(f"[Worker {worker_id}] Skipping corrupt image (batch idx {idx}): {type(e).__name__}: {e}")
                    continue

            if not rgb_images:
                continue

            # Process entire batch at once
            pixel_values = processor(rgb_images, return_tensors="pt")["pixel_values"]
            labels_tensor = torch.tensor(valid_labels, dtype=torch.long)

            # Split into training-sized batches and put directly in queue
            for i in range(0, len(valid_labels), train_batch_size):
                end_idx = min(i + train_batch_size, len(valid_labels))
                queue.put((pixel_values[i:end_idx], labels_tensor[i:end_idx]))

            samples_seen += len(valid_labels)
        except Exception as e:
            print(f"[Worker {worker_id}] Skipping batch due to error: {type(e).__name__}: {e}")
            continue

    queue.put(None)  # Signal this worker is done


class BatchIterator:
    """High-performance batch iterator with multiple prefetch workers.

    Yields ready-to-train (pixel_values, labels) batches directly.
    No DataLoader overhead - batches go straight from workers to training.
    """

    def __init__(self, dataset_name, split, processor, train_batch_size=64,
                 stream_batch_size=512, max_samples=None, num_workers=2, queue_size=8):
        self.dataset_name = dataset_name
        self.split = split
        self.processor = processor
        self.train_batch_size = train_batch_size
        self.stream_batch_size = stream_batch_size
        self.max_samples = max_samples
        self.num_workers = num_workers
        self.queue_size = queue_size

        # Known dataset sizes
        self._known_sizes = {
            ("aharley/rvl_cdip", "train"): 320000,
            ("aharley/rvl_cdip", "test"): 40000,
        }

    def __len__(self):
        key = (self.dataset_name, self.split)
        size = self._known_sizes.get(key, 100000)
        if self.max_samples:
            size = min(size, self.max_samples)
        return (size + self.train_batch_size - 1) // self.train_batch_size

    def __iter__(self):
        from multiprocessing import Process, Queue as MPQueue

        # Shared queue for all workers
        prefetch_queue = MPQueue(maxsize=self.queue_size)

        # Start multiple prefetch workers
        workers = []
        for worker_id in range(self.num_workers):
            proc = Process(
                target=_prefetch_worker,
                args=(
                    self.dataset_name, self.split, self.stream_batch_size,
                    self.train_batch_size, self.max_samples, self.processor,
                    prefetch_queue, worker_id, self.num_workers
                ),
                daemon=True
            )
            proc.start()
            workers.append(proc)

        # Track how many workers have finished
        workers_done = 0

        # Consume batches from queue
        while workers_done < self.num_workers:
            item = prefetch_queue.get()
            if item is None:
                workers_done += 1
                continue
            yield item  # (pixel_values, labels) ready for training

        # Cleanup
        for proc in workers:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()


class CachedDataset(Dataset):
    """PyTorch Dataset that loads preprocessed data from disk cache.

    Much faster than streaming - loads from local SSD instead of network.
    """

    def __init__(self, cache_path):
        self.cache_path = cache_path
        data = torch.load(cache_path)
        self.pixel_values = data["pixel_values"]
        self.labels = data["labels"]
        print(f"Loaded {len(self.labels)} samples from cache: {cache_path}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pixel_values[idx], self.labels[idx]


def preprocess_and_cache(dataset_name, split, processor, cache_dir, max_samples=None, batch_size=256):
    """Download dataset via streaming and preprocess to local cache.

    Uses streaming to avoid downloading the full dataset - only fetches what's needed.
    """
    from datasets import load_dataset
    from tqdm import tqdm

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{split}.pt")

    if os.path.exists(cache_path):
        print(f"Cache exists: {cache_path}")
        return cache_path

    print(f"Streaming and caching {split} split (max_samples={max_samples})...")

    # Use streaming to avoid downloading full dataset
    dataset = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
    batched_dataset = dataset.batch(batch_size=batch_size)

    all_pixels = []
    all_labels = []
    samples_seen = 0
    failed = 0

    # Stream and process batches
    for batch in tqdm(batched_dataset, desc=f"Caching {split}"):
        if max_samples and samples_seen >= max_samples:
            break

        images = batch["image"]
        labels = batch["label"]

        rgb_images = []
        valid_labels = []
        for idx, (img, label) in enumerate(zip(images, labels)):
            if max_samples and samples_seen + len(valid_labels) >= max_samples:
                break
            try:
                rgb_images.append(img.convert("RGB"))
                valid_labels.append(label)
            except Exception as e:
                print(f"Skipping corrupt image (sample {samples_seen + idx}): {type(e).__name__}: {e}")
                failed += 1
                continue

        if rgb_images:
            pixel_values = processor(rgb_images, return_tensors="pt")["pixel_values"]
            all_pixels.append(pixel_values)
            all_labels.extend(valid_labels)
            samples_seen += len(valid_labels)

    # Concatenate all batches
    pixel_tensor = torch.cat(all_pixels, dim=0)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)

    # Save to cache
    torch.save({
        "pixel_values": pixel_tensor,
        "labels": label_tensor
    }, cache_path)

    size_gb = os.path.getsize(cache_path) / 1e9
    print(f"Saved {len(all_labels)} samples to {cache_path} ({size_gb:.2f} GB, {failed} failed)")

    return cache_path


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

    if use_dendrites and GPA is not None:
        GPA.pai_tracker.set_optimizer_instance(optimizer)

    return optimizer, scheduler


def evaluate(model, dataloader, device):
    """Evaluate the model and return accuracy."""
    correct = 0
    total = 0
    model.eval()

    with torch.no_grad():
        for pixel_batch, label_batch in dataloader:
            pixel_batch = pixel_batch.to(device, non_blocking=True)
            label_batch = label_batch.to(device, non_blocking=True)

            if device.type == "cuda":
                with autocast("cuda"):
                    outputs = model(pixel_batch)
            else:
                outputs = model(pixel_batch)

            preds = outputs.logits.argmax(-1)
            correct += int((preds == label_batch).sum().item())
            total += label_batch.size(0)

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Accuracy: {accuracy * 100:.2f}% on {total} samples")
    return accuracy


def configure_pai_dimensions(model):
    """Configure PAI input/output dimensions for ViT layers."""
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

    try:
        for layer in model.vit.encoder.layer:
            if hasattr(layer.attention.attention, "query"):
                if hasattr(layer.attention.attention.query, "set_this_output_dimensions"):
                    layer.attention.attention.query.set_this_output_dimensions([-1, -1, 0])
            if hasattr(layer.attention.attention, "key"):
                if hasattr(layer.attention.attention.key, "set_this_output_dimensions"):
                    layer.attention.attention.key.set_this_output_dimensions([-1, -1, 0])
            if hasattr(layer.attention.attention, "value"):
                if hasattr(layer.attention.attention.value, "set_this_output_dimensions"):
                    layer.attention.attention.value.set_this_output_dimensions([-1, -1, 0])
            if hasattr(layer.attention.output.dense, "set_this_output_dimensions"):
                layer.attention.output.dense.set_this_output_dimensions([-1, -1, 0])
            if hasattr(layer.intermediate.dense, "set_this_output_dimensions"):
                layer.intermediate.dense.set_this_output_dimensions([-1, -1, 0])
            if hasattr(layer.output.dense, "set_this_output_dimensions"):
                layer.output.dense.set_this_output_dimensions([-1, -1, 0])
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
    stream_batch_size=512,
    num_workers=2,
    queue_size=8,
    cache_dir=None,
):
    """Train the model using cached data or HuggingFace streaming."""
    criterion = CrossEntropyLoss()
    scaler = GradScaler("cuda") if device.type == "cuda" else None

    if scaler:
        print("Using mixed precision (fp16) training")

    # Determine data loading mode
    use_cache = cache_dir is not None

    if use_cache:
        # Preprocess and cache if needed
        train_cache = preprocess_and_cache(dataset_name, "train", processor, cache_dir, max_samples)
        val_cache = preprocess_and_cache(dataset_name, "test", processor, cache_dir, max_samples)

        train_dataset = CachedDataset(train_cache)
        val_dataset = CachedDataset(val_cache)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

        steps_per_epoch = len(train_loader)
        print(f"Using cached data: {len(train_dataset)} train, {len(val_dataset)} val samples")
    else:
        # Streaming mode
        print(f"Using {num_workers} prefetch workers (stream_batch={stream_batch_size}, queue={queue_size})")
        steps_per_epoch = (320000 + batch_size - 1) // batch_size  # Estimate for rvl_cdip
        if max_samples:
            steps_per_epoch = (max_samples + batch_size - 1) // batch_size

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
        if use_dendrites and GPA is not None:
            GPA.pai_tracker.start_epoch()

        print(f"\nEpoch {epoch+1}/{epochs}")
        total_loss = 0.0
        total = 0
        batch_num = 0
        model.train()

        last_log_time = time.time()

        # Get data iterator based on mode
        if use_cache:
            data_iter = train_loader
        else:
            data_iter = BatchIterator(
                dataset_name, "train", processor,
                train_batch_size=batch_size,
                stream_batch_size=stream_batch_size,
                max_samples=max_samples,
                num_workers=num_workers,
                queue_size=queue_size
            )

        for pixel_batch, label_batch in data_iter:
            pixel_batch = pixel_batch.to(device)
            label_batch = label_batch.to(device)

            optimizer.zero_grad()

            if scaler:
                with autocast("cuda"):
                    outputs = model(pixel_batch)
                    loss = criterion(outputs.logits, label_batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
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
                current_time = time.time()
                elapsed = current_time - last_log_time
                samples_per_sec = (10 * batch_size) / elapsed if elapsed > 0 else 0
                current_lr = scheduler.get_last_lr()[0]
                print(f"  Step {global_step}: loss={loss.item():.4f}, lr={current_lr:.6f}, "
                      f"samples/sec={samples_per_sec:.1f}")
                last_log_time = current_time

        avg_loss = total_loss / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}, samples: {total}")

        # Validation
        print(f"Running validation...")
        if use_cache:
            accuracy = evaluate(model, val_loader, device)
        else:
            val_iter = BatchIterator(
                dataset_name, "test", processor,
                train_batch_size=batch_size,
                stream_batch_size=stream_batch_size,
                max_samples=max_samples,
                num_workers=num_workers,
                queue_size=queue_size
            )
            accuracy = evaluate(model, val_iter, device)

        if use_dendrites and GPA is not None:
            GPA.pai_tracker.set_optimizer_instance(optimizer)
            model, _, restructured = GPA.pai_tracker.add_validation_score(accuracy, model)

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

    if use_dendrites and GPA is not None:
        GPA.pai_tracker.save_graphs()
        print(f"PAI graphs saved to {save_name}/ folder")

    return model


def main():
    """Parse arguments and run training/evaluation."""
    parser = argparse.ArgumentParser(description="ViT tiny classifier on RVL-CDIP dataset")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--dataset", type=str, default="aharley/rvl_cdip", help="HF dataset identifier")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--eval", action="store_true", help="Run evaluation only")
    parser.add_argument("--training-epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Warmup ratio")
    # Data loading options
    parser.add_argument("--cache-dir", type=str, default=None, help="Cache directory for preprocessed data (faster, uses ~15GB)")
    parser.add_argument("--stream-batch-size", type=int, default=512, help="HF streaming fetch batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of prefetch worker processes")
    parser.add_argument("--queue-size", type=int, default=8, help="Prefetch queue size")
    # PAI options
    parser.add_argument("--use-dendrites", action="store_true", help="Enable PerforatedAI dendrites")
    parser.add_argument("--save-name", type=str, default="vit_rvlcdip", help="Save name for PAI outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    set_seed(args.seed)

    print("Loading processor and model...")
    processor = load_processor()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print("Using Apple Metal GPU (MPS)")

    model = load_model()

    if args.use_dendrites:
        print("Initializing PerforatedAI...")
        init_pai()
        GPA.pc.set_input_dimensions([-1, -1, 0])
        GPA.pc.set_testing_dendrite_capacity(False)
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
        print(f"\nStarting training on '{args.dataset}'...")
        print(f"  Batch size: {args.batch_size}")
        if args.cache_dir:
            print(f"  Mode: CACHED (fast) - {args.cache_dir}")
        else:
            print(f"  Mode: STREAMING - workers={args.num_workers}, queue={args.queue_size}")

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
            stream_batch_size=args.stream_batch_size,
            num_workers=args.num_workers,
            queue_size=args.queue_size,
            cache_dir=args.cache_dir,
        )

    if args.eval and not args.train:
        print(f"\nRunning evaluation on '{args.dataset}'...")
        if args.cache_dir:
            cache_path = os.path.join(args.cache_dir, "test.pt")
            if os.path.exists(cache_path):
                val_dataset = CachedDataset(cache_path)
                val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
                evaluate(model, val_loader, device)
            else:
                print(f"Cache not found: {cache_path}. Run with --train first to create cache.")
        else:
            val_iter = BatchIterator(
                args.dataset, "test", processor,
            train_batch_size=args.batch_size,
            stream_batch_size=args.stream_batch_size,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            queue_size=args.queue_size
        )
        evaluate(model, val_iter, device)


if __name__ == "__main__":
    main()
