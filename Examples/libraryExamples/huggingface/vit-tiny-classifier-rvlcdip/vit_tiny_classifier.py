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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import torch
import argparse
import random
import os
import time
from tqdm import tqdm

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


class PreprocessedDataset(torch.utils.data.IterableDataset):
    """PyTorch IterableDataset for preprocessed sharded data. Loads one shard at a time with prefetching."""

    def __init__(self, data_dir, split="train"):
        self.data_dir = os.path.join(data_dir, split)

        # Load metadata
        metadata_path = os.path.join(self.data_dir, "metadata.pt")
        if os.path.exists(metadata_path):
            self.metadata = torch.load(metadata_path)
            self.num_samples = self.metadata["num_samples"]
            self.shard_size = self.metadata["shard_size"]
            self.num_shards = self.metadata["num_shards"]
            self.compressed = self.metadata.get("compressed", False)
            self.fp16 = self.metadata.get("fp16", False)
        else:
            # Fallback for old format (individual .pt files)
            self.files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.pt') and not f.startswith('shard') and f != 'metadata.pt'])
            self.num_samples = len(self.files)
            self.shard_size = 1
            self.num_shards = self.num_samples
            self.compressed = False
            self.fp16 = False
            self.metadata = None

        print(f"Found {self.num_samples} preprocessed samples in {self.data_dir} "
              f"(shards={self.num_shards}, fp16={self.fp16}, compressed={self.compressed})")

    def __len__(self):
        return self.num_samples

    def _load_shard(self, shard_idx):
        """Load a shard file into memory."""
        shard_path = os.path.join(self.data_dir, f"shard_{shard_idx:05d}.pt")

        if self.compressed:
            import gzip
            import pickle
            with gzip.open(shard_path + ".gz", "rb") as f:
                data = pickle.load(f)
        else:
            data = torch.load(shard_path)

        return data

    def __iter__(self):
        import threading
        from queue import Queue

        if self.metadata is None:
            # Old format - individual files
            for f in self.files:
                data = torch.load(os.path.join(self.data_dir, f))
                pixel_values = data["pixel_values"]
                if pixel_values.dtype == torch.float16:
                    pixel_values = pixel_values.float()
                yield pixel_values, data["label"]
        else:
            # Sharded format with prefetching
            prefetch_queue = Queue(maxsize=2)  # Prefetch up to 2 shards ahead

            def prefetch_worker():
                """Background thread to load shards."""
                for shard_idx in range(self.num_shards):
                    shard_data = self._load_shard(shard_idx)
                    prefetch_queue.put(shard_data)
                prefetch_queue.put(None)  # Signal end

            # Start prefetch thread
            prefetch_thread = threading.Thread(target=prefetch_worker, daemon=True)
            prefetch_thread.start()

            # Consume shards from queue
            while True:
                shard_data = prefetch_queue.get()
                if shard_data is None:
                    break

                pixel_values = shard_data["pixel_values"]
                labels = shard_data["labels"]

                # Convert fp16 to fp32
                if pixel_values.dtype == torch.float16:
                    pixel_values = pixel_values.float()

                # Yield each sample from the shard
                for i in range(len(labels)):
                    yield pixel_values[i], labels[i]

            prefetch_thread.join()


class GPUPreloadedDataset(Dataset):
    """Dataset that keeps all data on GPU for maximum speed."""

    def __init__(self, pixel_values, labels):
        self.pixel_values = pixel_values
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.pixel_values[idx], self.labels[idx]


def preprocess_and_save(dataset_name, processor, output_dir, split="train", max_samples=None,
                        shard_size=5000, use_fp16=True, compress=True):
    """Preprocess dataset and save as optimized sharded files.

    Args:
        shard_size: Number of samples per shard file (larger = fewer files, faster loading)
        use_fp16: Save in half precision (50% smaller, minimal quality loss for images)
        compress: Use gzip compression (slower to save, faster to load, ~30% smaller)
    """
    save_dir = os.path.join(output_dir, split)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {split} dataset...")
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    num_shards = (num_samples + shard_size - 1) // shard_size
    print(f"Preprocessing {num_samples} samples into {num_shards} shards (fp16={use_fp16}, compress={compress})...")

    saved = 0
    failed = 0
    shard_pixels = []
    shard_labels = []
    shard_idx = 0
    total_bytes = 0

    for idx in tqdm(range(num_samples), desc=f"Preprocessing {split}"):
        try:
            example = dataset[idx]
            img = example["image"]
            pixel_values = processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]

            # Convert to fp16 to save space
            if use_fp16:
                pixel_values = pixel_values.half()

            shard_pixels.append(pixel_values)
            shard_labels.append(example["label"])
            saved += 1
        except Exception:
            failed += 1
            continue

        # Save shard when full
        if len(shard_pixels) >= shard_size:
            shard_path = os.path.join(save_dir, f"shard_{shard_idx:05d}.pt")
            shard_data = {
                "pixel_values": torch.stack(shard_pixels),
                "labels": torch.tensor(shard_labels, dtype=torch.long),
                "fp16": use_fp16
            }
            if compress:
                import gzip
                import pickle
                with gzip.open(shard_path + ".gz", "wb", compresslevel=1) as f:
                    pickle.dump(shard_data, f)
                total_bytes += os.path.getsize(shard_path + ".gz")
            else:
                torch.save(shard_data, shard_path)
                total_bytes += os.path.getsize(shard_path)

            shard_pixels = []
            shard_labels = []
            shard_idx += 1

    # Save remaining samples
    if shard_pixels:
        shard_path = os.path.join(save_dir, f"shard_{shard_idx:05d}.pt")
        shard_data = {
            "pixel_values": torch.stack(shard_pixels),
            "labels": torch.tensor(shard_labels, dtype=torch.long),
            "fp16": use_fp16
        }
        if compress:
            import gzip
            import pickle
            with gzip.open(shard_path + ".gz", "wb", compresslevel=1) as f:
                pickle.dump(shard_data, f)
            total_bytes += os.path.getsize(shard_path + ".gz")
        else:
            torch.save(shard_data, shard_path)
            total_bytes += os.path.getsize(shard_path)

    # Save metadata
    metadata = {
        "num_samples": saved,
        "num_shards": shard_idx + 1,
        "shard_size": shard_size,
        "fp16": use_fp16,
        "compressed": compress
    }
    torch.save(metadata, os.path.join(save_dir, "metadata.pt"))

    print(f"Saved {saved} samples in {shard_idx + 1} shards ({failed} failed)")
    print(f"Total size: {total_bytes / 1e9:.2f} GB ({total_bytes / saved / 1024:.1f} KB/sample)")
    return saved


def preprocess_to_single_file(dataset_name, processor, output_dir, split="train", max_samples=None, use_fp16=True):
    """Preprocess dataset and save as a single .pt file (for validation set)."""
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{split}.pt")

    print(f"Loading {split} dataset...")
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)

    num_samples = min(len(dataset), max_samples) if max_samples else len(dataset)
    print(f"Preprocessing {num_samples} samples into single file (fp16={use_fp16})...")

    all_pixels = []
    all_labels = []
    failed = 0

    for idx in tqdm(range(num_samples), desc=f"Preprocessing {split}"):
        try:
            example = dataset[idx]
            img = example["image"]
            pixel_values = processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0]
            if use_fp16:
                pixel_values = pixel_values.half()
            all_pixels.append(pixel_values)
            all_labels.append(example["label"])
        except Exception:
            failed += 1
            continue

    # Stack into tensors
    pixel_tensor = torch.stack(all_pixels)
    label_tensor = torch.tensor(all_labels, dtype=torch.long)

    torch.save({
        "pixel_values": pixel_tensor,
        "labels": label_tensor,
        "fp16": use_fp16
    }, save_path)

    print(f"Saved {len(all_labels)} samples to {save_path} ({failed} failed)")
    print(f"File size: {os.path.getsize(save_path) / 1e9:.2f} GB")
    return len(all_labels)


def load_preprocessed_single_file(data_dir, split="train", device=None):
    """Load preprocessed data from single .pt file, optionally to GPU."""
    load_path = os.path.join(data_dir, f"{split}.pt")
    print(f"Loading preprocessed data from {load_path}...")
    data = torch.load(load_path)

    pixel_values = data["pixel_values"]
    labels = data["labels"]

    # Convert fp16 back to fp32
    if pixel_values.dtype == torch.float16:
        print("Converting fp16 to fp32...")
        pixel_values = pixel_values.float()

    if device is not None and device.type == "cuda":
        print(f"Moving {split} data to GPU...")
        pixel_values = pixel_values.to(device)
        labels = labels.to(device)

    return GPUPreloadedDataset(pixel_values, labels)


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
            pixel_batch = torch.from_numpy(np.stack(batch_pixels))
            label_batch = torch.tensor(batch_labels, dtype=torch.long)
            # Use pinned memory for faster GPU transfer (only works with CUDA)
            if device.type == "cuda":
                pixel_batch = pixel_batch.pin_memory().to(device, non_blocking=True)
                label_batch = label_batch.pin_memory().to(device, non_blocking=True)
            else:
                pixel_batch = pixel_batch.to(device)
                label_batch = label_batch.to(device)
            yield pixel_batch, label_batch
            batch_pixels = []
            batch_labels = []

        if max_samples is not None and seen >= max_samples:
            break

    # Yield remaining samples
    if batch_pixels:
        pixel_batch = torch.from_numpy(np.stack(batch_pixels))
        label_batch = torch.tensor(batch_labels, dtype=torch.long)
        if device.type == "cuda":
            pixel_batch = pixel_batch.pin_memory().to(device, non_blocking=True)
            label_batch = label_batch.pin_memory().to(device, non_blocking=True)
        else:
            pixel_batch = pixel_batch.to(device)
            label_batch = label_batch.to(device)
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


def evaluate_dataloader(model, dataloader, device, use_fp16=True, data_on_gpu=False):
    """Evaluate the model using a DataLoader (faster than batch_iterator)."""
    correct = 0
    total = 0
    batch_num = 0
    model.eval()

    with torch.no_grad():
        for pixel_batch, label_batch in dataloader:
            if not data_on_gpu:
                pixel_batch = pixel_batch.to(device, non_blocking=True)
                label_batch = label_batch.to(device, non_blocking=True)

            if use_fp16 and device.type == "cuda":
                with autocast():
                    outputs = model(pixel_batch)
            else:
                outputs = model(pixel_batch)

            preds = outputs.logits.argmax(-1)
            correct += int((preds == label_batch).sum().item())
            total += label_batch.size(0)
            batch_num += 1

            if batch_num % 10 == 0:
                print(f"Eval batch {batch_num}: samples={total}, accuracy={correct/total*100:.2f}%")

    accuracy = (correct / total) if total > 0 else 0.0
    print(f"Test Accuracy: {accuracy * 100:.2f}% on {total} samples")
    return accuracy


def configure_pai_dimensions(model):
    """Configure PAI input/output dimensions for ViT layers."""
    # Patch embeddings projection: input is [batch, channels, height, width]
    try:
        patch_proj = model.vit.embeddings.patch_embeddings.projection
        if hasattr(patch_proj, "set_this_input_dimensions"):
            patch_proj.set_this_input_dimensions([-1, 0, -1, -1])
    except AttributeError:
        pass

    # Classifier: input is [batch, hidden_dim]
    try:
        clf = model.classifier
        if hasattr(clf, "set_this_input_dimensions"):
            clf.set_this_input_dimensions([-1, 0])
    except AttributeError:
        pass

    # Encoder layers: output is [batch, seq_len, hidden_dim]
    # Set output dimensions for all encoder layers
    try:
        for layer in model.vit.encoder.layer:
            # Attention query/key/value projections
            if hasattr(layer.attention.attention, "query"):
                if hasattr(layer.attention.attention.query, "set_this_output_dimensions"):
                    layer.attention.attention.query.set_this_output_dimensions([-1, -1, 0])
            if hasattr(layer.attention.attention, "key"):
                if hasattr(layer.attention.attention.key, "set_this_output_dimensions"):
                    layer.attention.attention.key.set_this_output_dimensions([-1, -1, 0])
            if hasattr(layer.attention.attention, "value"):
                if hasattr(layer.attention.attention.value, "set_this_output_dimensions"):
                    layer.attention.attention.value.set_this_output_dimensions([-1, -1, 0])
            # Attention output dense
            if hasattr(layer.attention.output.dense, "set_this_output_dimensions"):
                layer.attention.output.dense.set_this_output_dimensions([-1, -1, 0])
            # MLP intermediate dense
            if hasattr(layer.intermediate.dense, "set_this_output_dimensions"):
                layer.intermediate.dense.set_this_output_dimensions([-1, -1, 0])
            # MLP output dense
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
    streaming=False,
    preprocessed_dir=None,
    num_workers=4,
    use_fp16=True,
    preload_val_gpu=False,
):
    """Train the model, optionally with PAI dendrites.

    Args:
        preprocessed_dir: If set, load preprocessed data from this directory
        num_workers: Number of DataLoader workers for parallel loading
        use_fp16: Use mixed precision training (recommended for GPU)
        preload_val_gpu: Load validation set entirely to GPU
    """
    criterion = CrossEntropyLoss()

    # Setup mixed precision
    scaler = GradScaler() if use_fp16 and device.type == "cuda" else None
    if scaler:
        print("Using mixed precision (fp16) training")

    # Determine data loading strategy
    use_dataloader = preprocessed_dir is not None
    train_loader = None
    val_loader = None
    val_gpu_dataset = None

    if use_dataloader:
        # Load from preprocessed files using DataLoader
        print(f"Loading preprocessed data from {preprocessed_dir}")
        train_dataset = PreprocessedDataset(preprocessed_dir, split="train")
        # IterableDataset doesn't support shuffle - data is read sequentially by shard
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,  # IterableDataset works best with 0 workers for sharded data
            pin_memory=True,
        )
        steps_per_epoch = (len(train_dataset) + batch_size - 1) // batch_size

        # Validation: check if test.pt exists (single file) or test/ directory (sharded)
        test_single_file = os.path.join(preprocessed_dir, "test.pt")
        test_dir = os.path.join(preprocessed_dir, "test")

        if preload_val_gpu and device.type == "cuda":
            val_gpu_dataset = load_preprocessed_single_file(preprocessed_dir, split="test", device=device)
            val_loader = DataLoader(val_gpu_dataset, batch_size=batch_size, shuffle=False)
        elif os.path.exists(test_single_file):
            # Load from single file
            val_gpu_dataset = load_preprocessed_single_file(preprocessed_dir, split="test", device=None)
            val_loader = DataLoader(val_gpu_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        elif os.path.exists(test_dir):
            # Load from sharded directory
            val_dataset = PreprocessedDataset(preprocessed_dir, split="test")
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=True,
            )
        else:
            raise FileNotFoundError(f"No preprocessed test data found at {test_single_file} or {test_dir}")
    elif not streaming:
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

        # Timing variables
        last_log_time = time.time()

        # Get data iterator based on loading strategy
        if use_dataloader:
            data_iter = train_loader
        elif streaming:
            epoch_train_dataset = load_dataset(dataset_name, split="train", streaming=True, trust_remote_code=True)
            data_iter = batch_iterator(epoch_train_dataset, batch_size, device, processor, max_samples)
        else:
            data_iter = batch_iterator(train_dataset, batch_size, device, processor, max_samples)

        for batch_data in data_iter:
            if use_dataloader:
                pixel_batch, label_batch = batch_data
                if device.type == "cuda" and not preload_val_gpu:
                    pixel_batch = pixel_batch.to(device, non_blocking=True)
                    label_batch = label_batch.to(device, non_blocking=True)
            else:
                pixel_batch, label_batch = batch_data

            optimizer.zero_grad()

            # Mixed precision forward pass
            if scaler:
                with autocast():
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
                print(f"  Step {global_step}: batch={batch_num}, loss={loss.item():.4f}, lr={current_lr:.6f}, "
                      f"time={elapsed:.2f}s, samples/sec={samples_per_sec:.1f}")
                last_log_time = current_time

        avg_loss = total_loss / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}, samples: {total}")

        # Validation after each epoch
        print(f"Running validation after epoch {epoch+1}...")
        if use_dataloader:
            accuracy = evaluate_dataloader(model, val_loader, device, use_fp16, preload_val_gpu or val_gpu_dataset is not None)
        elif streaming:
            epoch_val_dataset = load_dataset(dataset_name, split="test", streaming=True, trust_remote_code=True)
            accuracy = evaluate(model, epoch_val_dataset, batch_size, device, processor, max_samples)
        else:
            accuracy = evaluate(model, val_dataset, batch_size, device, processor, max_samples)

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
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (use 256+ for A100)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for testing)")
    parser.add_argument("--dataset", type=str, default="aharley/rvl_cdip", help="HF dataset identifier")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset (slow, use --preprocess instead)")
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

    # Performance options
    parser.add_argument("--preprocess", action="store_true", help="Preprocess dataset and save to disk")
    parser.add_argument("--preprocess-dir", type=str, default="./preprocessed_data", help="Directory for preprocessed data")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0 for main process)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use mixed precision training")
    parser.add_argument("--no-fp16", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--preload-val-gpu", action="store_true", help="Preload validation set to GPU")

    args = parser.parse_args()

    # Handle fp16 flag
    use_fp16 = args.fp16 and not args.no_fp16

    # Set seed for reproducibility
    set_seed(args.seed)

    print("Loading processor and model...")
    processor = load_processor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Preprocessing mode: preprocess dataset and exit
    if args.preprocess:
        print(f"Preprocessing dataset to {args.preprocess_dir}...")
        preprocess_and_save(args.dataset, processor, args.preprocess_dir, split="train", max_samples=args.max_samples)
        preprocess_to_single_file(args.dataset, processor, args.preprocess_dir, split="test", max_samples=args.max_samples)
        print("Preprocessing complete! Run training with --preprocess-dir to use preprocessed data.")
        return

    model = load_model()

    # Initialize PAI if dendrites are enabled
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

    # Check if preprocessed data exists
    preprocessed_dir = None
    if os.path.exists(args.preprocess_dir) and os.path.exists(os.path.join(args.preprocess_dir, "train")):
        preprocessed_dir = args.preprocess_dir
        print(f"Using preprocessed data from {preprocessed_dir}")

    if args.train:
        print(f"Starting training on '{args.dataset}'...")
        if preprocessed_dir:
            print(f"  Using DataLoader with {args.num_workers} workers")
            print(f"  Mixed precision (fp16): {use_fp16}")
            print(f"  Batch size: {args.batch_size}")

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
            preprocessed_dir=preprocessed_dir,
            num_workers=args.num_workers,
            use_fp16=use_fp16,
            preload_val_gpu=args.preload_val_gpu,
        )

    if args.eval and not args.train:
        # Standalone eval (training already includes validation)
        print(f"Loading test dataset '{args.dataset}'...")
        if preprocessed_dir:
            val_dataset = PreprocessedDataset(preprocessed_dir, split="test")
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
            evaluate_dataloader(model, val_loader, device, use_fp16)
        else:
            test_dataset = load_dataset(args.dataset, split="test", streaming=args.streaming, trust_remote_code=True)
            evaluate(model, test_dataset, args.batch_size, device, processor, max_samples=args.max_samples)


if __name__ == "__main__":
    main()
