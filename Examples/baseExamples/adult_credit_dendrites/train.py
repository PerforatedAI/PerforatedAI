from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import types
from importlib.machinery import SourceFileLoader

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

CURRENT_DIR = Path(__file__).resolve().parent


# Search upward 
def _find_repo_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        direct = candidate / "perforatedai" / "__init__.py"
        if direct.exists():
            return candidate
        alt = candidate / "PerforatedAI"
        if (alt / "perforatedai" / "__init__.py").exists():
            return alt
    raise FileNotFoundError(
        f"Could not locate the local 'perforatedai' package when starting from {start}"
    )


REPO_ROOT = _find_repo_root(CURRENT_DIR)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

module_key = "perforatedai"
module_path = REPO_ROOT / "perforatedai" / "__init__.py"
if module_key in sys.modules:
    del sys.modules[module_key]
loader = SourceFileLoader(module_key, str(module_path))
module = types.ModuleType(loader.name)
module.__file__ = loader.path
module.__path__ = [str(module_path.parent)]
sys.modules[module_key] = module
loader.exec_module(module)

from metrics import accuracy_score, auc_score, f1_score_binary
from param_count import count_params

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

GPA.pc.set_using_safe_tensors(False)


DATA_CACHE_DIR = CURRENT_DIR / "data_cache"

DATASET_CONFIG = {
    "adult": {
        "target": "class",
        "positive_label": ">50K",
        "categorical_cols": [
            "workclass",
            "education",
            "marital-status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "native-country",
        ],
        "drop_columns": [],
        "local_glob": [
            "**/adult*.arff",
            "**/adult*.arff.gz",
            "**/adult*.csv",
            "**/adult*.csv.gz",
        ],
        "local_loader": "auto",
        "fetch_kwargs": {"name": "adult", "version": 2},
    },
    "credit": {
        "target": "default payment next month",
        "target_aliases": [
            "default.payment.next.month",
            "default_payment_next_month",
            "default payment next month",
            "y",
        ],
        "positive_label": 1,
        "categorical_cols": [
            "sex",
            "education",
            "marriage",
            "pay_0",
            "pay_2",
            "pay_3",
            "pay_4",
            "pay_5",
            "pay_6",
        ],
        "drop_columns": ["id"],
        "local_glob": [
            "**/default*credit*clients*.arff",
            "**/default*credit*clients*.arff.gz",
            "**/default*credit*clients*.csv",
            "**/default*credit*clients*.csv.gz",
        ],
        "local_loader": "auto",
        "fetch_kwargs": {
            "name": "default-of-credit-card-clients",
            "version": 1,
        },
    },
}


class TabularBinaryDataset(Dataset):
    """Torch dataset wrapping preprocessed tabular features and binary labels."""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        labels = labels.astype(np.float32).reshape(-1, 1)
        self.labels = torch.from_numpy(labels)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class AdultMLP(nn.Module):
    """Simple configurable MLP for Adult Income classification."""

    def __init__(self, input_dim: int, hidden_width: int, dropout: float):
        super().__init__()
        hidden_sizes = [
            hidden_width,
            hidden_width,
            max(1, hidden_width // 2),
        ]
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for size in hidden_sizes:
            linear = nn.Linear(prev_dim, size)
            layers.append(linear)
            prev_dim = size
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, 1)
        self.dropout_module = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear in self.hidden_layers:
            x = linear(x)
            x = F.relu(x, inplace=False)
            if self.dropout_module is not None:
                x = self.dropout_module(x)
        logits = self.output_layer(x)
        return logits.squeeze(-1)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dendritic MLP tabular benchmark")
    parser.add_argument(
        "--dataset",
        default="adult",
        choices=sorted(DATASET_CONFIG.keys()),
        help="Dataset to use (default: adult)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument(
        "--use-dendrites",
        dest="use_dendrites",
        action="store_true",
        help="Enable dendrites on hidden linear layers.",
    )
    parser.add_argument(
        "--no-dendrites",
        dest="use_dendrites",
        action="store_false",
        help="Disable dendrites (baseline).",
    )
    parser.set_defaults(use_dendrites=False)
    parser.add_argument(
        "--exclude-output-proj",
        dest="exclude_output_proj",
        action="store_true",
        default=True,
        help="Keep output projection vanilla (default).",
    )
    parser.add_argument(
        "--include-output-proj",
        dest="exclude_output_proj",
        action="store_false",
        help="Also dendritize the output layer.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device selection helper.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to store CSV logs and plots.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience on validation AUC.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional notes column to append to results CSV.",
    )
    parser.add_argument(
        "--max-dendrites",
        type=int,
        default=None,
        help="Optional cap on total dendrites (applies when --use-dendrites).",
    )
    parser.add_argument(
        "--fixed-switch-num",
        type=int,
        default=None,
        help="If set, force fixed PAI switch cadence with this many epochs.",
    )
    parser.add_argument(
        "--first-fixed-switch-num",
        type=int,
        default=None,
        help="Override first fixed switch length; defaults to --fixed-switch-num.",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preference == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def load_dataset(name: str, seed: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    import inspect
    from urllib.error import URLError

    import pandas as pd
    from scipy.io import arff  # type: ignore
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cfg = DATASET_CONFIG[name]

    frame = None
    local_patterns = cfg.get("local_glob")
    if local_patterns:
        if isinstance(local_patterns, (str, Path)):
            local_patterns = [str(local_patterns)]
        matches: List[Path] = []
        for pattern in local_patterns:
            matches.extend(DATA_CACHE_DIR.glob(pattern))
        if matches:
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            local_path = matches[0]
            loader_type = cfg.get("local_loader", "auto")
            suffixes = [suffix.lower() for suffix in local_path.suffixes]
            last_suffix = suffixes[-1] if suffixes else ""
            inner_suffix = suffixes[-2] if len(suffixes) > 1 else ""
            if loader_type == "auto":
                candidate = inner_suffix if last_suffix == ".gz" else last_suffix
                if candidate == ".arff":
                    loader_type = "arff_gz" if last_suffix == ".gz" else "arff"
                elif candidate in {".csv", ".txt"}:
                    loader_type = "csv_gz" if last_suffix == ".gz" else "csv"
            if loader_type in {"arff", "arff_gz"}:
                if loader_type == "arff_gz":
                    import gzip

                    with gzip.open(local_path, "rt", encoding="utf-8", errors="ignore") as handle:
                        data, _ = arff.loadarff(handle)
                else:
                    data, _ = arff.loadarff(local_path)
                frame = pd.DataFrame(data)
                for col in frame.columns:
                    if frame[col].dtype == object:
                        frame[col] = frame[col].apply(
                            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                        )
            elif loader_type in {"csv", "csv_gz"}:
                frame = pd.read_csv(local_path)
            else:
                raise ValueError(
                    f"Unsupported local loader '{loader_type}' for dataset '{name}'."
                )

    if frame is None:
        fetch_kwargs = cfg.get("fetch_kwargs", {})
        try:
            dataset = fetch_openml(
                fetch_kwargs.get("name", name),
                version=fetch_kwargs.get("version"),
                as_frame=True,
                data_home=str(DATA_CACHE_DIR),
            )
            frame = dataset.frame
        except URLError as err:
            raise RuntimeError(
                f"Failed to download the {name} dataset. Place a cached file under {DATA_CACHE_DIR}."
            ) from err

    if frame is None:
        raise RuntimeError(f"{name} dataset could not be loaded.")

    frame = frame.copy()
    frame.columns = [str(col) for col in frame.columns]

    def _normalize(label: str) -> str:
        return "".join(ch for ch in label.lower() if ch.isalnum())

    column_lookup = {_normalize(col): col for col in frame.columns}

    def resolve_single(label: str) -> str | None:
        return column_lookup.get(_normalize(label))

    drop_cols: List[str] = []
    for candidate in cfg.get("drop_columns", []):
        resolved = resolve_single(candidate)
        if resolved and resolved not in drop_cols:
            drop_cols.append(resolved)
    if drop_cols:
        frame = frame.drop(columns=drop_cols)

    frame = frame.dropna()

    target_candidates = [cfg["target"]] + cfg.get("target_aliases", [])
    target_col = None
    for candidate in target_candidates:
        resolved = resolve_single(candidate)
        if resolved:
            target_col = resolved
            break
    if target_col is None:
        raise KeyError(
            f"Target column not found for dataset '{name}'. Tried {target_candidates}."
        )

    features = frame.drop(columns=[target_col])
    labels_series = frame[target_col]

    positive_str = str(cfg["positive_label"]).strip().lower()

    categorical_cols: List[str] = []
    for candidate in cfg.get("categorical_cols", []):
        resolved = resolve_single(candidate)
        if (
            resolved
            and resolved in features.columns
            and resolved not in categorical_cols
        ):
            categorical_cols.append(resolved)
    for col in categorical_cols:
        features[col] = features[col].astype(str)

    continuous_cols = [col for col in features.columns if col not in categorical_cols]

    if continuous_cols:
        features[continuous_cols] = features[continuous_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        if features[continuous_cols].isnull().any().any():
            features = features.dropna()

    labels_series = labels_series.loc[features.index]
    labels = labels_series.apply(
        lambda x: 1 if str(x).strip().lower() == positive_str else 0
    ).astype(np.int32)
    labels = labels.to_numpy(dtype=np.int32)

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        labels,
        test_size=0.15,
        random_state=seed,
        stratify=labels,
    )
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features,
        train_labels,
        test_size=0.1764706,
        random_state=seed,
        stratify=train_labels,
    )

    scaler = None
    if continuous_cols:
        scaler = StandardScaler()
        scaler.fit(train_features[continuous_cols])

    encoder = None
    if categorical_cols:
        encoder_kwargs = {"handle_unknown": "ignore", "dtype": np.float32}
        if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
            encoder_kwargs["sparse_output"] = False
        else:
            encoder_kwargs["sparse"] = False
        encoder = OneHotEncoder(**encoder_kwargs)
        encoder.fit(train_features[categorical_cols])

    def transform(df: pd.DataFrame) -> np.ndarray:
        blocks: List[np.ndarray] = []
        if continuous_cols and scaler is not None:
            cont = scaler.transform(df[continuous_cols]).astype(np.float32)
            blocks.append(cont)
        if categorical_cols and encoder is not None:
            cat = encoder.transform(df[categorical_cols])
            if not isinstance(cat, np.ndarray):
                cat = cat.toarray()
            blocks.append(cat.astype(np.float32))
        if not blocks:
            raise ValueError("No features available after preprocessing.")
        if len(blocks) == 1:
            feats = blocks[0]
        else:
            feats = np.concatenate(blocks, axis=1)
        return feats.astype(np.float32)

    train_X = transform(train_features)
    val_X = transform(val_features)
    test_X = transform(test_features)

    return {
        "train": (train_X, train_labels),
        "val": (val_X, val_labels),
        "test": (test_X, test_labels),
    }


def build_dataloaders(
    splits: Dict[str, Tuple[np.ndarray, np.ndarray]], batch_size: int
) -> Dict[str, DataLoader]:
    train_dataset = TabularBinaryDataset(*splits["train"])
    val_dataset = TabularBinaryDataset(*splits["val"])
    test_dataset = TabularBinaryDataset(*splits["test"])

    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    }


def compute_metrics_from_logits(
    targets: torch.Tensor, logits: torch.Tensor
) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    y_true = targets.cpu().numpy().reshape(-1)
    y_prob = probs.cpu().numpy().reshape(-1)
    return {
        "auc": auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_prob),
        "f1": f1_score_binary(y_true, y_prob),
    }


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device).squeeze(-1)
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics_from_logits(targets_cat, logits_cat)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device).squeeze(-1)
            logits = model(features)
            loss = criterion(logits, labels)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics_from_logits(targets_cat, logits_cat)
    metrics["loss"] = total_loss / max(1, total_samples)
    return metrics


def prepare_dendritic_model(
    model: nn.Module,
    exclude_output_proj: bool,
    save_name: str,
) -> nn.Module:
    GPA.pc.set_testing_dendrite_capacity(False)
    GPA.pc.set_unwrapped_modules_confirmed(True)

    GPA.pc.set_modules_to_convert([nn.Linear])
    GPA.pc.set_modules_to_track([])
    GPA.pc.set_module_names_to_track([])
    GPA.pc.set_module_ids_to_convert([])
    GPA.pc.set_module_ids_to_track([])

    if exclude_output_proj:
        GPA.pc.set_module_ids_to_track([".output_layer"])

    model = UPA.initialize_pai(
        model,
        doing_pai=True,
        save_name=save_name,
        making_graphs=True,
        maximizing_score=True,
    )
    return model


def configure_pai_optim_scheduler(
    model: nn.Module, lr: float
) -> Tuple[torch.optim.Optimizer, ReduceLROnPlateau]:
    GPA.pai_tracker.set_optimizer(Adam)
    GPA.pai_tracker.set_scheduler(ReduceLROnPlateau)
    optimizer_args = {"lr": lr}
    scheduler_args = {"mode": "max", "factor": 0.5, "patience": 2}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
        model, optimizer_args, scheduler_args
    )
    return optimizer, scheduler


def update_params_progression(path: Path, history: List[Tuple[int, int]]) -> None:
    if not history:
        return
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "params"])
        for epoch, params in history:
            writer.writerow([epoch, params])


def append_best_scores_row(path: Path, row: Dict[str, object]) -> None:
    fieldnames = [
        "dataset",
        "model_id",
        "use_dendrites",
        "exclude_output",
        "width",
        "dropout",
        "params",
        "epochs_trained",
        "val_auc",
        "test_auc",
        "notes",
    ]
    existing_rows: List[Dict[str, object]] = []
    existing_fieldnames: List[str] = []
    if path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="") as existing_handle:
            reader = csv.DictReader(existing_handle)
            existing_rows = list(reader)
            existing_fieldnames = reader.fieldnames or []

    if existing_rows and "dataset" not in existing_fieldnames:
        for row in existing_rows:
            row["dataset"] = "adult"
        with path.open("w", newline="") as upgrade_handle:
            writer = csv.DictWriter(upgrade_handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)

    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_quality_plot(csv_path: Path, output_path: Path) -> None:
    if not csv_path.exists():
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params: List[float] = []
    aucs: List[float] = []
    labels: List[str] = []
    label_map = {
        "baseline_w512": "adult_baseline",
        "pai_w128_cap12": "adult_dendritic",
        "credit_baseline_w512": "credit_baseline",
        "credit_dend_w128_cap8_seed1337": "credit_dendritic",
    }
    with csv_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                params.append(float(row["params"]))
                aucs.append(float(row["val_auc"]))
                key = row["notes"] or row["model_id"]
                labels.append(label_map.get(key, key))
            except (KeyError, ValueError):
                continue
    if not params:
        return
    plt.figure(figsize=(6, 4))
    plt.scatter(params, aucs, c="tab:blue")
    for x, y, label in zip(params, aucs, labels):
        plt.annotate(
            label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8
        )
    plt.xlabel("Parameters")
    plt.ylabel("Validation AUC")
    plt.title("Validation AUC vs Parameter Count")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def benchmark_inference(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 32,
) -> float:
    model.eval()
    total_samples = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (features, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            features = features.to(device)
            _ = model(features)
            total_samples += features.size(0)
    duration = time.perf_counter() - start
    if duration <= 0 or total_samples == 0:
        return float("nan")
    return total_samples / duration


def log_inference_bench(
    path: Path,
    row: Dict[str, object],
    samples_per_second: float,
    device: torch.device,
) -> None:
    fieldnames = [
        "dataset",
        "model_id",
        "use_dendrites",
        "params",
        "samples_per_second",
        "device",
        "notes",
    ]
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "dataset": row.get("dataset"),
                "model_id": row.get("model_id"),
                "use_dendrites": row.get("use_dendrites"),
                "params": row.get("params"),
                "samples_per_second": samples_per_second,
                "device": str(device),
                "notes": row.get("notes"),
            }
        )


def train_model(args: argparse.Namespace) -> None:
    results_dir = CURRENT_DIR / args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = select_device(args.device)
    torch.set_num_threads(4)

    splits = load_dataset(args.dataset, args.seed)
    input_dim = splits["train"][0].shape[1]

    model = AdultMLP(input_dim, hidden_width=args.width, dropout=args.dropout)

    model_id = f"{args.dataset}_w{args.width}_d{args.dropout:.2f}_{'dend' if args.use_dendrites else 'base'}"

    params_history: List[Tuple[int, int]] = []

    if args.use_dendrites:
        # Allow interviewers to reproduce different trade-offs without editing the code.
        if args.max_dendrites is not None:
            GPA.pc.set_max_dendrites(args.max_dendrites)
        if args.fixed_switch_num is not None:
            GPA.pc.set_switch_mode(GPA.pc.DOING_FIXED_SWITCH)
            GPA.pc.set_fixed_switch_num(args.fixed_switch_num)
            first_switch = (
                args.first_fixed_switch_num
                if args.first_fixed_switch_num is not None
                else args.fixed_switch_num
            )
            GPA.pc.set_first_fixed_switch_num(first_switch)
        model = prepare_dendritic_model(
            model,
            exclude_output_proj=args.exclude_output_proj,
            save_name=f"{model_id}_seed{args.seed}",
        )

    model = model.to(device)
    initial_params = count_params(model)
    params_history.append((0, initial_params))

    dataloaders = build_dataloaders(splits, args.batch_size)
    criterion = nn.BCEWithLogitsLoss()

    if args.use_dendrites:
        optimizer, scheduler = configure_pai_optim_scheduler(model, args.lr)
    else:
        optimizer = Adam(model.parameters(), lr=args.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_auc = -math.inf
    best_model_snapshot = copy.deepcopy(model).cpu()
    best_epoch = 0
    best_params = initial_params
    epochs_without_improvement = 0

    train_loader = dataloaders["train"]
    val_loader = dataloaders["val"]

    last_epoch_ran = 0
    for epoch in range(1, args.epochs + 1):
        last_epoch_ran = epoch
        train_metrics = train_one_epoch(
            model, train_loader, device, criterion, optimizer
        )
        val_metrics = evaluate(model, val_loader, device, criterion)

        train_auc_value = train_metrics["auc"]
        if math.isnan(train_auc_value):
            train_auc_value = 0.0
        val_auc_value = val_metrics["auc"]
        if math.isnan(val_auc_value):
            val_auc_value = -math.inf

        if args.use_dendrites:
            GPA.pai_tracker.add_extra_score(train_auc_value, "train")
            (
                model,
                restructured,
                training_complete,
            ) = GPA.pai_tracker.add_validation_score(val_auc_value, model)
            model = model.to(device)
            if restructured:
                optimizer, scheduler = configure_pai_optim_scheduler(model, args.lr)
            if training_complete:
                print("PAI tracker flagged training complete.")
                params_history.append((epoch, count_params(model)))
                break
        else:
            scheduler.step(val_auc_value if math.isfinite(val_auc_value) else 0.0)

        current_params = count_params(model)
        params_history.append((epoch, current_params))

        improved = val_auc_value > best_val_auc
        if improved:
            best_val_auc = val_auc_value
            best_epoch = epoch
            best_model_snapshot = copy.deepcopy(model).cpu()
            best_params = current_params
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d} | "
            f"Train loss {train_metrics['loss']:.4f} | "
            f"Val AUC {val_metrics['auc']:.4f} | "
            f"Params {current_params:,}"
        )

        if epochs_without_improvement >= args.patience:
            print("Early stopping triggered.")
            break

    epochs_trained = last_epoch_ran

    best_model = copy.deepcopy(best_model_snapshot).to(device)
    best_params = count_params(best_model)
    test_metrics = evaluate(best_model, dataloaders["test"], device, criterion)

    if args.use_dendrites:
        params_progress_path = results_dir / "params_progression.csv"
        update_params_progression(params_progress_path, params_history)

    best_scores_path = results_dir / "best_test_scores.csv"
    row = {
        "dataset": args.dataset,
        "model_id": model_id,
        "use_dendrites": args.use_dendrites,
        "exclude_output": args.exclude_output_proj,
        "width": args.width,
        "dropout": args.dropout,
        "params": best_params,
        "epochs_trained": epochs_trained,
        "val_auc": best_val_auc,
        "test_auc": test_metrics["auc"],
        "notes": args.notes
        or f"{args.dataset}_{'dendritic' if args.use_dendrites else 'baseline'}",
    }
    append_best_scores_row(best_scores_path, row)

    bench_path = results_dir / "inference_bench.csv"
    samples_per_second = benchmark_inference(best_model, dataloaders["test"], device)
    log_inference_bench(bench_path, row, samples_per_second, device)

    plot_path = results_dir / "quality_vs_params.png"
    update_quality_plot(best_scores_path, plot_path)

    print(
        f"Best epoch {best_epoch} | "
        f"Best val AUC {best_val_auc:.4f} | "
        f"Test AUC {test_metrics['auc']:.4f}"
    )
    print(f"Results logged to {best_scores_path}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train_model(args)


if __name__ == "__main__":
    main()
