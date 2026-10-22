"""
Causal dilated 1D ResNet for BrainStorm 2026 Track 1: predict which of 8 tones
(or silence) is playing, at every millisecond, from a 1024-channel ECoG array.

This is the original model with normal training: early stopping on validation
balanced accuracy. resnet1d_perforatedai.py trains the same model with
PerforatedAI dendrites, importing the data, model and training step from here.

The data are the BrainStorm 2026 Track 1 parquet files on Hugging Face
(PrecisionNeuroscience/BrainStorm2026-Track1), downloaded into --data-dir on
first use. The public test split contains no real neural data, so the
validation split is the evaluation set.
"""

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score

DATASET = "PrecisionNeuroscience/BrainStorm2026-Track1"

# One residual block per dilation; together they set how far back the model sees
DILATIONS = {
    "short": [1, 2, 4, 8, 16, 32],  # 267 ms of history
    "long": [1, 2, 4, 8, 16, 32, 64, 128, 150],  # 1635 ms of history
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(data_dir, split):
    """(time, 1024) float32 features and (time,) tone labels for one split."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    for kind in ("features", "labels"):
        path = data_dir / f"{split}_{kind}.parquet"
        if not path.exists():
            from huggingface_hub import hf_hub_download

            hf_hub_download(
                repo_id=DATASET,
                filename=path.name,
                repo_type="dataset",
                local_dir=data_dir,
            )
        tables.append(pd.read_parquet(path))
    features, labels = tables
    channels = [c for c in features.columns if str(c).isdigit()]
    return features[channels].to_numpy(np.float32), labels["label"].to_numpy()


def fit_projection(x, n_components=32):
    """PCA fitted on the training data, each component scaled to unit variance."""
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    std = pca.fit_transform(x).std(axis=0) + 1e-6
    weight = (pca.components_ / std[:, None]).astype(np.float32)
    bias = (-(pca.mean_ @ pca.components_.T) / std).astype(np.float32)
    return weight, bias


class CausalConv1d(nn.Conv1d):
    """Conv1d whose output at t only sees inputs at t, t - d, ..., t - (k - 1) d."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True):
        super().__init__(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.left_pad, 0)))


class ResidualBlock(nn.Module):
    """Two dilated causal convolutions with batch norm, dropout and an identity skip."""

    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.dropout(F.relu(self.bn1(self.conv1(x))))
        h = self.bn2(self.conv2(h))
        return F.relu(h + x)


class ResNet1d(nn.Module):
    """Map (batch, 32, time) PCA signals to logits (batch, n_classes, time)."""

    def __init__(
        self,
        n_classes,
        in_channels=32,
        width=32,
        dilations=DILATIONS["long"],
        stem_kernel=15,
        kernel_size=3,
        dropout=0.2,
    ):
        super().__init__()
        self.stem = CausalConv1d(in_channels, width, stem_kernel, bias=False)
        self.stem_bn = nn.BatchNorm1d(width)
        self.blocks = nn.ModuleList(
            ResidualBlock(width, kernel_size, d, dropout) for d in dilations
        )
        self.head = nn.Conv1d(width, n_classes, kernel_size=1)
        # Input samples (including the current one) that each output depends on
        self.receptive_field = stem_kernel + 2 * (kernel_size - 1) * sum(dilations)

    def forward(self, x):
        h = F.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        return self.head(h)


class BrainStormData:
    """PCA-projected train and validation recordings, plus a random chunk sampler."""

    def __init__(self, data_dir, device):
        x_train, y_train = load_split(data_dir, "train")
        x_val, y_val = load_split(data_dir, "validation")
        self.classes, self.y_train = np.unique(y_train, return_inverse=True)
        self.y_val = np.searchsorted(self.classes, y_val)
        weight, bias = fit_projection(x_train)
        # Train as (time, 32) for chunking, validation as one (1, 32, time) sequence
        self.x_train = torch.from_numpy(x_train @ weight.T + bias).to(device)
        x_val = torch.from_numpy(x_val @ weight.T + bias)
        self.x_val = x_val.T.unsqueeze(0).to(device)
        self.y_train_t = torch.from_numpy(self.y_train.astype(np.int64)).to(device)
        # Inverse class frequency, mean 1, so silence doesn't dominate the loss
        counts = np.bincount(self.y_train, minlength=len(self.classes))
        weights = 1.0 / np.maximum(counts, 1)
        weights = weights / weights.sum() * len(self.classes)
        self.class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

    @property
    def x_train_seq(self):
        """The whole training recording as one (1, 32, time) sequence, for scoring."""
        return self.x_train.T.unsqueeze(0)

    def batch(self, receptive_field, batch_size, supervised_len):
        """Random chunks whose last supervised_len timesteps have a full history."""
        chunk_len = receptive_field - 1 + supervised_len
        device = self.x_train.device
        n_starts = len(self.x_train) - chunk_len + 1
        starts = torch.randint(0, n_starts, (batch_size, 1), device=device)
        offsets = torch.arange(chunk_len, device=device)
        xb = self.x_train[starts + offsets]
        yb = self.y_train_t[starts + offsets[receptive_field - 1 :]]
        return xb.transpose(1, 2), yb


def train_epoch(net, data, optimizer, criterion, receptive_field, args):
    """One epoch of random chunks; every timestep with a full history is supervised."""
    net.train()
    total = 0.0
    for _ in range(args.steps_per_epoch):
        xb, yb = data.batch(receptive_field, args.batch_size, args.supervised_len)
        loss = criterion(net(xb)[:, :, receptive_field - 1 :], yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        total += loss.item()
    return total / args.steps_per_epoch


@torch.no_grad()
def balanced_accuracy(net, x, y):
    """Balanced accuracy (%) at every timestep of a (1, 32, time) recording."""
    net.eval()
    pred = net(x)[0].argmax(dim=0).cpu().numpy()
    return 100.0 * balanced_accuracy_score(y, pred)


def common_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        default="data",
        help="where the BrainStorm parquet files are, or will be downloaded",
    )
    parser.add_argument(
        "--memory",
        choices=sorted(DILATIONS),
        default="long",
        help="history the model sees: short = 267 ms, long = 1.6 s",
    )
    parser.add_argument("--width", type=int, default=32, help="channels per layer")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps-per-epoch", type=int, default=25)
    parser.add_argument(
        "--supervised-len",
        type=int,
        default=512,
        help="supervised timesteps per training chunk",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=6,
        help="epochs without improvement before halving the learning rate",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="optional JSON file for the results and per-epoch history",
    )
    return parser


def main():
    parser = common_args("Train the original 1D ResNet on BrainStorm.")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="epochs without improvement before stopping",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = BrainStormData(args.data_dir, device)
    net = ResNet1d(
        len(data.classes), width=args.width, dilations=DILATIONS[args.memory]
    )
    net.to(device)
    rf = net.receptive_field
    n_params = sum(p.numel() for p in net.parameters())
    print(
        f"{args.memory} model ({rf} ms of history), width {args.width}: "
        f"{n_params:,} parameters, on {device}"
    )

    criterion = nn.CrossEntropyLoss(weight=data.class_weights)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.lr_patience
    )

    best, best_epoch, best_state, history = -1.0, 0, None, []
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(net, data, optimizer, criterion, rf, args)
        lr = optimizer.param_groups[0]["lr"]
        train_acc = balanced_accuracy(net, data.x_train_seq, data.y_train)
        val_acc = balanced_accuracy(net, data.x_val, data.y_val)
        scheduler.step(val_acc)
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "loss": loss,
                "train_accuracy": train_acc,
                "val_accuracy": val_acc,
            }
        )
        if val_acc > best:
            best, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(net.state_dict())
        print(
            f"epoch {epoch:3d}  lr {lr:.1e}  loss {loss:.4f}  train {train_acc:.2f}%  "
            f"val {val_acc:.2f}%  (best {best:.2f}% at epoch {best_epoch})"
        )
        if epoch - best_epoch >= args.patience:
            break

    net.load_state_dict(best_state)
    print(f"Best validation {best:.2f}% at epoch {best_epoch}, {n_params:,} parameters")
    if args.out:
        result = {
            "val_accuracy": best,
            "best_epoch": best_epoch,
            "epochs": len(history),
            "params": n_params,
            "args": vars(args),
            "history": history,
        }
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
