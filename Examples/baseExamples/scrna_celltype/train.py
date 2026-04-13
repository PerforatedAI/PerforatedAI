import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score


def train_epoch(model: nn.Module, loader, optimizer, criterion,
                device: str) -> tuple[float, float]:
    model.train()
    total_loss = correct = total = 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out  = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (out.argmax(1) == yb).sum().item()
        total      += len(yb)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: str) -> tuple[float, list, list]:
    model.eval()
    preds, labels = [], []
    for Xb, yb in loader:
        preds.extend(model(Xb.to(device)).argmax(1).cpu().numpy())
        labels.extend(yb.numpy())
    return accuracy_score(labels, preds), preds, labels


def bootstrap_ci_seeds(pai_acc: float, baseline_accs: list,
                       n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """
    Bootstrap 95% CI for PAI improvement over baseline.

    Bootstraps over per-seed baseline accuracies rather than per-prediction,
    which avoids artificially narrow intervals when PAI and baseline produce
    correlated outputs on the same test set.

    CI excluding zero indicates statistical significance.
    """
    rng  = np.random.default_rng(seed)
    accs = np.array(baseline_accs)
    n    = len(accs)

    deltas = [
        pai_acc - rng.choice(accs, size=n, replace=True).mean()
        for _ in range(n_boot)
    ]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
