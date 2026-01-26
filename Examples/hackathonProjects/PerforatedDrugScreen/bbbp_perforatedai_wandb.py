#!/usr/bin/env python
# bbbp_perforatedai_wandb.py
import os
import time
import argparse
import random
from typing import Tuple, Optional

import numpy as np

def stratified_split_indices(dataset, train_ratio=0.8, val_ratio=0.1, seed=0, max_tries=50):
    """Stratify by task-0 label to avoid AUC=nan (val/test must include both classes)."""
    import numpy as np
    rng = np.random.default_rng(seed)

    ys = []
    valid_idx = []
    for i in range(len(dataset)):
        y = dataset[i].y
        if y is None:
            continue
        y0 = y.view(-1)[0].item()
        if np.isnan(y0):
            continue
        ys.append(int(y0))
        valid_idx.append(i)

    valid_idx = np.array(valid_idx)
    ys = np.array(ys)
    idx0 = valid_idx[ys == 0]
    idx1 = valid_idx[ys == 1]

    for _ in range(max_tries):
        rng.shuffle(idx0); rng.shuffle(idx1)

        def split_class(idxs):
            n = len(idxs)
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)
            train = idxs[:n_train]
            val = idxs[n_train:n_train + n_val]
            test = idxs[n_train + n_val:]
            return train, val, test

        tr0, va0, te0 = split_class(idx0)
        tr1, va1, te1 = split_class(idx1)

        train_idx = np.concatenate([tr0, tr1])
        val_idx   = np.concatenate([va0, va1])
        test_idx  = np.concatenate([te0, te1])
        rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)

        def has_both(idxs):
            labs = []
            for j in idxs:
                y0 = dataset[j].y.view(-1)[0].item()
                if not np.isnan(y0):
                    labs.append(int(y0))
            return (0 in labs) and (1 in labs)

        if has_both(val_idx) and has_both(test_idx):
            return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

    raise RuntimeError("Failed to create stratified split; try a different seed.")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from sklearn.metrics import roc_auc_score

from perforatedai.tracker_perforatedai import PAINeuronModuleTracker


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.reshape(-1)
    y_score = y_score.reshape(-1)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def compute_auc_multitask(
    y_true: torch.Tensor,
    y_prob: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> float:
    y_true = y_true.detach().cpu()
    y_prob = y_prob.detach().cpu()

    if y_true.dim() == 1:
        y_true = y_true.view(-1, 1)
    if y_prob.dim() == 1:
        y_prob = y_prob.view(-1, 1)

    if mask is None:
        mask = ~torch.isnan(y_true)
    else:
        mask = mask.bool()

    aucs = []
    T = y_true.size(1)
    for t in range(T):
        m = mask[:, t]
        if m.sum().item() < 10:
            continue
        yt = y_true[m, t].numpy()
        yp = y_prob[m, t].numpy()
        auc = safe_roc_auc(yt, yp)
        if not np.isnan(auc):
            aucs.append(auc)

    if len(aucs) == 0:
        return float("nan")
    return float(np.mean(aucs))


class GIN_MoleculeNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        nn1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(nn1))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            nnk = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(nnk))
            self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        g = global_add_pool(x, batch)
        logits = self.head(g)
        return logits


@torch.no_grad()
def eval_epoch(model, loader, device) -> Tuple[float, float]:
    model.eval()
    ys = []
    ps = []
    masks = []
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y
        if y is None:
            continue
        if y.dim() == 1:
            y = y.view(-1, 1)

        mask = ~torch.isnan(y)
        y_f = torch.nan_to_num(y, nan=0.0).float()

        loss = F.binary_cross_entropy_with_logits(
            logits[mask], y_f[mask], reduction="mean"
        ) if mask.any() else torch.tensor(0.0, device=device)

        total_loss += float(loss.item()) * int(batch.num_graphs)
        total_count += int(batch.num_graphs)

        prob = torch.sigmoid(logits)
        ys.append(y)
        ps.append(prob)
        masks.append(mask)

    if total_count == 0:
        return float("nan"), float("nan")

    y_all = torch.cat(ys, dim=0)
    p_all = torch.cat(ps, dim=0)
    m_all = torch.cat(masks, dim=0)
    auc = compute_auc_multitask(y_all, p_all, m_all)
    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, auc


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        y = batch.y
        if y is None:
            continue
        if y.dim() == 1:
            y = y.view(-1, 1)

        mask = ~torch.isnan(y)
        if not mask.any():
            continue

        y_f = torch.nan_to_num(y, nan=0.0).float()

        loss = F.binary_cross_entropy_with_logits(logits[mask], y_f[mask], reduction="mean")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * int(batch.num_graphs)
        total_count += int(batch.num_graphs)

    return total_loss / max(total_count, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="dendrites",
                    choices=["baseline", "dendrites"],
                    help="baseline = no tracker; dendrites = enable PerforatedAI tracker")


    ap.add_argument("--task", type=str, default="BBBP",
                    help="MoleculeNet task name: BBBP, HIV, BACE, ClinTox, Tox21, SIDER, etc.")
    ap.add_argument("--root", type=str, default="./data_molnet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--data_frac", type=float, default=1.0)

    # Dendrites / tracker
    ap.add_argument("--doing_pai", action="store_true", help="Enable PerforatedAI dendritic optimization via tracker mode.")
    ap.add_argument("--save_name", type=str, default="PerforatedDrugScreen", help="Tracker save prefix.")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="PerforatedDrugScreen")
    ap.add_argument("--wandb_run_name", type=str, default="", help="If empty, auto-name using task/seed/config.")
    args = ap.parse_args()

    # Make Perforated show up in leaderboard names
    if args.wandb_run_name.strip() == "":
        args.wandb_run_name = f"Perforated_{args.task}_GIN_hd{args.hidden_dim}_L{args.num_layers}_seed{args.seed}_{'dendrites' if args.doing_pai else 'baseline'}"

    # Avoid interactive debugger halts from PerforatedAI internals
    os.environ["PYTHONBREAKPOINT"] = "0"

    set_seed(args.seed)

    device = torch.device("cpu")
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")

    dataset = MoleculeNet(root=args.root, name=args.task)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val

    train_idx, val_idx, test_idx = stratified_split_indices(dataset, train_ratio=0.8, val_ratio=0.1, seed=args.seed)
    train_ds = dataset[train_idx]
    val_ds   = dataset[val_idx]
    test_ds  = dataset[test_idx]

    if args.data_frac < 1.0:
        k = max(1, int(len(train_ds) * args.data_frac))
        train_ds = train_ds[:k]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    in_dim = dataset.num_node_features
    sample_y = dataset[0].y
    out_dim = int(sample_y.numel()) if sample_y is not None else 1

    model = GIN_MoleculeNet(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        out_dim=out_dim,
    )

    # -----------------------------
    # PerforatedAI tracker (optional)
    # -----------------------------
    tracker = None
    if args.doing_pai:
        tracker = PAINeuronModuleTracker(
            doing_pai=True,
            save_name=f"{args.save_name}_{args.task}",
            making_graphs=True,
        )

        from perforatedai.utils_perforatedai import GPA
        GPA.pai_tracker = tracker
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_weight_decay_accepted(True)

        model = tracker.initialize(model)

    model.to(device)

    # Optimizer / scheduler
    if tracker is not None:
        tracker.set_optimizer(torch.optim.Adam)
        tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

        optim_args = {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
        sched_args = {"mode": "max", "patience": 8, "factor": 0.5}

        # Safety: Adam doesn't accept model= kwarg
        optim_args.pop('model', None)
        optimizer, scheduler = tracker.setup_optimizer(model, optim_args, sched_args)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=8, factor=0.5)

    # W&B
    use_wandb = False
    if args.wandb:
        import wandb
        use_wandb = True
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        wandb.config.update({
            "device": str(device),
            "dataset": f"MoleculeNet/{args.task}",
            "out_dim": out_dim,
        }, allow_val_change=True)

    best_val_auc = -1.0
    best_test_auc_at_best_val = float("nan")
    best_epoch = -1
    best_params = -1

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        _, val_auc = eval_epoch(model, val_loader, device)
        _, test_auc = eval_epoch(model, test_loader, device)

        # Scheduler step
        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

        # Tracker update (only if dendrites enabled)
        restructured = False
        training_complete = False
        if tracker is not None:
            model, restructured, training_complete = tracker.add_validation_score(val_auc, model)
            if restructured:
                # IMPORTANT: re-setup optimizer/scheduler after restructure
                optim_args = {"params": model.parameters(), "lr": args.lr, "weight_decay": args.weight_decay}
                optimizer, scheduler = tracker.setup_optimizer(model, optim_args, sched_args)

        lr_now = optimizer.param_groups[0]["lr"]
        params = count_params(model)
        dt = time.time() - t0

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_auc_at_best_val = test_auc
            best_epoch = epoch
            best_params = params

        line = (
            f"[{args.task}][{epoch:03d}] "
            f"train_loss={train_loss:.4f} val_auc={val_auc:.4f} test_auc={test_auc:.4f} "
            f"lr={lr_now:.2e} restruct={int(bool(restructured))} complete={bool(training_complete)} "
            f"params={params} time={dt:.1f}s data_frac={args.data_frac}"
        )
        print(line)

        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_loss,
                "val/auc": val_auc,
                "test/auc": test_auc,
                "opt/lr": lr_now,
                "pai/restructured": int(bool(restructured)),
                "pai/training_complete": int(bool(training_complete)),
                "model/params": params,
                "time/epoch_sec": dt,
                "data_frac": args.data_frac,
            })

        # Some tracker modes can terminate early (good to log)
        if training_complete:
            print("Tracker ended training early (training_complete=True).")
            break

    print("\n=== Summary ===")
    print(f"Run: {args.wandb_run_name}")
    print(f"Task: {args.task}")
    print(f"Best val AUC: {best_val_auc:.4f} @ epoch {best_epoch} (params={best_params})")
    print(f"Test AUC @ best val: {best_test_auc_at_best_val:.4f}")

    if use_wandb:
        import wandb
        wandb.summary["best/val_auc"] = best_val_auc
        wandb.summary["best/test_auc_at_best_val"] = best_test_auc_at_best_val
        wandb.summary["best/epoch"] = best_epoch
        wandb.summary["best/params"] = best_params
        wandb.finish()


if __name__ == "__main__":
    main()
