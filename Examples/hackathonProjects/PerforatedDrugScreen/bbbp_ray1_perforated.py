#!/usr/bin/env python
import os
import time
import argparse
import random
import shutil
import tempfile
from typing import Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool
from sklearn.metrics import roc_auc_score

import ray
from ray import train
from ray.train import Checkpoint

# -------------------------
# Utils
# -------------------------
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
    mask: Optional[torch.Tensor] = None
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

    return float(np.mean(aucs)) if aucs else float("nan")

def stratified_split_indices(dataset, train_ratio=0.8, val_ratio=0.1, seed=0, max_tries=50):
    rng = np.random.default_rng(seed)
    ys, valid_idx = [], []
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
        rng.shuffle(idx0)
        rng.shuffle(idx1)

        def split_class(idxs):
            n = len(idxs)
            n_train = int(train_ratio * n)
            n_val = int(val_ratio * n)
            train_idx = idxs[:n_train]
            val_idx = idxs[n_train:n_train + n_val]
            test_idx = idxs[n_train + n_val:]
            return train_idx, val_idx, test_idx

        tr0, va0, te0 = split_class(idx0)
        tr1, va1, te1 = split_class(idx1)

        train_idx = np.concatenate([tr0, tr1])
        val_idx = np.concatenate([va0, va1])
        test_idx = np.concatenate([te0, te1])

        rng.shuffle(train_idx)
        rng.shuffle(val_idx)
        rng.shuffle(test_idx)

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

# -------------------------
# Model
# -------------------------
class GIN_MoleculeNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        nn1 = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(nn1))
        self.norms.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            nnk = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
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
        return self.head(g)

@torch.no_grad()
def eval_epoch(model, loader, device) -> Tuple[float, float]:
    model.eval()
    ys, ps, masks = [], [], []
    total_loss, total_count = 0.0, 0

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

        loss = (
            F.binary_cross_entropy_with_logits(logits[mask], y_f[mask], reduction="mean")
            if mask.any()
            else torch.tensor(0.0, device=device)
        )

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
    return total_loss / max(total_count, 1), auc

def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss, total_count = 0.0, 0

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

# -------------------------
# Ray train loop
# -------------------------
def ray_train_loop(config: dict):
    # Avoid PerforatedAI pdb prompts inside Ray workers
    os.environ["PYTHONBREAKPOINT"] = "0"

    # Make W&B more stable inside Ray workers
    # (avoids service connection issues on shutdown in many Ray setups)
    os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")
    os.environ.setdefault("WANDB_SILENT", "true")

    ctx = train.get_context()
    rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()
    is_rank0 = (rank == 0)

    mode = config["mode"]  # baseline | dendrites
    doing_pai = bool(config.get("doing_pai", False)) and (mode == "dendrites")

    # Keep dendrites in single-worker unless you intentionally implement multi-worker-safe PAI output
    if doing_pai and world_size != 1:
        raise RuntimeError("Run dendrites mode with --ray_workers 1 (PAI files + GPA are global).")

    seed = int(config["seed"])
    set_seed(seed)

    device = torch.device("cuda" if (config["device"] == "cuda" and torch.cuda.is_available()) else "cpu")

    # Ensure PAI outputs go exactly where YOU want (not Ray worker temp dirs)
    pai_dir = os.path.abspath(config.get("pai_dir", os.path.join(os.getcwd(), "PAI")))
    os.makedirs(pai_dir, exist_ok=True)

    # Optional: if you want outputs per run, you can also create subfolders, but this is simplest.
    # We'll write PAI2.* into pai_dir/PAI2*
    pai_prefix = os.path.join(pai_dir, "PAI2")

    dataset = MoleculeNet(root=config["root"], name=config["task"])
    train_idx, val_idx, test_idx = stratified_split_indices(dataset, seed=seed)

    train_ds = dataset[train_idx]
    val_ds = dataset[val_idx]
    test_ds = dataset[test_idx]

    data_frac = float(config.get("data_frac", 1.0))
    if data_frac < 1.0:
        k = max(1, int(len(train_ds) * data_frac))
        train_ds = train_ds[:k]

    train_loader = DataLoader(train_ds, batch_size=int(config["batch_size"]), shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=int(config["batch_size"]), shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=int(config["batch_size"]), shuffle=False)

    in_dim = dataset.num_node_features
    out_dim = int(dataset[0].y.numel()) if dataset[0].y is not None else 1

    model = GIN_MoleculeNet(
        in_dim=in_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        out_dim=out_dim,
    )

    # W&B: only init on rank0
    wandb_run = None
    if is_rank0 and bool(config.get("wandb", False)):
        import wandb
        wandb_run = wandb.init(
            project=config.get("wandb_project", "PerforatedDrugScreen"),
            entity=config.get("wandb_entity", None),
            name=config.get("wandb_run_name", f"BBBP_RAY_{mode}_seed{seed}"),
            config={k: v for k, v in config.items()},
        )

    tracker = None
    if doing_pai:
        from perforatedai.tracker_perforatedai import PAINeuronModuleTracker
        from perforatedai.utils_perforatedai import GPA

        tracker = PAINeuronModuleTracker(
            doing_pai=True,
            save_name=pai_prefix,        # <-- writes pai_dir/PAI2.png + csvs
            making_graphs=True,
        )
        GPA.pai_tracker = tracker
        GPA.pc.set_unwrapped_modules_confirmed(True)
        GPA.pc.set_testing_dendrite_capacity(False)
        GPA.pc.set_weight_decay_accepted(True)

        model = tracker.initialize(model)

    model.to(device)

    # Optimizer/scheduler (tracker-aware)
    lr = float(config["lr"])
    wd = float(config["weight_decay"])
    sched_args = {"mode": "max", "patience": 8, "factor": 0.5}

    if tracker is not None:
        from perforatedai.utils_perforatedai import GPA

        tracker.set_optimizer(torch.optim.Adam)
        tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)

        optim_args = {"params": model.parameters(), "lr": lr, "weight_decay": wd}
        optim_args.pop("model", None)  # safety
        optimizer, scheduler = tracker.setup_optimizer(model, optim_args, sched_args)

        # SAFETY FIX for the GPA optimizer_instance None crash
        # Some codepaths inside tracker expect this to be set.
        GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
        GPA.pai_tracker.member_vars["scheduler_instance"] = scheduler
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **sched_args)

    best_val_auc = -1.0
    best_test_at_best_val = float("nan")
    best_epoch = -1
    best_params = -1

    epochs = int(config["epochs"])

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer, device)
        _, val_auc = eval_epoch(model, val_loader, device)
        _, test_auc = eval_epoch(model, test_loader, device)

        scheduler.step(val_auc if not np.isnan(val_auc) else 0.0)

        restructured = False
        training_complete = False
        if tracker is not None:
            from perforatedai.utils_perforatedai import GPA

            model, restructured, training_complete = tracker.add_validation_score(val_auc, model)

            if restructured:
                optim_args = {"params": model.parameters(), "lr": lr, "weight_decay": wd}
                optimizer, scheduler = tracker.setup_optimizer(model, optim_args, sched_args)
                GPA.pai_tracker.member_vars["optimizer_instance"] = optimizer
                GPA.pai_tracker.member_vars["scheduler_instance"] = scheduler

        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_test_at_best_val = test_auc
            best_epoch = epoch
            best_params = count_params(model)

        lr_now = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        metrics = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/auc": val_auc,
            "test/auc": test_auc,
            "opt/lr": lr_now,
            "model/params": count_params(model),
            "pai/restructured": int(bool(restructured)),
            "pai/training_complete": int(bool(training_complete)),
            "time/epoch_sec": dt,
            "data_frac": data_frac,
        }

        # log to W&B on rank0 only
        if wandb_run is not None:
            try:
                wandb_run.log(metrics, step=epoch)
            except Exception:
                pass

        # report to Ray each epoch (Ray aggregates)
        train.report(metrics)

        if training_complete:
            break

    # -------------------------
    # Final: ensure Ray returns non-None metrics + export PAI2.png
    # -------------------------
    final_metrics = {
        "best_val_auc": best_val_auc,
        "best_test_auc_at_best_val": best_test_at_best_val,
        "best_epoch": best_epoch,
        "best_params": best_params,
        "mode": mode,
    }

    # Create a small checkpoint dir with key artifacts (PAI2.png if exists)
    ckpt_dir = tempfile.mkdtemp(prefix="ray_ckpt_")
    try:
        # Save a tiny summary text too
        with open(os.path.join(ckpt_dir, "summary.txt"), "w") as f:
            f.write(str(final_metrics) + "\n")
            f.write(f"pai_dir={pai_dir}\n")
            f.write(f"pai_prefix={pai_prefix}\n")

        # Copy PAI2.png if it exists; otherwise copy PAI.png if present
        pai2_png = pai_prefix + ".png"
        fallback_png = os.path.join(pai_dir, "PAI.png")

        if os.path.exists(pai2_png):
            shutil.copy2(pai2_png, os.path.join(ckpt_dir, "PAI2.png"))
        elif os.path.exists(fallback_png):
            shutil.copy2(fallback_png, os.path.join(ckpt_dir, "PAI.png"))

        # Send a final report WITH checkpoint so driver result.metrics is not None
        ckpt = Checkpoint.from_directory(ckpt_dir)
        train.report(final_metrics, checkpoint=ckpt)

    finally:
        # W&B finish safely
        if wandb_run is not None:
            try:
                wandb_run.finish()
            except Exception:
                pass

# -------------------------
# Driver
# -------------------------
def main():
    ap = argparse.ArgumentParser()

    # mode
    ap.add_argument("--mode", type=str, default="baseline", choices=["baseline", "dendrites"])
    ap.add_argument("--doing_pai", action="store_true")

    # data/model
    ap.add_argument("--task", type=str, default="BBBP")
    ap.add_argument("--root", type=str, default="./data_molnet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=64)
    ap.add_argument("--num_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--data_frac", type=float, default=1.0)

    # PAI output control
    ap.add_argument("--pai_dir", type=str, default=os.path.join(os.getcwd(), "PAI"))

    # Ray
    ap.add_argument("--ray_workers", type=int, default=1, help="Use 1 for dendrites; baseline can be >1.")

    # W&B
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="PerforatedDrugScreen")
    ap.add_argument("--wandb_entity", type=str, default=None)
    ap.add_argument("--wandb_run_name", type=str, default=None)

    args = ap.parse_args()

    ray.init(ignore_reinit_error=True)

    config = vars(args)
    if config["wandb_run_name"] is None:
        config["wandb_run_name"] = f"BBBP_RAY_{args.mode}_seed{args.seed}"

    from ray.train.torch import TorchTrainer
    from ray.train import ScalingConfig, RunConfig

    trainer = TorchTrainer(
        train_loop_per_worker=ray_train_loop,
        train_loop_config=config,
        scaling_config=ScalingConfig(num_workers=args.ray_workers, use_gpu=(args.device == "cuda")),
        run_config=RunConfig(name=f"BBBP_GIN_{args.mode}_hd{args.hidden_dim}_L{args.num_layers}_seed{args.seed}"),
    )

    result = trainer.fit()

    print("\n=== RAY RESULT (driver) ===")
    print("error:", result.error)
    print("metrics:", result.metrics)

    # If you want to retrieve checkpoint dir content:
    if result.checkpoint is not None:
        out_dir = os.path.abspath("./ray_result_artifacts")
        os.makedirs(out_dir, exist_ok=True)
        result.checkpoint.to_directory(out_dir)
        print(f"Saved Ray checkpoint artifacts to: {out_dir}")
        print("Artifacts:", os.listdir(out_dir))

if __name__ == "__main__":
    main()
