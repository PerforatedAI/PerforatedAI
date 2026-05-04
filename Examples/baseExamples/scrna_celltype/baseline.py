import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from config import SEEDS, N_EPOCHS_BASE, LR, D_MODEL, D_FF
from data import build_loaders
from model import GeneTransformer
from train import train_epoch, evaluate


def run_baseline(train_adata, test_adata, label_col: str, device: str,
                 d_model: int = D_MODEL, d_ff: int = D_FF) -> dict:
    results   = []
    criterion = nn.CrossEntropyLoss()

    for seed in SEEDS:
        print(f'\n{"─"*55}\nBaseline  seed={seed}\n{"─"*55}')
        np.random.seed(seed)
        torch.manual_seed(seed)

        train_loader, val_loader, test_loader, le, n_genes, n_cell_types = \
            build_loaders(train_adata, test_adata, label_col, seed)

        model = GeneTransformer(n_genes, n_cell_types, d_model=d_model, d_ff=d_ff).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=LR)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

        hist = {'loss': [], 'train': [], 'val': []}
        best_val, best_snap = 0.0, None
        t0 = time.time()

        for epoch in range(1, N_EPOCHS_BASE + 1):
            loss, tacc = train_epoch(model, train_loader, opt, criterion, device)
            vacc, _, _ = evaluate(model, val_loader, device)
            sched.step(1 - vacc)
            hist['loss'].append(loss)
            hist['train'].append(tacc)
            hist['val'].append(vacc)
            if vacc > best_val:
                best_val  = vacc
                best_snap = copy.deepcopy(model)
            if epoch % 20 == 0:
                print(f'  E{epoch:03d} | loss {loss:.4f} | train {tacc:.4f} | val {vacc:.4f}')

        test_acc, preds, labels = evaluate(best_snap, test_loader, device)
        n_params = sum(p.numel() for p in best_snap.parameters() if p.requires_grad)
        print(f'  → Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)  [{time.time()-t0:.0f}s]')

        results.append({
            'seed': seed, 'test_acc': test_acc, 'preds': preds,
            'labels': labels, 'history': hist, 'n_params': n_params, 'le': le,
        })

    accs     = [r['test_acc'] for r in results]
    mean_acc = float(np.mean(accs))
    std_acc  = float(np.std(accs))

    print(f'\n{"="*55}')
    print(f'BASELINE  mean : {mean_acc*100:.2f}%  ±  {std_acc*100:.2f}%')
    print(f'          runs : {[f"{a*100:.2f}%" for a in accs]}')
    print(f'          params: {results[0]["n_params"]:,}')
    print(f'{"="*55}')

    r0 = results[0]
    print(f'\nClassification report (seed={r0["seed"]}):')
    print(classification_report(
        r0['labels'], r0['preds'],
        labels=list(range(len(r0['le'].classes_))),
        target_names=r0['le'].classes_,
        digits=4, zero_division=0,
    ))

    return {
        'results':       results,
        'mean_acc':      mean_acc,
        'std_acc':       std_acc,
        'n_params':      results[0]['n_params'],
        'seed42_preds':  results[0]['preds'],
        'seed42_labels': results[0]['labels'],
        'le':            results[0]['le'],
    }
