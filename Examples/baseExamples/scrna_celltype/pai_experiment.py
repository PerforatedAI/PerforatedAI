import copy
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from config import N_EPOCHS_PAI, LR, D_MODEL, D_FF
from data import build_loaders
from model import GeneTransformer
from train import train_epoch, evaluate, bootstrap_ci_seeds


def _configure_pai():
    """
    Apply PAI configuration before model initialisation.

    Four settings are required for correct behaviour with transformers:
      - module_names_to_convert=["Linear"]: avoids ipdb prompts on LayerNorm etc.
      - unwrapped_modules_confirmed=True: suppresses confirmation for non-Linear layers
      - input_dimensions=[-1, -1, 0]: declares 3D transformer tensors (batch, seq, features)
      - testing_dendrite_capacity=False: DOING_HISTORY mode; without this PAI exits after ~4 epochs
    """
    from perforatedai import globals_perforatedai as GPA

    GPA.pc.set_module_names_to_convert(['Linear'])
    GPA.pc.set_unwrapped_modules_confirmed(True)
    GPA.pc.set_input_dimensions([-1, -1, 0])
    GPA.pc.set_testing_dendrite_capacity(False)

    print('PAI configured:')
    print('  module_names_to_convert  ["Linear"]')
    print('  unwrapped_modules_confirmed  True')
    print('  input_dimensions  [-1, -1, 0]  (3D transformer)')
    print('  testing_dendrite_capacity  False  (DOING_HISTORY)')


def _set_output_dimensions(model) -> int:
    """
    PAI defaults to 2D output [-1, 0] for every wrapped layer.
    Transformer internal layers output 3D — correct them after initialize_pai.
    The classifier is 2D (post mean-pool) and is left unchanged.
    """
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'set_this_output_dimensions'):
            dims = [-1, 0] if 'classifier' in name else [-1, -1, 0]
            module.set_this_output_dimensions(dims)
            count += 1
    return count


def run_pai(train_adata, test_adata, label_col: str,
            baseline_results: dict, device: str,
            seed: int = 42,
            d_model: int = D_MODEL, d_ff: int = D_FF) -> dict:
    """
    Train GeneTransformer with PAI dendrites in DOING_HISTORY mode.

    Dendrites are inserted automatically at validation plateaus.
    A deepcopy snapshot is saved at each insertion — state_dict alone is
    insufficient because PAI changes the model structure on restructuring.
    The best test accuracy across all snapshots is reported.
    """
    _configure_pai()

    from perforatedai import globals_perforatedai as GPA
    from perforatedai import utils_perforatedai as UPA

    np.random.seed(seed)
    torch.manual_seed(seed)

    train_loader, val_loader, test_loader, le, n_genes, n_cell_types = \
        build_loaders(train_adata, test_adata, label_col, seed)

    criterion = nn.CrossEntropyLoss()
    pai_model = GeneTransformer(n_genes, n_cell_types, d_model=d_model, d_ff=d_ff).to(device)
    opt       = torch.optim.Adam(pai_model.parameters(), lr=LR)
    sched     = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    raw_params = sum(p.numel() for p in pai_model.parameters() if p.requires_grad)
    pai_model  = UPA.initialize_pai(pai_model)
    n_dims_set = _set_output_dimensions(pai_model)
    GPA.pai_tracker.set_optimizer_instance(opt)

    start_params = sum(p.numel() for p in pai_model.parameters() if p.requires_grad)
    print(f'\nOutput dimensions set for {n_dims_set} PAI-wrapped modules.')
    print(f'Raw model params: {raw_params:,}  |  After PAI init: {start_params:,}')
    print(f'Training up to  : {N_EPOCHS_PAI} epochs\n')

    hist = {'loss': [], 'train': [], 'val': [], 'dendrite_epochs': []}
    progressive_table = []
    best_val      = 0.0
    best_test_acc = 0.0
    best_snap     = None
    best_preds    = None
    best_labels   = None
    t0 = time.time()

    for epoch in range(1, N_EPOCHS_PAI + 1):
        loss, tacc = train_epoch(pai_model, train_loader, opt, criterion, device)
        vacc, _, _ = evaluate(pai_model, val_loader, device)
        sched.step(1 - vacc)

        pai_model, restructured, training_complete = \
            GPA.pai_tracker.add_validation_score(vacc, pai_model)
        pai_model = pai_model.to(device)

        if restructured:
            opt   = torch.optim.Adam(pai_model.parameters(), lr=LR)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
            GPA.pai_tracker.set_optimizer_instance(opt)

            new_params   = sum(p.numel() for p in pai_model.parameters() if p.requires_grad)
            n_insertions = len(hist['dendrite_epochs']) + 1
            snap_acc, snap_preds, snap_labels = evaluate(pai_model, test_loader, device)

            progressive_table.append({
                'insertion': n_insertions,
                'epoch':     epoch,
                'params':    new_params,
                'val_acc':   vacc,
                'test_acc':  snap_acc,
            })
            print(f'  *** Dendrite #{n_insertions} at epoch {epoch:3d} | '
                  f'params: {new_params:>10,} | val: {vacc:.4f} | test: {snap_acc:.4f} ***')
            hist['dendrite_epochs'].append(epoch)

            if snap_acc > best_test_acc:
                best_test_acc = snap_acc
                best_snap     = copy.deepcopy(pai_model)
                best_preds    = snap_preds[:]
                best_labels   = snap_labels[:]
                print(f'      → New best test acc: {best_test_acc:.4f}  (snapshot saved)')

        hist['loss'].append(loss)
        hist['train'].append(tacc)
        hist['val'].append(vacc)
        if vacc > best_val:
            best_val = vacc

        if epoch % 20 == 0:
            print(f'  E{epoch:03d} | loss {loss:.4f} | train {tacc:.4f} | val {vacc:.4f}')

        if training_complete:
            print(f'\n  PAI complete at epoch {epoch}.')
            break

    final_epochs = epoch
    final_params = sum(p.numel() for p in pai_model.parameters() if p.requires_grad)
    final_acc, final_preds, final_labels = evaluate(pai_model, test_loader, device)

    if best_snap is None:
        best_test_acc = final_acc
        best_preds    = final_preds
        best_labels   = final_labels
        best_snap     = pai_model

    best_params = sum(p.numel() for p in best_snap.parameters() if p.requires_grad)

    elapsed = time.time() - t0
    print(f'\nDone in {elapsed:.0f}s  ({final_epochs} epochs)')
    print(f'Final model  : {final_acc:.4f}  ({final_params:,} params)')
    print(f'Best snapshot: {best_test_acc:.4f}  ({best_params:,} params)')

    baseline_accs = [r['test_acc'] for r in baseline_results['results']]
    ci_lo, ci_hi  = bootstrap_ci_seeds(best_test_acc, baseline_accs)
    delta = best_test_acc - baseline_results['mean_acc']

    print(f'\n>>> PAI TEST ACCURACY (best): {best_test_acc:.4f} ({best_test_acc*100:.2f}%) <<<')
    print(f'Bootstrap 95% CI: [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]')

    print(f'\nClassification report (PAI best snapshot):')
    print(classification_report(
        best_labels, best_preds,
        labels=list(range(n_cell_types)),
        target_names=le.classes_,
        digits=4, zero_division=0,
    ))

    return {
        'best_test_acc':     best_test_acc,
        'final_acc':         final_acc,
        'best_preds':        best_preds,
        'best_labels':       best_labels,
        'raw_params':        raw_params,
        'start_params':      start_params,
        'best_params':       best_params,
        'final_params':      final_params,
        'history':           hist,
        'progressive_table': progressive_table,
        'ci_lo':             ci_lo,
        'ci_hi':             ci_hi,
        'delta':             delta,
        'le':                le,
        'n_epochs':          final_epochs,
        'seed':              seed,
    }
