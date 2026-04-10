import matplotlib.pyplot as plt
import numpy as np

_STUDY_NAMES = {
    'smarter':   'Xin 2016',
    'smartseq2': 'Segerstolpe 2016',
    'celseq2':   'Muraro 2016',
    'inDrop1': 'Baron 2016 (b1)', 'inDrop2': 'Baron 2016 (b2)',
    'inDrop3': 'Baron 2016 (b3)', 'inDrop4': 'Baron 2016 (b4)',
}


def plot_baseline(baseline_results: dict, save_path: str = None, tag: str = 'full'):
    if save_path is None:
        save_path = f'baseline_results_{tag}.png'

    results  = baseline_results['results']
    mean_acc = baseline_results['mean_acc']
    std_acc  = baseline_results['std_acc']
    colors   = ['steelblue', 'darkorange', 'seagreen']

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for i, r in enumerate(results):
        axes[0].plot(r['history']['train'], color=colors[i], alpha=0.35, linewidth=1)
        axes[0].plot(r['history']['val'], color=colors[i], linewidth=1.5,
                     label=f"seed {r['seed']}  ({r['test_acc']*100:.2f}%)")
    axes[0].axhline(mean_acc, color='black', linestyle=':', alpha=0.7,
                    label=f'mean {mean_acc*100:.2f}%')
    axes[0].set_title('Baseline — Train (faint) & Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=8)

    axes[1].plot(results[0]['history']['loss'], color='steelblue', linewidth=1.5)
    axes[1].set_title('Baseline — Training Loss (seed 42)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')

    plt.suptitle(
        f'Baseline GeneTransformer  |  mean test acc {mean_acc*100:.2f}% ± {std_acc*100:.2f}%',
        fontweight='bold',
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')


def plot_pai(pai_results: dict, baseline_results: dict,
             save_path: str = None, tag: str = 'full'):
    if save_path is None:
        save_path = f'pai_results_{tag}.png'

    hist       = pai_results['history']
    prog       = pai_results['progressive_table']
    best_acc   = pai_results['best_test_acc']
    baseline_m = baseline_results['mean_acc']
    baseline_s = baseline_results['std_acc']
    ci_lo      = pai_results['ci_lo']
    ci_hi      = pai_results['ci_hi']
    delta      = pai_results['delta']

    fig, axes = plt.subplots(1, 3, figsize=(17, 4))

    axes[0].plot(hist['train'], color='steelblue', alpha=0.5, linewidth=1, label='Train')
    axes[0].plot(hist['val'], color='darkorange', linewidth=1.5, label='Val')
    for i, ep in enumerate(hist['dendrite_epochs']):
        axes[0].axvline(ep - 1, color='red', linestyle='--', alpha=0.7,
                        label='Dendrite added' if i == 0 else '')
    axes[0].set_title('PAI — Accuracy (dendrite insertions in red)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend(fontsize=8)

    axes[1].axhline(baseline_m * 100, color='gray', linestyle='--', linewidth=1.2,
                    label=f'Baseline mean ({baseline_m*100:.2f}%)')
    if prog:
        xs = [0] + [r['insertion'] for r in prog]
        ys = [baseline_m * 100] + [r['test_acc'] * 100 for r in prog]
        axes[1].plot(xs, ys, 'o-', color='steelblue', linewidth=1.5,
                     label='After each insertion')
    axes[1].axhline(best_acc * 100, color='green', linestyle='--', linewidth=1.2,
                    label=f'PAI best ({best_acc*100:.2f}%)')
    axes[1].set_xlabel('Dendrite insertions')
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Test Accuracy vs Dendrite Insertions')
    axes[1].legend(fontsize=8)

    cats   = ['Baseline\n(3-seed mean)', 'PAI\n(best snap)']
    vals   = [baseline_m * 100, best_acc * 100]
    colors = ['#5B9BD5', '#ED7D31']
    bars   = axes[2].bar(cats, vals, yerr=[baseline_s * 100, 0], capsize=5,
                         color=colors, edgecolor='white')
    for bar, v in zip(bars, vals):
        axes[2].text(bar.get_x() + bar.get_width() / 2, v + 0.15,
                     f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].set_title(
        f'PAI vs Baseline\nΔ={delta*100:+.2f}%  CI [{ci_lo*100:.2f}%, {ci_hi*100:.2f}%]'
    )
    axes[2].set_ylim([max(0, min(vals) - 5), min(100, max(vals) + 3)])

    plt.suptitle(
        'GeneTransformer + PAI Dendrites — Human Pancreas scib benchmark  |  10% training data',
        fontweight='bold', fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')


def plot_compression_comparison(full_baseline: dict, small_baseline: dict,
                                small_pai: dict, test_tech: str = 'smarter',
                                save_path: str = None):
    if save_path is None:
        save_path = f'compression_comparison_{test_tech}.png'

    full_acc  = full_baseline['mean_acc']
    full_p    = full_baseline['n_params']
    small_acc = small_baseline['mean_acc']
    small_p   = small_baseline['n_params']
    pai_acc   = small_pai['best_test_acc']
    pai_p     = small_pai['best_params']

    labels = [
        f'Full vanilla\n(256-dim)\n{full_p:,} params',
        f'Small vanilla\n(128-dim)\n{small_p:,} params',
        f'Small + PAI\n(128-dim + dendrites)\n{pai_p:,} params',
    ]
    accs   = [full_acc * 100, small_acc * 100, pai_acc * 100]
    colors = ['#5B9BD5', '#A9C4E4', '#ED7D31']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(labels, accs, color=colors, edgecolor='white', width=0.5)
    for bar, v in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.05,
                     f'{v:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    axes[0].axhline(full_acc * 100, color='#5B9BD5', linestyle='--',
                    linewidth=1.2, alpha=0.6, label='Full vanilla reference')
    axes[0].set_ylabel('Test Accuracy (%)')
    axes[0].set_title('Test Accuracy')
    axes[0].set_ylim([max(0, min(accs) - 3), min(100, max(accs) + 2)])
    axes[0].legend(fontsize=8)

    params_m = [p / 1e6 for p in [full_p, small_p, pai_p]]
    bars2 = axes[1].bar(labels, params_m, color=colors, edgecolor='white', width=0.5)
    for bar, v, p in zip(bars2, params_m, [full_p, small_p, pai_p]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                     f'{v:.2f}M\n({p/full_p*100:.0f}%)',
                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    axes[1].axhline(full_p / 1e6, color='#5B9BD5', linestyle='--',
                    linewidth=1.2, alpha=0.6, label='Full vanilla reference')
    axes[1].set_ylabel('Parameters (millions)')
    axes[1].set_title('Parameter Count  (% of full vanilla)')
    axes[1].legend(fontsize=8)

    test_label = _STUDY_NAMES.get(test_tech, test_tech)
    plt.suptitle(
        f'Compression Experiment — GeneTransformer + PAI Dendrites\n'
        f'Human Pancreas scib benchmark  |  test: {test_label}',
        fontweight='bold', fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f'Saved: {save_path}')


def print_progressive_table(pai_results: dict, baseline_results: dict):
    prog       = pai_results['progressive_table']
    raw_p      = pai_results.get('raw_params', pai_results['start_params'])
    final_p    = pai_results['final_params']
    baseline_m = baseline_results['mean_acc']
    best_acc   = pai_results['best_test_acc']
    best_val   = max(r['val_acc'] for r in prog) if prog else 0.0

    print('─' * 68)
    print('PROGRESSIVE DENDRITE TABLE')
    print('─' * 68)
    print(f'{"Step":>6} {"Epoch":>7} {"Params":>12} {"Val Acc":>10} {"Test Acc":>10}')
    print(f'{"─"*6} {"─"*7} {"─"*12} {"─"*10} {"─"*10}')
    print(f'{"0 (base)":>8} {"—":>7} {raw_p:>12,} {"—":>10} {baseline_m*100:>9.2f}%')

    for row in prog:
        delta     = (row['test_acc'] - baseline_m) * 100
        best_mark = ' ← best' if row['test_acc'] == best_acc else ''
        print(f'{row["insertion"]:>8} {row["epoch"]:>7} {row["params"]:>12,} '
              f'{row["val_acc"]*100:>9.2f}% {row["test_acc"]*100:>9.2f}%  '
              f'({delta:+.2f}%){best_mark}')

    delta_final = (pai_results['final_acc'] - baseline_m) * 100
    print(f'{"FINAL":>8} {pai_results["n_epochs"]:>7} {final_p:>12,} '
          f'{best_val*100:>9.2f}% {pai_results["final_acc"]*100:>9.2f}%  '
          f'({delta_final:+.2f}%)')
    print('─' * 68)


def print_final_summary(pai_results: dict, baseline_results: dict,
                        full_baseline: dict = None, test_tech: str = 'smarter'):
    """
    If full_baseline is provided, prints the compression 4-model table.
    Otherwise prints the standard baseline vs PAI comparison.
    """
    best_acc   = pai_results['best_test_acc']
    baseline_m = baseline_results['mean_acc']
    baseline_s = baseline_results['std_acc']
    ci_lo      = pai_results['ci_lo']
    ci_hi      = pai_results['ci_hi']
    delta      = pai_results['delta']
    raw_p      = pai_results.get('raw_params', pai_results['start_params'])
    best_p     = pai_results['best_params']
    final_p    = pai_results['final_params']
    baseline_p = baseline_results['n_params']
    prog       = pai_results['progressive_table']

    err_bl  = 1 - baseline_m
    err_pai = 1 - best_acc
    rel_red = (err_bl - err_pai) / err_bl * 100 if err_bl > 0 else 0.0

    sig = ('SIGNIFICANT — CI excludes zero'
           if ci_lo > 0 else 'NOT SIGNIFICANT at 95% level — CI includes zero')

    test_label = _STUDY_NAMES.get(test_tech, test_tech)

    print('\n' + '=' * 65)
    print('FINAL RESULTS SUMMARY')
    print('Dataset : Human Pancreas scib benchmark (Luecken et al. 2022)')
    print(f'Split   : Cross-technology  (test: {test_label})')
    print('Training: 10% stratified subsample (rare cell type stress test)')
    print('=' * 65)

    if full_baseline is not None:
        full_acc = full_baseline['mean_acc']
        full_p   = full_baseline['n_params']

        print(f'\n{"Model":<30} {"Params":>12} {"Test Acc":>10} {"vs Full Vanilla":>16}')
        print('─' * 72)
        print(f'{"Full vanilla  (256-dim)":<30} {full_p:>12,} {full_acc*100:>9.2f}%  {"—":>16}')
        print(f'{"Small vanilla (128-dim)":<30} {baseline_p:>12,} {baseline_m*100:>9.2f}%  '
              f'{(baseline_m - full_acc)*100:>+15.2f}%')
        print(f'{"Small + PAI   (best snap)":<30} {best_p:>12,} {best_acc*100:>9.2f}%  '
              f'{(best_acc - full_acc)*100:>+15.2f}%')
        print('─' * 72)

        param_savings = (1 - best_p / full_p) * 100
        print(f'\nCompression: Small+PAI uses {param_savings:.0f}% fewer params than full vanilla')
        print(f'             and achieves {(best_acc - full_acc)*100:+.2f}pp '
              f'{"higher" if best_acc >= full_acc else "lower"} accuracy')
        print(f'\nSmall+PAI vs small baseline:')
        print(f'  Δ = {delta*100:+.2f}%  |  Bootstrap 95% CI: [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]')
        print(f'  Verdict: {sig}')
        print(f'  Relative error reduction (vs small baseline): {rel_red:.1f}%')
    else:
        print(f'{"Metric":<35} {"Baseline":>10} {"PAI":>10} {"Delta":>8}')
        print('─' * 65)
        print(f'{"Test Accuracy":<35} {baseline_m*100:>9.2f}% {best_acc*100:>9.2f}% '
              f'{delta*100:>+7.2f}%')
        print(f'{"Std (3 seeds)":<35} {baseline_s*100:>9.2f}% {"—":>10}')
        print(f'{"Relative error reduction":<35} {"—":>10} {rel_red:>9.1f}%')
        print(f'{"Parameters (raw model)":<35} {baseline_p:>10,} {raw_p:>10,}')
        print(f'{"Parameters (PAI best snap)":<35} {"—":>10} {best_p:>10,} '
              f'{best_p - baseline_p:>+8,}')
        print(f'{"Dendrite insertions":<35} {"—":>10} '
              f'{len(pai_results["history"]["dendrite_epochs"]):>10}')
        print(f'{"Bootstrap 95% CI":<35} {"":>10} [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%]')
        print('─' * 65)
        print(f'Verdict: {sig}')

        breakthrough = next((r for r in prog if r['test_acc'] > baseline_m), None)
        if breakthrough:
            overhead = (breakthrough['params'] - baseline_p) / baseline_p * 100
            print(f'\nBaseline exceeded at insertion #{breakthrough["insertion"]} '
                  f'(+{overhead:.0f}% params, {breakthrough["params"]:,} total)')
        print(f'Dendrites inserted at epochs: {pai_results["history"]["dendrite_epochs"]}')

    print('=' * 65)
