"""
Permuted MNIST Benchmark - 持续学习标准测试
这是持续学习领域最重要的benchmark之一

实验设计:
- 10个任务，每个任务是MNIST的一个随机排列
- 任务序列: Task 1 → Task 2 → ... → Task 10
- 测量: 每个任务学完后，对所有之前任务的准确率
- 关键指标: Average Accuracy, Forgetting Measure

参考论文:
- Goodfellow et al. "An Empirical Investigation of Catastrophic Forgetting" (2013)
- Kirkpatrick et al. "Overcoming catastrophic forgetting" (EWC, PNAS 2017)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os
import time
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../perforatedai'))

from perforatedai.utils_perforatedai import initialize_pai
import perforatedai.globals_perforatedai as GPA


class PermutedMNIST:
    """Permuted MNIST数据集生成器"""

    def __init__(self, num_tasks=10, seed=42):
        """
        Args:
            num_tasks: 任务数量
            seed: 随机种子（保证可重现）
        """
        self.num_tasks = num_tasks
        self.seed = seed

        # 生成排列索引
        self.permutations = []
        np.random.seed(seed)
        for i in range(num_tasks):
            if i == 0:
                # 第一个任务不排列（原始MNIST）
                self.permutations.append(np.arange(784))
            else:
                # 其他任务随机排列
                self.permutations.append(np.random.permutation(784))

    def get_task_data(self, task_id, train=True, batch_size=64):
        """获取指定任务的数据加载器

        Args:
            task_id: 任务ID (0到num_tasks-1)
            train: 训练集还是测试集
            batch_size: batch大小

        Returns:
            DataLoader
        """
        # 加载MNIST
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            './data',
            train=train,
            download=True,
            transform=transform
        )

        # 创建排列的dataset
        permuted_dataset = PermutedDataset(dataset, self.permutations[task_id])

        loader = torch.utils.data.DataLoader(
            permuted_dataset,
            batch_size=batch_size,
            shuffle=train
        )

        return loader


class PermutedDataset(torch.utils.data.Dataset):
    """应用排列的Dataset包装器"""

    def __init__(self, base_dataset, permutation):
        """
        Args:
            base_dataset: 原始MNIST dataset
            permutation: 排列索引 (784,)
        """
        self.base_dataset = base_dataset
        self.permutation = torch.LongTensor(permutation)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        # 展平并应用排列
        img_flat = img.view(-1)  # [784]
        img_permuted = img_flat[self.permutation]

        # 恢复形状
        img_permuted = img_permuted.view(1, 28, 28)

        return img_permuted, label


class SimpleNet(nn.Module):
    """简单的全连接网络"""
    def __init__(self, hidden_size=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train_on_task(model, train_loader, optimizer, criterion, epochs=1, verbose=False):
    """在一个任务上训练"""
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        if verbose:
            acc = 100. * correct / total
            print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.2f}%")


def evaluate_on_task(model, test_loader):
    """评估任务性能"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100. * correct / total
    return accuracy


def permuted_mnist_experiment(method='baseline', num_tasks=10, epochs_per_task=1):
    """Permuted MNIST实验

    Args:
        method: 'baseline', 'selective', 'selective_replay'
        num_tasks: 任务数量
        epochs_per_task: 每个任务的训练epochs

    Returns:
        results: 包含所有任务准确率的字典
    """
    print(f"\n{'=' * 80}")
    print(f"Permuted MNIST Experiment: {method.upper()}")
    print(f"{'=' * 80}")
    print(f"Tasks: {num_tasks}")
    print(f"Epochs per task: {epochs_per_task}\n")

    # 创建数据生成器
    pm_dataset = PermutedMNIST(num_tasks=num_tasks)

    # 创建模型
    model = SimpleNet(hidden_size=256)
    model = initialize_pai(model)

    # 配置方法
    if method == 'baseline':
        GPA.pc.set_selective_plasticity_enabled(False)
        print("[CONFIG] Baseline\n")

    elif method == 'selective':
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 5.0
        config['surprise_weight'] = 0.5
        config['warmup_steps'] = 1000
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] SelectivePlasticity (moderate)\n")

    elif method == 'selective_strict':
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 10.0
        config['surprise_weight'] = 0.8
        config['warmup_steps'] = 500
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] SelectivePlasticity (strict)\n")

    elif method == 'adaptive':
        # Phase 2 Lite: Simple threshold adjustment
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 3.0
        config['surprise_weight'] = 0.3
        config['warmup_steps'] = 1000
        config['plasticity_threshold'] = 0.05
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] Phase 2 Lite (lenient threshold)\n")

    elif method == 'adaptive_full':
        # Phase 2 Full: With task-aware adaptive threshold
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 4.0
        config['surprise_weight'] = 0.4
        config['warmup_steps'] = 1000
        config['plasticity_threshold'] = 0.08
        config['use_adaptive_modulation'] = True
        config['adaptive_decay'] = 0.95
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] Phase 2 Full (task-aware adaptive threshold)\n")

    elif method == 'bayesian':
        # Bayesian: Kalman-inspired uncertainty-driven (TOO AGGRESSIVE)
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 2.0
        config['surprise_weight'] = 0.2
        config['warmup_steps'] = 500
        config['plasticity_threshold'] = 0.5
        config['trace_decay'] = 0.80  # Too fast!
        config['adaptive_threshold'] = True
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] Bayesian (variance-driven, trace_decay=0.80)\n")

    elif method == 'kalman':
        # Pure Kalman: K = P/(P+R)
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        # Simulating Kalman via conservative eligibility decay
        config['gate_temperature'] = 3.0  # Moderate
        config['surprise_weight'] = 0.3
        config['warmup_steps'] = 1000
        config['plasticity_threshold'] = 0.5  # P threshold
        config['trace_decay'] = 0.95  # Conservative! (simulates P growth)
        config['adaptive_threshold'] = True
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] Pure Kalman (conservative P, trace_decay=0.95)\n")

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # 存储结果
    # accuracy_matrix[i][j] = 在学完Task i后，对Task j的准确率
    accuracy_matrix = np.zeros((num_tasks, num_tasks))

    # 存储每个任务的test loader
    test_loaders = []
    for i in range(num_tasks):
        test_loaders.append(pm_dataset.get_task_data(i, train=False, batch_size=1000))

    print("-" * 80)
    print("Sequential Task Learning")
    print("-" * 80)

    # 顺序学习每个任务
    for task_id in range(num_tasks):
        print(f"\n[Task {task_id+1}/{num_tasks}] Learning...")

        # 获取训练数据
        train_loader = pm_dataset.get_task_data(task_id, train=True, batch_size=256)

        # 训练
        start_time = time.time()
        train_on_task(model, train_loader, optimizer, criterion,
                     epochs=epochs_per_task, verbose=False)
        train_time = time.time() - start_time

        # 评估所有已学习的任务
        print(f"  Training time: {train_time:.1f}s")
        print(f"  Evaluating on all tasks:")

        for eval_task_id in range(task_id + 1):
            acc = evaluate_on_task(model, test_loaders[eval_task_id])
            accuracy_matrix[task_id][eval_task_id] = acc

            if eval_task_id == task_id:
                print(f"    Task {eval_task_id+1} (current): {acc:.2f}%")
            else:
                print(f"    Task {eval_task_id+1}: {acc:.2f}%", end="")
                # 计算遗忘
                initial_acc = accuracy_matrix[eval_task_id][eval_task_id]
                forgetting = initial_acc - acc
                if forgetting > 5:
                    print(f" (forgot {forgetting:.2f}%)")
                else:
                    print()

    # 计算指标
    print("\n" + "=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    # Average Accuracy: 最后对所有任务的平均准确率
    final_avg_acc = accuracy_matrix[num_tasks-1, :].mean()

    # Average Forgetting
    forgetting_list = []
    for j in range(num_tasks - 1):  # 不包括最后一个任务（还没机会被遗忘）
        max_acc = accuracy_matrix[j][j]  # 刚学完时的准确率
        final_acc = accuracy_matrix[num_tasks-1][j]  # 最后的准确率
        forgetting_list.append(max_acc - final_acc)

    avg_forgetting = np.mean(forgetting_list)

    # Forward Transfer (学新任务对后续任务的帮助)
    # 暂不计算（需要zero-shot评估）

    # Backward Transfer (学新任务对之前任务的影响)
    backward_transfer = -avg_forgetting  # 负的遗忘就是正向迁移

    print(f"\nKey Metrics:")
    print(f"  Average Accuracy (final):  {final_avg_acc:.2f}%")
    print(f"  Average Forgetting:        {avg_forgetting:.2f}%")
    print(f"  Backward Transfer:         {backward_transfer:.2f}%")

    print(f"\nPer-task final accuracy:")
    for i in range(num_tasks):
        print(f"  Task {i+1}: {accuracy_matrix[num_tasks-1][i]:.2f}%")

    print("\n" + "=" * 80)

    results = {
        'accuracy_matrix': accuracy_matrix,
        'avg_accuracy': final_avg_acc,
        'avg_forgetting': avg_forgetting,
        'backward_transfer': backward_transfer,
        'method': method,
    }

    return results


def compare_methods():
    """对比不同方法"""

    print("=" * 80)
    print("PERMUTED MNIST - PURE KALMAN FILTER TEST")
    print("=" * 80)
    print("\nRunning 3 methods: Baseline, Phase 1, Pure Kalman")
    print("(Each task gets 1 epoch of training for speed)\n")
    print("NEW: 'kalman' - Pure K=P/(P+R), conservative P decay\n")

    methods = ['baseline', 'selective', 'kalman']
    all_results = {}

    for method in methods:
        all_results[method] = permuted_mnist_experiment(
            method=method,
            num_tasks=10,
            epochs_per_task=1
        )

    # 最终对比
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON - PERMUTED MNIST (10 TASKS)")
    print("=" * 80)

    print(f"\n{'Method':<25} {'Avg Accuracy':<15} {'Avg Forgetting':<15} {'Backward Transfer':<15}")
    print("-" * 80)

    for method in methods:
        r = all_results[method]
        print(f"{method:<25} "
              f"{r['avg_accuracy']:.2f}%{'':<8} "
              f"{r['avg_forgetting']:.2f}%{'':<8} "
              f"{r['backward_transfer']:.2f}%")

    print("-" * 80)

    # 找最佳方法
    best_acc = max(methods, key=lambda m: all_results[m]['avg_accuracy'])
    best_forgetting = min(methods, key=lambda m: all_results[m]['avg_forgetting'])

    print(f"\n[BEST] Highest average accuracy: {best_acc} "
          f"({all_results[best_acc]['avg_accuracy']:.2f}%)")
    print(f"[BEST] Lowest forgetting: {best_forgetting} "
          f"({all_results[best_forgetting]['avg_forgetting']:.2f}%)")

    # 改进幅度
    baseline_acc = all_results['baseline']['avg_accuracy']
    for method in ['selective', 'kalman']:
        if method in all_results:
            improvement = all_results[method]['avg_accuracy'] - baseline_acc
            print(f"\n{method} improvement over baseline: {improvement:+.2f}%")

    # Kalman分析
    if 'kalman' in all_results:
        print("\n" + "=" * 80)
        print("PURE KALMAN FILTER ANALYSIS")
        print("=" * 80)
        kalman_vs_selective = all_results['kalman']['avg_accuracy'] - all_results['selective']['avg_accuracy']
        print(f"Kalman vs Phase 1 Selective: {kalman_vs_selective:+.2f}%")
        print(f"Kalman accuracy: {all_results['kalman']['avg_accuracy']:.2f}%")
        print(f"Kalman forgetting: {all_results['kalman']['avg_forgetting']:.2f}%")
        print(f"Phase 1 forgetting: {all_results['selective']['avg_forgetting']:.2f}%")

        # Task-level comparison
        print(f"\nPer-task comparison (final accuracy):")
        print(f"Task    Baseline  Phase1   Kalman   Improvement")
        print("-" * 60)
        for i in range(10):
            b_acc = all_results['baseline']['accuracy_matrix'][-1, i]
            s_acc = all_results['selective']['accuracy_matrix'][-1, i]
            k_acc = all_results['kalman']['accuracy_matrix'][-1, i]
            improvement = k_acc - s_acc
            print(f"T{i+1:2d}     {b_acc:5.1f}%    {s_acc:5.1f}%   {k_acc:5.1f}%     {improvement:+5.1f}%")

    print("\n" + "=" * 80)
    print("[COMPLETE] Permuted MNIST benchmark finished!")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    results = compare_methods()

    print("\n" + "=" * 80)
    print("NOTES FOR PUBLICATION")
    print("=" * 80)
    print("\n1. This is a SINGLE RUN. For publication, you need:")
    print("   - Run 5-10 times with different seeds")
    print("   - Report mean ± std")
    print("   - Perform t-test for significance (p<0.05)")
    print("\n2. Current setup: 1 epoch per task (for speed)")
    print("   - For final results, use 3-5 epochs per task")
    print("\n3. Compare with SOTA methods:")
    print("   - EWC (Elastic Weight Consolidation)")
    print("   - SI (Synaptic Intelligence)")
    print("   - LwF (Learning without Forgetting)")
    print("\n4. For paper, create visualizations:")
    print("   - Accuracy matrix heatmap")
    print("   - Learning curves for each task")
    print("   - Forgetting curves")
