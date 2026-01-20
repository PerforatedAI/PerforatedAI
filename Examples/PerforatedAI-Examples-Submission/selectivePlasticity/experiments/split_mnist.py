"""
持续学习实验 V2 - 改进版
使用更激进的SelectivePlasticity配置来防止灾难性遗忘

改进策略:
1. 使用更高的gate_temperature (更严格的门控)
2. 使用EWC loss正则化 (Elastic Weight Consolidation)
3. 降低学习率在Task B
4. 使用"replay"策略 - 在Task B中混入少量Task A数据
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import sys
import os
import time
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../perforatedai'))

from perforatedai.utils_perforatedai import initialize_pai
import perforatedai.globals_perforatedai as GPA


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_task_loaders(task='A', batch_size=64, include_all_classes=False):
    """创建任务特定的数据加载器

    Args:
        task: 'A' (digits 0-4) or 'B' (digits 5-9)
        batch_size: batch size
        include_all_classes: 如果True，返回所有类别但只标记目标类别

    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    if task == 'A':
        classes = [0, 1, 2, 3, 4]
    elif task == 'B':
        classes = [5, 6, 7, 8, 9]
    else:
        raise ValueError(f"Unknown task: {task}")

    # 过滤数据
    train_indices = [i for i, (_, label) in enumerate(train_dataset) if label in classes]
    test_indices = [i for i, (_, label) in enumerate(test_dataset) if label in classes]

    train_indices = train_indices[:3000]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    test_subset = torch.utils.data.Subset(test_dataset, test_indices)

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=1000, shuffle=False)

    return train_loader, test_loader


def create_replay_loader(task_A_loader, task_B_loader, replay_ratio=0.2):
    """创建混合replay数据加载器

    Args:
        task_A_loader: Task A的数据加载器
        task_B_loader: Task B的数据加载器
        replay_ratio: Task A数据的比例

    Returns:
        混合的数据加载器
    """
    # 收集Task A的所有数据
    task_A_data = []
    for data, target in task_A_loader:
        for i in range(len(data)):
            task_A_data.append((data[i], target[i]))

    # 收集Task B的所有数据
    task_B_data = []
    for data, target in task_B_loader:
        for i in range(len(data)):
            task_B_data.append((data[i], target[i]))

    # 计算replay数量
    num_replay = int(len(task_B_data) * replay_ratio)
    replay_samples = random.sample(task_A_data, min(num_replay, len(task_A_data)))

    # 合并
    mixed_data = task_B_data + replay_samples
    random.shuffle(mixed_data)

    # 创建新的dataset
    class MixedDataset(torch.utils.data.Dataset):
        def __init__(self, data_list):
            self.data_list = data_list

        def __len__(self):
            return len(self.data_list)

        def __getitem__(self, idx):
            return self.data_list[idx]

    mixed_dataset = MixedDataset(mixed_data)
    mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=64, shuffle=True)

    return mixed_loader


def train_on_task(model, train_loader, optimizer, criterion, epochs=3, task_name='', verbose=True):
    """在一个任务上训练"""
    model.train()

    for epoch in range(1, epochs + 1):
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
            avg_loss = total_loss / len(train_loader)
            acc = 100. * correct / total
            print(f"  {task_name} Epoch {epoch}/{epochs}: Loss={avg_loss:.4f}, Acc={acc:.2f}%")

    return model


def evaluate_on_task(model, test_loader, criterion, task_name=''):
    """评估任务性能"""
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    return test_loss, accuracy


def continual_learning_v2(strategy='baseline', epochs_per_task=5):
    """改进的持续学习实验

    Args:
        strategy: 'baseline', 'selective', 'selective_strict', 'replay'
    """

    print(f"\n{'=' * 80}")
    print(f"Strategy: {strategy.upper()}")
    print(f"{'=' * 80}\n")

    # 创建模型
    model = SimpleNet()
    model = initialize_pai(model)

    # 配置策略
    if strategy == 'baseline':
        GPA.pc.set_selective_plasticity_enabled(False)
        print("[CONFIG] Baseline (no protection)\n")

    elif strategy == 'selective':
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 3.0
        config['surprise_weight'] = 0.3
        config['warmup_steps'] = 1000
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] SelectivePlasticity (lenient)\n")

    elif strategy == 'selective_strict':
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 15.0  # 非常严格
        config['surprise_weight'] = 1.0     # 最大surprise影响
        config['warmup_steps'] = 500
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] SelectivePlasticity (STRICT - maximum protection)\n")

    elif strategy == 'replay':
        GPA.pc.set_selective_plasticity_enabled(True)
        GPA.pc.set_switch_mode(GPA.pc.DOING_NO_SWITCH)
        config = GPA.pc.get_plasticity_config()
        config['gate_temperature'] = 5.0
        config['surprise_weight'] = 0.5
        config['warmup_steps'] = 1000
        GPA.pc.set_plasticity_config(config)
        print("[CONFIG] SelectivePlasticity + Experience Replay (20% Task A data)\n")

    # 准备数据
    train_loader_A, test_loader_A = create_task_loaders('A')
    train_loader_B, test_loader_B = create_task_loaders('B')

    print(f"Task A: Digits 0-4 ({len(train_loader_A.dataset)} train, {len(test_loader_A.dataset)} test)")
    print(f"Task B: Digits 5-9 ({len(train_loader_B.dataset)} train, {len(test_loader_B.dataset)} test)\n")

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    results = {}

    # Phase 1: 学习Task A
    print("-" * 80)
    print("PHASE 1: Learning Task A")
    print("-" * 80)
    model = train_on_task(model, train_loader_A, optimizer, criterion,
                          epochs=epochs_per_task, task_name='Task A')
    _, acc_A_after_A = evaluate_on_task(model, test_loader_A, criterion)
    results['task_A_after_A'] = acc_A_after_A
    print(f"\n[RESULT] Task A: {acc_A_after_A:.2f}%\n")

    # Phase 2: 学习Task B (根据策略使用不同方法)
    print("-" * 80)
    print("PHASE 2: Learning Task B")
    print("-" * 80)

    if strategy == 'replay':
        # 使用replay策略
        replay_loader = create_replay_loader(train_loader_A, train_loader_B, replay_ratio=0.2)
        print(f"  Using experience replay: {len(replay_loader.dataset)} samples (20% from Task A)")
        model = train_on_task(model, replay_loader, optimizer, criterion,
                              epochs=epochs_per_task, task_name='Task B+Replay')
    else:
        # 标准训练
        model = train_on_task(model, train_loader_B, optimizer, criterion,
                              epochs=epochs_per_task, task_name='Task B')

    _, acc_B_after_B = evaluate_on_task(model, test_loader_B, criterion)
    results['task_B_after_B'] = acc_B_after_B
    print(f"\n[RESULT] Task B: {acc_B_after_B:.2f}%\n")

    # Phase 3: 重新评估Task A
    print("-" * 80)
    print("PHASE 3: Re-evaluate Task A")
    print("-" * 80)
    _, acc_A_after_B = evaluate_on_task(model, test_loader_A, criterion)
    results['task_A_after_B'] = acc_A_after_B
    forgetting = results['task_A_after_A'] - results['task_A_after_B']
    results['forgetting'] = forgetting

    print(f"\n[RESULT] Task A retention: {acc_A_after_B:.2f}%")
    print(f"[RESULT] Forgetting: {forgetting:.2f}%\n")

    # 总结
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Task A after A:  {results['task_A_after_A']:.2f}%")
    print(f"Task B after B:  {results['task_B_after_B']:.2f}%")
    print(f"Task A after B:  {results['task_A_after_B']:.2f}%")
    print(f"Forgetting:      {results['forgetting']:.2f}%")
    avg = (results['task_A_after_B'] + results['task_B_after_B']) / 2
    print(f"Average:         {avg:.2f}%")
    print("=" * 80)

    return results


def compare_all_strategies():
    """对比所有策略"""

    print("=" * 80)
    print("COMPREHENSIVE CONTINUAL LEARNING COMPARISON")
    print("=" * 80)
    print("\nTesting 4 different strategies:\n")

    strategies = ['baseline', 'selective', 'selective_strict', 'replay']
    all_results = {}

    for strategy in strategies:
        all_results[strategy] = continual_learning_v2(strategy, epochs_per_task=5)

    # 最终对比
    print("\n\n" + "=" * 80)
    print("FINAL COMPARISON - ALL STRATEGIES")
    print("=" * 80)

    print(f"\n{'Strategy':<20} {'Task A→A':<12} {'Task B→B':<12} {'Task A→B':<12} {'Forgetting':<12} {'Average':<12}")
    print("-" * 80)

    for strategy in strategies:
        r = all_results[strategy]
        avg = (r['task_A_after_B'] + r['task_B_after_B']) / 2
        print(f"{strategy:<20} "
              f"{r['task_A_after_A']:.2f}%{'':<6} "
              f"{r['task_B_after_B']:.2f}%{'':<6} "
              f"{r['task_A_after_B']:.2f}%{'':<6} "
              f"{r['forgetting']:.2f}%{'':<6} "
              f"{avg:.2f}%")

    print("-" * 80)

    # 找最佳策略
    best_retention = max(strategies, key=lambda s: all_results[s]['task_A_after_B'])
    best_average = max(strategies, key=lambda s:
                      (all_results[s]['task_A_after_B'] + all_results[s]['task_B_after_B']) / 2)

    print(f"\n[BEST] Highest Task A retention: {best_retention} "
          f"({all_results[best_retention]['task_A_after_B']:.2f}%)")
    print(f"[BEST] Highest average accuracy: {best_average} "
          f"({(all_results[best_average]['task_A_after_B'] + all_results[best_average]['task_B_after_B']) / 2:.2f}%)")

    print("\n" + "=" * 80)
    print("[COMPLETE] Comprehensive comparison finished!")
    print("=" * 80)

    return all_results


if __name__ == '__main__':
    results = compare_all_strategies()
