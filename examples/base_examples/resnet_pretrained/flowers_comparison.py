import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from perforatedai import utils_perforatedai as UPA


NUM_CLASSES = 102


def set_fixed_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_torchvision_resnet18(num_classes):
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_perforated_cascor_resnet18(num_classes):
    from perforatedai import library_perforatedai as LPA

    base_model = torchvision.models.get_model('resnet18', weights=None, num_classes=1000)
    model = LPA.ResNetPAIPreFC(base_model)
    model = UPA.from_hf_pretrained(model, 'perforated-ai/resnet-18-perforated-cascor')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_logits(output):
    return output.logits if hasattr(output, 'logits') else output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = get_logits(model(data))
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = get_logits(model(data))
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    return test_loss, accuracy


def run_training(args, device, train_loader, test_loader, model_name, model_builder):
    print(f'\n===== Running model: {model_name} =====')
    model = model_builder(NUM_CLASSES).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.0)

    last_loss = None
    acc_history = []
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        last_loss, last_acc = test(model, device, test_loader)
        acc_history.append(last_acc)
        scheduler.step()

    if args.save_model:
        output_name = model_name.replace('/', '_').replace('-', '_') + '.pt'
        torch.save(model.state_dict(), output_name)

    return last_loss, acc_history


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Flowers-102 Transfer Learning Comparison')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1e-3)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    set_fixed_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    dataset1 = datasets.Flowers102('../data', split='train', download=True, transform=train_transform)
    dataset2 = datasets.Flowers102('../data', split='test', download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    results = []
    for model_name, builder in [
        ('torchvision/resnet-18', build_torchvision_resnet18),
        ('perforated-ai/resnet-18-perforated-cascor', build_perforated_cascor_resnet18),
    ]:
        loss, acc_history = run_training(args, device, train_loader, test_loader, model_name, builder)
        results.append((model_name, loss, acc_history))

    print('\n===== Final Comparison =====')
    result_map = {model_name: acc_history for model_name, _, acc_history in results}
    tv_hist = result_map['torchvision/resnet-18']
    pai_hist = result_map['perforated-ai/resnet-18-perforated-cascor']

    print(f"{'epoch':>6} {'tv_score':>10} {'tv_delta':>10} {'pai_score':>10} {'pai_delta':>10}")
    for i in range(len(tv_hist)):
        tv_score = tv_hist[i]
        pai_score = pai_hist[i]
        tv_delta = 'N/A' if i == 0 else f'{(tv_hist[i] - tv_hist[i - 1]):+.2f}'
        pai_delta = 'N/A' if i == 0 else f'{(pai_hist[i] - pai_hist[i - 1]):+.2f}'
        print(f"{i + 1:6d} {tv_score:10.2f} {tv_delta:>10} {pai_score:10.2f} {pai_delta:>10}")


if __name__ == '__main__':
    main()
