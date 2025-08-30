from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


class Net(nn.Module):
    def __init__(self, num_classes, width):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, int(32 * width), 3, 1)
        self.conv2 = nn.Conv2d(int(32 * width), int(64 * width), 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

        self.fc1 = nn.Linear(144 * int(64 * width), int(128 * width))
        self.fc2 = nn.Linear(int(128 * width), num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            if args.dry_run:
                break
        # Determine the predictions the network was making
        pred = output.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        # Increment how many times it was correct
        correct += pred.eq(target.view_as(pred)).sum()
    # Add the new score to the tracker which may restructured the model with PB Nodes
    GPA.pai_tracker.add_extra_score(
        100.0 * correct / len(train_loader.dataset), "train"
    )
    model.to(device)


def test(model, device, test_loader, optimizer, scheduler, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    # Display Metrics
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    # Add the new score to the tracker which may restructured the model with PB Nodes
    model, restructured, training_complete = GPA.pai_tracker.add_validation_score(
        100.0 * correct / len(test_loader.dataset), model
    )
    model.to(device)
    # If it was restructured reset the optimizer and scheduler
    if restructured:
        optimArgs = {"params": model.parameters(), "lr": args.lr}
        schedArgs = {"step_size": 1, "gamma": args.gamma}
        optimizer, scheduler = GPA.pai_tracker.setup_optimizer(
            model, optimArgs, schedArgs
        )
    return model, optimizer, scheduler, training_complete


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("--save-name", type=str, default="PB")
    parser.add_argument("--dataset", type=str, default="MNIST")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10000,
        metavar="N",
        help="number of epochs to train (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--width", type=float, default=1.0, metavar="M", help="width multiplier"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--no-mps",
        action="store_true",
        default=False,
        help="disables macOS GPU training",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": args.batch_size}
    test_kwargs = {"batch_size": args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    if args.dataset == "MNIST":
        num_classes = 10
        # Define the data loaders
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        dataset1 = datasets.MNIST(
            "./data", train=True, download=True, transform=transform
        )
        dataset2 = datasets.MNIST("./data", train=False, transform=transform)
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    elif args.dataset == "EMNIST":
        num_classes = 47
        transform_train = transforms.Compose(
            [
                transforms.CenterCrop(26),
                transforms.Resize((28, 28)),
                transforms.RandomRotation(10),
                transforms.RandomAffine(5),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        # Dataset
        dataset1 = datasets.EMNIST(
            root="./data",
            split="balanced",
            train=True,
            download=True,
            transform=transform_train,
        )

        dataset2 = datasets.EMNIST(
            root="./data",
            split="balanced",
            train=False,
            download=True,
            transform=transform_test,
        )
        train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
        test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # Set up some global parameters for PAI code
    GPA.testing_dendrite_capacity = True

    model = Net(num_classes, args.width).to(device)

    model = UPA.initialize_pai(model)

    # Setup the optimizer and scheduler
    GPA.pai_tracker.set_optimizer(optim.Adadelta)
    GPA.pai_tracker.set_scheduler(StepLR)
    optimArgs = {"params": model.parameters(), "lr": args.lr}
    schedArgs = {"step_size": 1, "gamma": args.gamma}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        model, optimizer, scheduler, training_complete = test(
            model, device, test_loader, optimizer, scheduler, args
        )
        if training_complete:
            break

    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


if __name__ == "__main__":
    main()
