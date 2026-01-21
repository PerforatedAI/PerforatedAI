import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim

from model import Net
from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=True, download=True, transform=transform),
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("./data", train=False, transform=transform),
    batch_size=1000,
    shuffle=False
)

model = Net().to(device)
model = UPA.initialize_pai(model)

GPA.pai_tracker.set_optimizer(optim.Adadelta)
optimizer = GPA.pai_tracker.setup_optimizer(
    model, {"params": model.parameters(), "lr": 1.0}, None
)

def train(epoch):
    model.train()
    correct = 0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

    acc = 100.0 * correct / len(train_loader.dataset)
    GPA.pai_tracker.add_extra_score(acc, "train")
    print(f"Epoch {epoch} | Train Accuracy: {acc:.2f}%")

def validate():
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    acc = 100.0 * correct / len(test_loader.dataset)
    print(f"Validation Accuracy: {acc:.2f}%")

    new_model, restructured, _ = GPA.pai_tracker.add_validation_score(acc, model)

    if restructured:
        print(">>> MODEL RESTRUCTURED (Phase 2 triggered)")
        print(">>> Demo complete. Stopping after successful Phase-2.")
        exit(0)

    return new_model

for epoch in range(1, 4):
    train(epoch)
    model = validate()
