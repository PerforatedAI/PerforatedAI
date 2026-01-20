
import torch
import wandb

def train_model(model, train_loader, val_loader, device, config):
    # Initialize WandB with the configuration
    wandb.init(project="pytorch-dendritic-optimization-hackathon", config=config)
    
    # Extract parameters from wandb.config (allows for future sweeps)
    lr = wandb.config.learning_rate
    epochs = wandb.config.epochs
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning Rate Scheduler: Helps improve accuracy by refining weights as training progresses
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=wandb.config.step_size, gamma=wandb.config.gamma)
    
    # Watch the model to visualize gradients and topology in WandB
    wandb.watch(model, log="all", log_freq=10)

    initial_val_loss = 0.0
    initial_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Step the scheduler and get current LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        train_loss /= len(train_loader)

        model.eval()
        correct, total, val_loss = 0, 0, 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc
        })

    wandb.finish()
