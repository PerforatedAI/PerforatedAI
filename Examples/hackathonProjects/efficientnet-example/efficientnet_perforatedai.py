import logging
import argparse

import torch
import torchvision
import sklearn

from perforatedai import globals_perforatedai as GPA
from perforatedai import utils_perforatedai as UPA


def train_model(path_dataset: str,
                efficientnet_version: int,
                validation_size: int,
                learning_rate: float,
                batch_size: int):
    # 1. Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. Load dataset
    data_transformers = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = torchvision.datasets.ImageFolder(
        root=path_dataset,
        transform=data_transformers,
    )

    dataset_size = len(dataset)
    train_size = int((1 - validation_size / 100) * dataset_size)
    validation_size = dataset_size - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    class_names = dataset.classes

    # 3. Create model
    if efficientnet_version == 1:
        model = torchvision.models.efficientnet_b1(weights='DEFAULT')
    elif efficientnet_version == 2:
        model = torchvision.models.efficientnet_b2(weights='DEFAULT')
    elif efficientnet_version == 3:
        model = torchvision.models.efficientnet_b3(weights='DEFAULT')
    elif efficientnet_version == 4:
        model = torchvision.models.efficientnet_b4(weights='DEFAULT')
    elif efficientnet_version == 5:
        model = torchvision.models.efficientnet_b5(weights='DEFAULT')
    elif efficientnet_version == 6:
        model = torchvision.models.efficientnet_b6(weights='DEFAULT')
    elif efficientnet_version == 7:
        model = torchvision.models.efficientnet_b7(weights='DEFAULT')
    else:
        model = torchvision.models.efficientnet_b0(weights='DEFAULT')

    feature_count = model.classifier[-1].in_features
    model.classifier = torch.nn.Linear(feature_count, len(class_names)).to(device)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    # 4. Set up the optimizer, the loss and the learning rate scheduler
    criterion = torch.nn.CrossEntropyLoss()
    model.global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)

    GPA.pc.set_testing_dendrite_capacity(False)

    GPA.pc.append_module_names_to_convert(['MBConv', 'Conv2dNormActivation', ])
    GPA.pc.set_module_ids_to_convert(['.classifier'])
    GPA.pc.set_modules_to_convert([])
    GPA.pc.set_module_names_to_track(['MBConv', 'Conv2dNormActivation', 'Linear', 'Conv2d'])

    model = UPA.initialize_pai(model, maximizing_score=False)

    GPA.pai_tracker.set_optimizer(torch.optim.Adam)
    GPA.pai_tracker.set_scheduler(torch.optim.lr_scheduler.ReduceLROnPlateau)
    optimizer_args = {'params': model.parameters(), 'lr': learning_rate}
    scheduler_args = {'mode': 'min', 'patience': 5, 'factor': 0.001}
    optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimizer_args, scheduler_args)

    # For small datasets set GPA.pc.set_initial_correlation_batches to a number less than the
    # total batches you have in one epoch.
    if len(train_dataset) < 1000:
        GPA.pc.set_initial_correlation_batches(
            (len(train_dataset) / batch_size) - 1
        )

    model.to(device)

    # 5. Begin training
    epochs_trained = 0
    training_loss_best = 200
    training_accuracy_best = 0
    validation_loss_best = 200
    validation_accuracy_best = 0
    logging.info(f'''Starting training:
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {len(train_dataset)}
            Validation size: {len(validation_dataset)}
        ''')
    while True:
        epochs_trained += 1
        all_labels = []
        all_preds = []
        logging.info(f'Training epoch {epochs_trained}')
        model.train()

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        # Calculate metrics
        training_loss = loss.item()
        training_accuracy = sklearn.metrics.accuracy_score(
            all_labels,
            all_preds)
        GPA.pai_tracker.add_extra_score(training_loss, 'Train')

        if training_loss < training_loss_best:
            training_loss_best = training_loss
        if training_accuracy > training_accuracy_best:
            training_accuracy_best = training_accuracy

        logging.info(f'''Training results epoch {epochs_trained}: 
        training loss: {training_loss}
        training accuracy: {training_accuracy}
        best training loss: {training_loss_best}
        best training accuracy: {training_accuracy_best}''')

        # Evaluation round
        model.eval()

        all_val_labels = []
        all_val_preds = []
        validation_loss = 0.0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds.cpu().numpy())

        # Calculate Validation metrics
        # Average validation loss
        validation_loss /= len(validation_loader)
        validation_accuracy = (torch.tensor(all_val_labels) == torch.tensor(all_val_preds)).float().mean().item()

        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(validation_loss, model)

        if validation_loss < validation_loss_best:
            validation_loss_best = validation_loss
        if validation_accuracy > validation_accuracy_best:
            validation_accuracy_best = validation_accuracy

        logging.info(f'''Validation results epoch {epochs_trained}:
        validation loss: {validation_loss}
        validation accuracy: {validation_accuracy}
        best validation loss: {validation_loss_best}
        best validation accuracy: {validation_accuracy_best}
        training complete: {training_complete}
        ''')
        if training_complete:
            # Break the loop or do whatever you need to do once training is over
            pass
        elif restructured:
            optimizer_args = {'params': model.parameters(), 'lr': learning_rate}
            scheduler_args = {'mode': 'min', 'patience': 5, 'factor': 0.001}
            optimizer, scheduler = GPA.pai_tracker.setup_optimizer(model, optimizer_args, scheduler_args)

        # End training
        if training_complete:
            # Stop training
            break


def get_args():
    parser = argparse.ArgumentParser(description='Train EfficientNet on an inferred dataset.')
    parser.add_argument('--path-dataset', '-p', required=True, dest='path_dataset', type=str,
                        help='The path of the inferred dataset folder')
    parser.add_argument('--efficientnet-version', '-f', dest='efficientnet_version', metavar='F',
                        type=int, default=0, help='EfficientNet version (0-7)')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--learning-rate', '-l', dest='lr', metavar='LR', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--validation', '-v', dest='val', type=int, default=20,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        train_model(
            path_dataset=args.path_dataset,
            efficientnet_version=args.efficientnet_version,
            validation_size=args.val,
            learning_rate=args.lr,
            batch_size=args.batch_size)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! ')
        torch.cuda.empty_cache()
        train_model(
            path_dataset=args.path_dataset,
            efficientnet_version=args.efficientnet_version,
            validation_size=args.val,
            learning_rate=args.lr,
            batch_size=args.batch_size)
