import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.axes import Axes
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from download import NASDAQDatasetInfo, NASDAQDownloader
from model import StockPredictionModel
from stocks import StocksDataLoaders, StocksDataset

# Initialize important variables
RANDOM_SEED: int = 1290
SEQUENCE_LENGTH: int = 30
TRAIN_FRACTION: float = 0.8
VAL_FRACTION: float = 0.1
BATCH_SIZE: int = 256
LEARNING_RATE: float = 0.0005
EPOCHS: int = 10
MODEL_INFO_DIR: Path = Path("model_info")

# ANSI escape codes
RED: str = "\033[31m"
RESET: str = "\033[0m"

torch.manual_seed(RANDOM_SEED)
device: torch.device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")


@dataclass
class ProcessedData:
    train_X: np.ndarray
    train_y: np.ndarray
    val_X: np.ndarray
    val_y: np.ndarray
    test_X: np.ndarray
    test_y: np.ndarray


def make_windows_per_ticker(stock: pl.DataFrame) -> ProcessedData:
    stock = stock.sort("Date")
    stock = stock.filter(pl.col("Close") > 0)

    if len(stock) < SEQUENCE_LENGTH + 10:
        empty_X: np.ndarray = np.empty((0, SEQUENCE_LENGTH - 1))
        empty_y: np.ndarray = np.empty((0,))
        return ProcessedData(empty_X, empty_y, empty_X, empty_y, empty_X, empty_y)

    prices: np.ndarray = stock["Close"].to_numpy()
    log_returns: np.ndarray = np.diff(np.log(prices))  # Calculates log return

    n: int = len(log_returns)
    train_idx: int = int(TRAIN_FRACTION * n)
    val_idx: int = int((TRAIN_FRACTION + VAL_FRACTION) * n)

    def windowing(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i : i + SEQUENCE_LENGTH - 1])
            y.append(data[i + SEQUENCE_LENGTH - 1])
        return np.array(X), np.array(y)

    train_X, train_y = windowing(log_returns[:train_idx])
    val_X, val_y = windowing(log_returns[train_idx:val_idx])
    test_X, test_y = windowing(log_returns[val_idx:])

    return ProcessedData(train_X, train_y, val_X, val_y, test_X, test_y)


def prepare_data(stocks_dir: Path) -> Tuple[StocksDataLoaders, StandardScaler]:
    all_train_X, all_train_y = [], []
    all_val_X, all_val_y = [], []
    all_test_X, all_test_y = [], []

    files: List[Path] = list(stocks_dir.glob("*.csv"))
    for file in tqdm(files, desc=f"Parsing all CSVs from {stocks_dir}"):
        df: pl.DataFrame = pl.read_csv(file).drop_nulls()
        # Ensures that model has enough context to learn something
        if len(df) < SEQUENCE_LENGTH + 10:
            continue

        processed_data: ProcessedData = make_windows_per_ticker(df)

        # .size = # of elements
        if processed_data.train_X.size > 0:
            all_train_X.append(processed_data.train_X)
            all_train_y.append(processed_data.train_y)
        if processed_data.val_X.size > 0:
            all_val_X.append(processed_data.val_X)
            all_val_y.append(processed_data.val_y)
        if processed_data.test_X.size > 0:
            all_test_X.append(processed_data.test_X)
            all_test_y.append(processed_data.test_y)

    train_X: np.ndarray = np.vstack(all_train_X)
    train_y: np.ndarray = np.concatenate(all_train_y)

    val_X: np.ndarray = np.vstack(all_val_X)
    val_y: np.ndarray = np.concatenate(all_val_y)

    test_X: np.ndarray = np.vstack(all_test_X)
    test_y: np.ndarray = np.concatenate(all_test_y)

    scaler: StandardScaler = StandardScaler()
    # Scaler fits only on the training targets
    train_y_raw: np.ndarray = train_y.reshape(-1, 1)
    scaler.fit(train_y_raw)

    # Flatten to scale, then reshape back
    train_X_shape: Tuple = train_X.shape
    train_X = scaler.transform(train_X.reshape(-1, 1)).reshape(train_X_shape)  # pyright: ignore[reportAttributeAccessIssue]

    val_X_shape: Tuple = val_X.shape
    val_X = scaler.transform(val_X.reshape(-1, 1)).reshape(val_X_shape)  # pyright: ignore[reportAttributeAccessIssue]

    test_X_shape: Tuple = test_X.shape
    test_X = scaler.transform(test_X.reshape(-1, 1)).reshape(test_X_shape)  # pyright: ignore[reportAttributeAccessIssue]

    # Scale targets
    train_y = scaler.transform(train_y_raw).flatten()  # pyright: ignore[reportAttributeAccessIssue]
    val_y = scaler.transform(val_y.reshape(-1, 1)).flatten()  # pyright: ignore[reportAttributeAccessIssue]
    test_y = scaler.transform(test_y.reshape(-1, 1)).flatten()  # pyright: ignore[reportAttributeAccessIssue]

    train_loader: DataLoader = DataLoader(
        StocksDataset(train_X, train_y),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader: DataLoader = DataLoader(
        StocksDataset(val_X, val_y),
        batch_size=BATCH_SIZE,
    )
    test_loader: DataLoader = DataLoader(
        StocksDataset(test_X, test_y),
        batch_size=BATCH_SIZE,
    )
    return StocksDataLoaders(train_loader, val_loader, test_loader), scaler


def train_step(
    model: StockPredictionModel,
    train_loader: DataLoader,
    criterion: nn.MSELoss,
    optimizer: optim.Adam,
    epoch: int,
) -> float:
    model.train()
    train_loss: float = 0
    for X_train, y_train in tqdm(
        train_loader, desc=f"Number of Train Batches Left for Epoch - {epoch}"
    ):
        X_train = X_train.unsqueeze(-1).to(device)
        y_train = y_train.unsqueeze(-1).to(device)

        y_train_pred: torch.Tensor = model(X_train)
        loss: torch.Tensor = criterion(y_train_pred, y_train)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return train_loss / len(train_loader)


def val_step(
    model: StockPredictionModel,
    val_loader: DataLoader,
    scaler: StandardScaler,
    scheduler: ReduceLROnPlateau,
    epoch: int,
) -> Tuple[float, float]:
    model.eval()
    val_predictions: List[torch.Tensor] = []
    val_targets: List[torch.Tensor] = []
    dim_accuracy: float = 0.0
    with torch.no_grad():
        for X_val, y_val in tqdm(
            val_loader, desc=f"Number of Val Batches Left for Epoch - {epoch}"
        ):
            X_val = X_val.unsqueeze(-1).to(device)
            y_val = y_val.unsqueeze(-1)

            y_val_pred: torch.Tensor = model(X_val)

            val_predictions.append(y_val_pred.cpu())
            val_targets.append(y_val)

    final_predictions_scaled: torch.Tensor = torch.vstack(val_predictions)
    final_targets_scaled: torch.Tensor = torch.vstack(val_targets)

    final_predictions: np.ndarray = scaler.inverse_transform(
        final_predictions_scaled.numpy()
    )
    final_targets: np.ndarray = scaler.inverse_transform(final_targets_scaled.numpy())

    dim_correct: np.ndarray = np.sign(final_predictions) == np.sign(final_targets)
    dim_accuracy = np.mean(dim_correct).item()

    val_rmse: float = root_mean_squared_error(final_targets, final_predictions)

    scheduler.step(val_rmse)

    return val_rmse, dim_accuracy


def plot_model_performance(
    all_losses: List[float], all_rsme: List[float], all_dim_accuracies: List[float]
) -> None:
    print("Saving model performance...")

    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    color = "tab:blue"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Training Loss (MSE)", color=color)
    ax1.plot(all_losses, color=color, label="Train Loss")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2: Axes = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("Val RMSE (Log Returns)", color=color)
    ax2.plot(all_rsme, color=color, label="Val RMSE")
    ax2.tick_params(axis="y", labelcolor=color)
    ax1.set_title("Loss and Error Over Epochs")

    ax3.plot(all_dim_accuracies, color="tab:green", label="Directional Accuracy")
    ax3.axhline(
        y=0.50, color="black", linestyle="--", alpha=0.5, label="Random Guess (50%)"
    )
    ax3.set_xlabel("Epochs")
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_title("Model Directional Accuracy")

    fig.tight_layout()
    plt.savefig(MODEL_INFO_DIR / "baseline_model_performance.png")


def main() -> None:
    model_save_dir: Path = MODEL_INFO_DIR / "models"
    os.makedirs(model_save_dir, exist_ok=True)

    downloader: NASDAQDownloader = NASDAQDownloader()
    info: NASDAQDatasetInfo = downloader.download_dataset(stop_if_dest_dir_exists=True)

    data_loaders, scaler = prepare_data(info.stocks_directory)

    model: StockPredictionModel = StockPredictionModel(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        output_dim=1,
        dropout=0.3,
        device=device,
    ).to(device)
    criterion: nn.MSELoss = nn.MSELoss().to(device)
    optimizer: optim.Adam = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-5,  # Weight decay penalizes large weights
    )
    scheduler: ReduceLROnPlateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    all_losses: List[float] = []
    all_rmse: List[float] = []
    all_dim_accuracies: List[float] = []
    for epoch in tqdm(range(EPOCHS), desc="Number of Epochs Left"):
        print()

        losses: float = train_step(
            model, data_loaders.train, criterion, optimizer, epoch
        )
        all_losses.append(losses)

        print()

        rsme, dim_accuray = val_step(model, data_loaders.val, scaler, scheduler, epoch)
        all_rmse.append(rsme)
        all_dim_accuracies.append(dim_accuray)

        print()

        torch.save(model.state_dict(), model_save_dir / f"model_{epoch}.pt")
    plot_model_performance(all_losses, all_rmse, all_dim_accuracies)
    print("Model training complete!")


def float_in_range(low: float, high: float) -> Callable:
    def checker(value: str) -> float:
        try:
            f_value = float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"{RED}'{value}' is not a valid float.{RESET}"
            )

        f_value = float(value)
        if f_value < low or f_value >= high:
            raise argparse.ArgumentTypeError(
                f"{RED}Value must be in [{low}, {high}){RESET}"
            )
        return f_value

    return checker


def parse_args() -> argparse.Namespace:
    global \
        RANDOM_SEED, \
        SEQUENCE_LENGTH, \
        TRAIN_FRACTION, \
        VAL_FRACTION, \
        BATCH_SIZE, \
        LEARNING_RATE, \
        EPOCHS, \
        MODEL_INFO_DIR

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Dendritic LSTM Stock Prediction Model",
        description="This program trains a Dendritic LSTM Stock Prediction Model on data from thousands of companies from the NASDAQ.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--random_seed", help="Set the random seed.", type=int, default=RANDOM_SEED
    )
    parser.add_argument(
        "--sequence_length",
        help="How many days the model should consider to predict the price for the next day.",
        type=int,
        default=SEQUENCE_LENGTH,
    )
    parser.add_argument(
        "--train_fraction",
        help="How big the training dataset should be. This parameter should be in [0, 1).",
        type=float_in_range(0.0, 1.0),
        default=TRAIN_FRACTION,
    )
    parser.add_argument(
        "--val_fraction",
        help="How big the validation dataset should be. This parameter should be in [0, 1).",
        type=float_in_range(0.0, 1.0),
        default=VAL_FRACTION,
    )
    parser.add_argument(
        "--batch_size",
        help="How big each batch size should be. This parameter should ideally be a power of 2.",
        type=int,
        default=BATCH_SIZE,
    )
    parser.add_argument(
        "--learning_rate",
        help="The model learning rate.",
        type=float,
        default=LEARNING_RATE,
    )
    parser.add_argument(
        "--epochs",
        help="The number of epochs.",
        type=int,
        default=EPOCHS,
    )
    parser.add_argument(
        "--model_info_dir",
        help="The directory to save all model related stuff to.",
        type=str,
        default=MODEL_INFO_DIR,
    )
    args: argparse.Namespace = parser.parse_args()

    if args.train_fraction + args.val_fraction >= 1.0:
        parser.error(
            f"{RED}The sum of --train_fraction ({args.train_fraction}) and "
            f"--val_fraction ({args.val_fraction}) must be less than 1.0 "
            f"to leave room for the test set.{RESET}"
        )
    return args


if __name__ == "__main__":
    args: argparse.Namespace = parse_args()

    RANDOM_SEED = args.random_seed
    torch.manual_seed(RANDOM_SEED)
    SEQUENCE_LENGTH = args.sequence_length
    TRAIN_FRACTION = args.train_fraction
    VAL_FRACTION = args.val_fraction
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    EPOCHS = args.epochs
    MODEL_INFO_DIR = Path(args.model_info_dir)

    main()
