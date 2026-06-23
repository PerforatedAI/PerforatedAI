from dataclasses import dataclass
from typing import Self, Tuple

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


@dataclass
class StocksDataLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


class StocksDataset(Dataset):
    def __init__(self: Self, X: np.ndarray, y: np.ndarray) -> None:
        self.sequences = torch.from_numpy(X).to(torch.float32)
        self.targets = torch.from_numpy(y).to(torch.float32)

    def __len__(self: Self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self: Self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]
