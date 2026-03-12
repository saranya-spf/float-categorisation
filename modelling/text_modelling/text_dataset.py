from typing import Dict, Any

import pandas as pd

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, X: pd.Series, y: pd.DataFrame):
        super().__init__()
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return {
            "text": self.X.iloc[index],
            "labels": torch.tensor(self.y.iloc[index].values, dtype=torch.float32),
        }
