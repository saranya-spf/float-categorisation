from typing import Dict, Any, Optional

import pandas as pd

import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, X: pd.Series, y: Optional[pd.DataFrame] = None):
        super().__init__()
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True) if y is not None else None

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {"text": self.X.iloc[index]}
        if self.y is not None:
            item["labels"] = torch.tensor(
                self.y.iloc[index].values, dtype=torch.float32
            )
        return item
