from typing import Optional, Dict, Any
import pandas as pd

import torch
from torch.utils.data import Dataset

from modelling.text_modelling.text_processor import TextProcessor


class TextDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        include_labels: bool = True,
        cols_to_keep: Optional[list] = None,
    ):
        super().__init__()
        self.include_labels = include_labels
        self.processor = TextProcessor(df, cols_to_keep=cols_to_keep)

        self.X, self.y = self.processor.process_text()
        self.label_columns = self.y.columns.tolist()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = {"text": self.X.iloc[index]}

        if self.include_labels:
            item["labels"] = torch.tensor(
                self.y.iloc[index].values, dtype=torch.float32
            )

        return item
