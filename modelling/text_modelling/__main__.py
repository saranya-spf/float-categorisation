import sys

sys.path.insert(0, "/Users/saranya.pal/Desktop/Projects/float_categorization")

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modelling.text_modelling.text_dataset import TextDataset
from modelling.text_modelling.data_collator import TextCollator


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
    )

    collator = TextCollator()
    
    data = TextDataset(df=df, include_labels=False)
    loader = DataLoader(dataset=data, batch_size=8, shuffle=False, collate_fn=collator)

    for txt in loader:
        print(txt)