import sys

sys.path.insert(0, "/Users/saranya.pal/Desktop/Projects/float_categorization")

import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modelling.text_modelling.text_dataset import TextDataset
from modelling.text_modelling.data_collator import TextCollator
from modelling.text_modelling.classification_model import BERTClassifier


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
    )

    collator = TextCollator()
    
    data = TextDataset(df=df, include_labels=True)
    loader = DataLoader(dataset=data, 
            batch_size=8, 
            shuffle=False, 
            collate_fn=collator
        )
    
    model = BERTClassifier()
    model.train()

    i = 0
    for batch_x, batch_y in loader:
        output = model(**batch_x)
        print("output:", output.shape)
        i += 1
        
        if i == 2:
            print(output)
            print(torch.argmax(output, dim=1))
            break