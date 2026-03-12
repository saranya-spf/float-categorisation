import sys

sys.path.insert(0, "/Users/saranya.pal/Desktop/Projects/float_categorization")

import pandas as pd

from modelling.text_modelling.classification_model import BERTClassifier
from modelling.text_modelling.trainer import Trainer, build_datasets


def run_pipeline(file_path: str):
    df = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = build_datasets(df)
    
    model = BERTClassifier(fine_tune=True, num_classes=y_train.shape[1])
    
    trainer = Trainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    trainer.train(model=model)
    

if __name__ == "__main__":
    FILE_PATH = "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
    run_pipeline(FILE_PATH)
