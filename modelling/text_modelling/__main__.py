import sys
import warnings

sys.path.insert(0, "/Users/saranya.pal/Desktop/Projects/float_categorization")

warnings.filterwarnings("ignore")

import pandas as pd
import torch

from modelling.text_modelling.classification_model import BERTClassifier
from modelling.text_modelling.trainer import Trainer, build_datasets


def run_pipeline(file_path: str):
    df = pd.read_csv(file_path)

    # print(df[" GL Code "].value_counts())

    X_train, X_test, y_train, y_test = build_datasets(df)

    model = BERTClassifier(fine_tune=True, num_classes=y_train.shape[1])
    trainer = Trainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    trainer.train(model=model)


def evaluate(file_path: str):
    df = pd.read_csv(file_path)
    X_train, X_test, y_train, y_test = build_datasets(df, test_size=0.2)

    model_path = "/Users/saranya.pal/Desktop/Projects/float_categorization/models_dict/best_model.pt"
    state_dict = torch.load(model_path, map_location="cpu")
    num_classes = state_dict["classifier.2.bias"].shape[0]

    model = BERTClassifier(fine_tune=True, num_classes=num_classes)
    trainer = Trainer(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    _, all_preds = trainer.predict(
        model,
        model_path,
        X_test,
    )

    class_names = y_test.columns.tolist()
    actual_labels = y_test.values.argmax(axis=1)

    results = pd.DataFrame(
        {
            "text": X_test.values,
            "actual_label": [class_names[i] for i in actual_labels],
            "predicted_label": [class_names[i] for i in all_preds],
            "correct": actual_labels == all_preds,
        }
    )

    print(results.to_string())
    print(f"\nAccuracy: {results['correct'].mean():.3f}")

    output_path = "/Users/saranya.pal/Desktop/Projects/float_categorization/analysis/evaluation_results.csv"
    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    FILE_PATH_1 = "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
    FILE_PATH_2 = "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Float Sample Data - Sheet2.csv"

    # run_pipeline(FILE_PATH)

    evaluate(FILE_PATH_1)
