import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# from float_categorization.deterministic.fuzzy_rule_classifier import FuzzyRuleClassifier
from float_categorization.pre_processing.pre_processor import PreProcessor
from float_categorization.modelling.models.ensemble_model import EnsembleClassifier

MODELS_DIR = Path(__file__).resolve().parents[2] / "models_dict"


# def run_deterministic(df: pd.DataFrame) -> pd.DataFrame:
#     fuzz_classifier = FuzzyRuleClassifier(df)
#     return fuzz_classifier.fit_rules(df, cols=[" Transaction Detail "])


def prepare_ml_data(df: pd.DataFrame, test_size: float = 0.2, min_samples: int = 2):
    processor = PreProcessor(df, is_training=True)
    processor()
    X, y = processor.process_for_training()
    y_labels = y.values.argmax(axis=1)
    label_names = y.columns.tolist()

    #   Filter out classes with fewer than min_samples
    #   Comment this below block later
    class_counts = np.bincount(y_labels)
    valid_mask = np.array([class_counts[label] >= min_samples for label in y_labels])
    X = X[valid_mask]
    y = y[valid_mask]

    #   Drop empty class columns and recompute labels
    y = y.loc[:, y.sum() > 0]
    label_names = y.columns.tolist()
    y_labels = y.values.argmax(axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y_labels, random_state=42
    )
    return X_train, X_test, y_train, y_test, label_names, processor


def train_ml_model(
    X_train, X_test, y_train, y_test, Classifier, **kwargs
) -> EnsembleClassifier:
    y_train_labels = y_train.values.argmax(axis=1)
    y_test_labels = y_test.values.argmax(axis=1)

    model = EnsembleClassifier(
        Classifier=Classifier, num_classes=y_train.shape[1], **kwargs
    )
    model.fit(X_train, y_train_labels)

    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test_labels, preds)
    f1 = f1_score(y_test_labels, preds, average="weighted")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 score (weighted): {f1:.4f}")

    return model


def run_ensemble_pipeline(
    file_path: str,
    Classifier,
    use_deterministic: bool = False,
    save_model: bool = True,
    save_dir: str = "",
    **kwargs,
) -> None:
    df = pd.read_csv(file_path)

    # if use_deterministic:
    #     result_df = run_deterministic(df)
    #     unmatched = result_df[result_df["predicted_label"].isna()]
    # else:
    result_df = df.copy()
    unmatched = result_df

    X_train, X_test, y_train, y_test, label_names, processor = prepare_ml_data(
        unmatched
    )
    model = train_ml_model(X_train, X_test, y_train, y_test, Classifier, **kwargs)

    if save_model:
        save_path = save_dir if save_dir else str(MODELS_DIR / "ensemble_model.joblib")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "label_names": label_names}, save_path)
        print(f"Model saved to {save_path}")


def inference_ensemble_models(
    file_path: str,
    model_path: str,
    use_deterministic: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # if use_deterministic:
    #     result_df = run_deterministic(df)
    #     unmatched = result_df[result_df["predicted_label"].isna()]
    # else:
    result_df = df.copy()
    unmatched = result_df

    processor = PreProcessor(unmatched, is_training=False)
    processor()
    X_all = processor.X_combined

    saved = joblib.load(model_path)
    model = saved["model"]
    label_names = saved["label_names"]

    preds = model.predict(X_all)
    result_df.loc[unmatched.index, "predicted_label"] = [label_names[p] for p in preds]

    return result_df


def evaluate_ensemble_models(
    file_path: str,
    model_path: str,
    output_path: str,
    use_deterministic: bool = False,
) -> None:
    df = pd.read_csv(file_path)

    # if use_deterministic:
    #     result_df = run_deterministic(df)
    #     unmatched = result_df[result_df["predicted_label"].isna()]
    # else:
    result_df = df.copy()
    unmatched = result_df

    X_train, X_test, y_train, y_test, label_names, processor = prepare_ml_data(
        unmatched
    )
    y_test_labels = y_test.values.argmax(axis=1)

    saved = joblib.load(model_path)
    model = saved["model"]

    preds = model.predict(X_test)

    results = pd.DataFrame(
        {
            "actual_label": [label_names[i] for i in y_test_labels],
            "predicted_label": [label_names[i] for i in preds],
            "correct": y_test_labels == preds,
        }
    )

    print(results.to_string())
    print(f"\nAccuracy: {results['correct'].mean():.3f}")
    print(
        f"F1 score (weighted): {f1_score(y_test_labels, preds, average='weighted'):.3f}"
    )

    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    
def fine_tune_ensemble(file_path: str, n_trials: int = 20):
    pass


if __name__ == "__main__":
    file_path = "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
    result_df = run_ensemble_pipeline(file_path, use_deterministic=False)
