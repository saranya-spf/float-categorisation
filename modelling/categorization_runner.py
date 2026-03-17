import pandas as pd
from sklearn.model_selection import train_test_split

from modelling.deterministic.fuzzy_rule_classifier import FuzzyRuleClassifier
from modelling.pre_processing.text_processor import TextProcessor
from modelling.modelling.trainer import Trainer
from modelling.modelling.models.classification_model import BERTClassifier


def run_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    fuzz_classifier = FuzzyRuleClassifier(df)
    return fuzz_classifier.fit_rules(df, cols=[" Transaction Detail "])


def prepare_ml_data(df: pd.DataFrame, test_size: float = 0.2):
    unmatched = df[df["predicted_label"].isna()]

    text_processor = TextProcessor(unmatched)
    X, y = text_processor.process_for_training()
    y_labels = y.values.argmax(axis=1)
    label_names = y.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y_labels, random_state=42
    )
    return X_train, X_test, y_train, y_test, unmatched.index, label_names


def train_ml_model(X_train, X_test, y_train, y_test) -> Trainer:
    model = BERTClassifier(fine_tune=True, num_classes=y_train.shape[1])
    trainer = Trainer(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        imbalance_training=True,
    )
    trainer.train(
        model=model,
        restore_best_weights=True,
        verbose=True,
    )
    return trainer


def run_pipeline(file_path: str, model_dict_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    result_df = run_deterministic(df)

    X_train, X_test, y_train, y_test, unmatched_idx, label_names = prepare_ml_data(result_df)
    trainer = train_ml_model(X_train, X_test, y_train, y_test)

    unmatched_X = result_df.loc[unmatched_idx, "aggregated_text"]
    model = BERTClassifier(fine_tune=True, num_classes=len(label_names))
    _, preds = trainer.predict(model, model_dict_path, unmatched_X)
    result_df.loc[unmatched_idx, "predicted_label"] = [label_names[p] for p in preds]

    return result_df
