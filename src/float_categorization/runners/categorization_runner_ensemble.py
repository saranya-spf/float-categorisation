import numpy as np
import pandas as pd
import joblib
import spacy
import scipy.sparse as sp
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from float_categorization.modelling.models.ensemble_model import EnsembleClassifier
from float_categorization.pre_processing.pre_processor import PreProcessor
from float_categorization.runners.decide_deterministic import deterministic_decider
from float_categorization.constant import LABEL_COL


MODELS_DIR = Path(__file__).resolve().parents[3] / "models_dict"


def prepare_ml_data(df: pd.DataFrame, test_size: float = 0.2):
    processor = PreProcessor(df, is_training=True)
    processor()
    X, y = processor.process_for_training()
    y_labels = y.values.argmax(axis=1)
    label_names = y.columns.tolist()

    # Identify classes with too few samples for stratified split.
    # Classes with only 1 sample go to train only (can't stratify-split them).
    class_counts = np.bincount(y_labels, minlength=len(label_names))
    min_for_split = max(
        2, int(np.ceil(1.0 / test_size))
    )  # need at least 1 in each split

    splittable_mask = np.array(
        [class_counts[label] >= min_for_split for label in y_labels]
    )
    rare_mask = ~splittable_mask

    # Rare samples go directly to training set
    X_rare = X[rare_mask]
    y_rare = y[rare_mask]

    # Splittable samples get stratified split
    X_split = X[splittable_mask]
    y_split = y[splittable_mask]
    y_split_labels = y_split.values.argmax(axis=1)

    X_train_s, X_test, y_train_s, y_test = train_test_split(
        X_split, y_split, test_size=test_size, stratify=y_split_labels, random_state=42
    )

    # Combine rare samples into training set
    X_train = sp.vstack([X_train_s, X_rare]) if rare_mask.any() else X_train_s
    y_train = pd.concat([y_train_s, y_rare], axis=0) if rare_mask.any() else y_train_s

    # Compute balanced class weights from training labels
    y_train_labels = y_train.values.argmax(axis=1)
    unique_classes = np.unique(y_train_labels)
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train_labels)
    class_weights = {int(c): float(w) for c, w in zip(unique_classes, weights)}

    n_rare = int(rare_mask.sum())
    if n_rare > 0:
        rare_classes = [label_names[c] for c in np.unique(y_labels[rare_mask])]
        print(f"Rare classes ({n_rare} samples, train-only): {rare_classes}")
    print(
        f"Class weight range: {min(class_weights.values()):.3f} - {max(class_weights.values()):.3f}"
    )

    return X_train, X_test, y_train, y_test, label_names, processor, class_weights


def train_ml_model(
    X_train,
    X_test,
    y_train,
    y_test,
    label_names,
    Classifier,
    feature_names=None,
    class_weights=None,
    **kwargs,
) -> EnsembleClassifier:
    y_train_labels = y_train.values.argmax(axis=1)
    y_test_labels = y_test.values.argmax(axis=1)

    model = EnsembleClassifier(
        Classifier=Classifier,
        num_classes=y_train.shape[1],
        class_weights=class_weights,
        **kwargs,
    )
    model.fit(X_train, y_train_labels)
    model.get_feature_importance(feature_names)

    preds = np.asarray(model.predict(X_test)).astype(int).ravel()
    accuracy = accuracy_score(y_test_labels, preds)
    f1_w = f1_score(y_test_labels, preds, average="weighted")
    f1_m = f1_score(y_test_labels, preds, average="macro")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test F1 score (weighted): {f1_w:.4f}")
    print(f"Test F1 score (macro):    {f1_m:.4f}")

    # Per-class report so minority class performance is visible
    test_label_names = [
        label_names[i]
        for i in sorted(np.unique(np.concatenate([y_test_labels, preds])))
    ]
    print(
        f"\n{classification_report(y_test_labels, preds, target_names=test_label_names, zero_division=0)}"
    )

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
    matched, unmatched = deterministic_decider(df, use_deterministic)

    # Remove rows whose label belongs to a category the deterministic already handles,
    # so the ML model is trained only on the remaining label categories.
    # The deterministic predicted_labels are spacy-processed (lemmatized, no stopwords/punct),
    # but unmatched gl_code values are only lowercased (ManualFeatureTransformer).
    # Build a mapping: raw_lowered_label -> spacy_cleaned_label to bridge the gap.
    if use_deterministic and not matched.empty:
        deterministic_labels = set(matched["predicted_label"].unique())
        label_col = LABEL_COL.strip().casefold()

        nlp = spacy.load("en_core_web_sm")
        nlp.disable_pipes("ner", "parser")
        unique_labels = unmatched[label_col].dropna().unique().tolist()
        raw_to_cleaned = {}
        for label in unique_labels:
            doc = nlp(label)
            cleaned = " ".join(
                t.lemma_
                for t in doc
                if not t.is_stop and not t.is_punct and t.text.strip()
            )
            raw_to_cleaned[label] = cleaned

        # Find raw labels whose spacy-cleaned form is a deterministic label
        raw_labels_to_remove = {
            raw
            for raw, cleaned in raw_to_cleaned.items()
            if cleaned in deterministic_labels
        }

        before = len(unmatched)
        unmatched = unmatched[~unmatched[label_col].isin(raw_labels_to_remove)]
        print(
            f"Deterministic covers {len(deterministic_labels)} labels: {sorted(deterministic_labels)}"
        )
        print(f"Mapped raw labels to remove: {sorted(raw_labels_to_remove)}")
        print(
            f"Filtered {before - len(unmatched)} rows with deterministic labels from ML training data "
            f"({before} -> {len(unmatched)} rows)"
        )

    use_class_weights = kwargs.pop("use_class_weights", True)

    X_train, X_test, y_train, y_test, label_names, processor, class_weights = (
        prepare_ml_data(unmatched, test_size=kwargs.pop("test_size", 0.2))
    )

    feature_names = list(processor.X_initial.columns) + list(
        processor.tf_encoder.vectorizer.get_feature_names_out()
    )
    # n_manual = len(processor.X_initial.columns)
    # n_tfidf = len(processor.tf_encoder.vectorizer.get_feature_names_out())
    # print(f"\n--- All features ({len(feature_names)}) = {n_manual} manual + {n_tfidf} TF-IDF ---")
    # print(f"  Manual: {', '.join(processor.X_initial.columns)}")
    # print(f"  TF-IDF: {', '.join(processor.tf_encoder.vectorizer.get_feature_names_out())}")

    # # Save pre-processed data to CSV
    # preprocessed_path = Path(file_path).resolve().parent / "preprocessed_training_data.csv"
    # processor.processed_df.to_csv(preprocessed_path, index=False)
    # print(f"\nPre-processed data saved to {preprocessed_path}")

    model = train_ml_model(
        X_train,
        X_test,
        y_train,
        y_test,
        label_names,
        Classifier,
        feature_names=feature_names,
        class_weights=class_weights if use_class_weights else None,
        **kwargs,
    )

    if save_model:
        save_path = save_dir if save_dir else str(MODELS_DIR / "ensemble_model.joblib")
        save_path = Path(save_path)
        if save_path.is_dir() or not save_path.suffix:
            save_path = save_path / "ensemble_model.joblib"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "label_names": label_names,
                "tf_encoder": processor.tf_encoder,
            },
            str(save_path),
        )
        print(f"Model saved to {save_path}")


def inference_ensemble_models(
    file_path: str,
    model_path: str,
    use_deterministic: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    matched, unmatched = deterministic_decider(df, use_deterministic)

    processor = PreProcessor(unmatched, is_training=True)
    processor()

    saved = joblib.load(model_path)
    model = saved["model"]
    label_names = saved["label_names"]
    tf_encoder = saved["tf_encoder"]

    X_all = processor.process_for_inference(tf_encoder)
    preds = np.asarray(model.predict(X_all)).astype(int).ravel()
    unmatched = unmatched.copy()
    unmatched["predicted_label"] = [label_names[p] for p in preds]

    return pd.concat([matched, unmatched]).sort_index()


def evaluate_ensemble_models(
    file_path: str,
    model_path: str,
    output_path: str,
    use_deterministic: bool = False,
) -> None:
    df = pd.read_csv(file_path)
    matched, unmatched = deterministic_decider(df, use_deterministic)

    processor = PreProcessor(unmatched, is_training=True)
    processor()

    saved = joblib.load(model_path)
    model = saved["model"]
    tf_encoder = saved["tf_encoder"]
    label_names = saved["label_names"]

    X = processor.process_for_inference(tf_encoder)
    y = pd.get_dummies(processor.processed_df["gl_code"], dtype=int)
    y_labels = y.values.argmax(axis=1)

    preds = np.asarray(model.predict(X)).astype(int).ravel()

    results = pd.DataFrame(
        {
            "actual_label": [label_names[i] for i in y_labels],
            "predicted_label": [label_names[i] for i in preds],
            "correct": y_labels == preds,
        }
    )

    print(results.to_string())
    print(f"\nAccuracy: {results['correct'].mean():.3f}")
    print(f"F1 score (weighted): {f1_score(y_labels, preds, average='weighted'):.3f}")

    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
