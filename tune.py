"""
Optuna hyperparameter optimization for the float categorization pipeline.

Tunes:
  - CatBoost hyperparameters
  - TF-IDF hyperparameters
  - Class weight smoothing factor
  - Whether to use deterministic rules

Uses stratified 85/15 train/val split for optimization,
then retrains best config on full training data and evaluates on held-out test set.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import optuna
import scipy.sparse as sp
from pathlib import Path
from math import ceil
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight

from float_categorization.modelling.models.ensemble_model import EnsembleClassifier
from float_categorization.pre_processing.pre_processor import PreProcessor
from float_categorization.pre_processing.text_processor import TextProcessor
from float_categorization.runners.decide_deterministic import deterministic_decider
from float_categorization.constant import LABEL_COL

optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "Copy of Float Sample Data - Sheet5.csv"
TEST_PATH = ROOT / "data" / "test_data.csv"
MODEL_SAVE_DIR = ROOT / "models_dict_latest"


# ─────────────────── helpers ────────────────────────────────────────


def filter_deterministic_labels(matched, unmatched, use_deterministic):
    """Remove rows whose label belongs to a deterministic-covered category."""
    if not use_deterministic or matched.empty:
        return unmatched
    import spacy

    deterministic_labels = set(matched["predicted_label"].unique())
    label_col = LABEL_COL.strip().casefold()
    nlp = spacy.load("en_core_web_sm")
    nlp.disable_pipes("ner", "parser")
    unique_labels = unmatched[label_col].dropna().unique().tolist()
    raw_to_cleaned = {}
    for label in unique_labels:
        doc = nlp(label)
        cleaned = " ".join(
            t.lemma_ for t in doc if not t.is_stop and not t.is_punct and t.text.strip()
        )
        raw_to_cleaned[label] = cleaned
    raw_labels_to_remove = {
        raw
        for raw, cleaned in raw_to_cleaned.items()
        if cleaned in deterministic_labels
    }
    return unmatched[~unmatched[label_col].isin(raw_labels_to_remove)]


def preprocess_and_encode(df, tfidf_params, fit=True, tf_encoder=None):
    """Run preprocessing and TF-IDF encoding. Returns (X, y_dummies, label_names, tf_encoder)."""
    processor = PreProcessor(df, is_training=True)
    processor()

    X_initial, y_series = (
        processor.processed_df.drop("gl_code", axis=1),
        processor.processed_df["gl_code"],
    )

    vectorizer = TfidfVectorizer(
        input="content",
        lowercase=False,
        stop_words=tfidf_params.get("stop_words"),
        ngram_range=tuple(tfidf_params["ngram_range"]),
        max_df=tfidf_params["max_df"],
        min_df=tfidf_params["min_df"],
        max_features=tfidf_params["max_features"],
        sublinear_tf=tfidf_params["sublinear_tf"],
        norm=tfidf_params["norm"],
        use_idf=True,
        smooth_idf=True,
        analyzer=tfidf_params.get("analyzer", "word"),
    )

    if fit:
        encoded = vectorizer.fit_transform(X_initial["aggregated_text"])
    else:
        encoded = tf_encoder.transform(X_initial["aggregated_text"])

    X_initial_no_text = X_initial.drop("aggregated_text", axis=1)
    X = sp.hstack([sp.csr_matrix(X_initial_no_text.values), encoded]).tocsr()
    y = pd.get_dummies(y_series, dtype=int)

    if fit:
        return X, y, y.columns.tolist(), vectorizer
    return X, y, y.columns.tolist(), tf_encoder


def split_with_rare(X, y, label_names, test_size=0.15):
    """Stratified split that puts rare classes only in train."""
    y_labels = y.values.argmax(axis=1)
    class_counts = np.bincount(y_labels, minlength=len(label_names))
    min_for_split = max(2, int(ceil(1.0 / test_size)))
    splittable_mask = np.array([class_counts[l] >= min_for_split for l in y_labels])
    rare_mask = ~splittable_mask

    X_rare, y_rare = X[rare_mask], y[rare_mask]
    X_split, y_split = X[splittable_mask], y[splittable_mask]
    y_split_labels = y_split.values.argmax(axis=1)

    X_train_s, X_val, y_train_s, y_val = train_test_split(
        X_split, y_split, test_size=test_size, stratify=y_split_labels, random_state=42
    )
    X_train = sp.vstack([X_train_s, X_rare]) if rare_mask.any() else X_train_s
    y_train = pd.concat([y_train_s, y_rare], axis=0) if rare_mask.any() else y_train_s
    return X_train, X_val, y_train, y_val


def compute_smoothed_weights(y_train, alpha=1.0):
    """Compute class weights with smoothing. alpha=1 → balanced, alpha=0 → uniform."""
    y_labels = y_train.values.argmax(axis=1)
    unique = np.unique(y_labels)
    balanced = compute_class_weight("balanced", classes=unique, y=y_labels)
    balanced_dict = {int(c): float(w) for c, w in zip(unique, balanced)}

    if alpha >= 1.0:
        return balanced_dict

    smoothed = {}
    for c, w in balanced_dict.items():
        smoothed[c] = (1.0 - alpha) * 1.0 + alpha * w
    return smoothed


# ─────────────────── Optuna objective ───────────────────────────────

# Pre-load and preprocess data outside objective for efficiency
print("Loading and preprocessing training data...")
train_df_raw = pd.read_csv(TRAIN_PATH)
matched_train, unmatched_train = deterministic_decider(
    train_df_raw, use_deterministic=True
)
unmatched_train = filter_deterministic_labels(
    matched_train, unmatched_train, use_deterministic=True
)
print(f"Training data: {len(unmatched_train)} rows after deterministic filtering")


def objective(trial):
    # ── TF-IDF hyperparameters ──
    tfidf_params = {
        "ngram_range": [1, trial.suggest_int("tfidf_ngram_max", 1, 4)],
        "max_df": trial.suggest_float("tfidf_max_df", 0.8, 1.0),
        "min_df": trial.suggest_int("tfidf_min_df", 1, 5),
        "max_features": trial.suggest_int("tfidf_max_features", 500, 5000, step=250),
        "sublinear_tf": trial.suggest_categorical("tfidf_sublinear_tf", [True, False]),
        "norm": "l2",
        "stop_words": None,
        "analyzer": trial.suggest_categorical("tfidf_analyzer", ["word", "char_wb"]),
    }

    # ── Preprocess ──
    X, y, label_names, tf_enc = preprocess_and_encode(
        unmatched_train, tfidf_params, fit=True
    )
    X_train, X_val, y_train, y_val = split_with_rare(X, y, label_names, test_size=0.15)

    # ── Class weights ──
    weight_alpha = trial.suggest_float("weight_alpha", 0.0, 1.0)
    class_weights = compute_smoothed_weights(y_train, alpha=weight_alpha)

    # ── CatBoost hyperparameters ──
    cb_params = {
        "n_estimators": trial.suggest_int("cb_n_estimators", 200, 1000, step=50),
        "max_depth": trial.suggest_int("cb_max_depth", 4, 10),
        "learning_rate": trial.suggest_float("cb_learning_rate", 0.01, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("cb_l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("cb_random_strength", 0.0, 5.0),
        "bagging_temperature": trial.suggest_float("cb_bagging_temp", 0.0, 3.0),
        "random_state": 42,
    }

    # ── Train ──
    y_train_labels = y_train.values.argmax(axis=1)
    y_val_labels = y_val.values.argmax(axis=1)

    model = EnsembleClassifier(
        Classifier=CatBoostClassifier,
        num_classes=y_train.shape[1],
        class_weights=class_weights,
        verbose=0,
        **cb_params,
    )
    model.fit(X_train, y_train_labels)

    preds = np.asarray(model.predict(X_val)).astype(int).ravel()
    f1_w = f1_score(y_val_labels, preds, average="weighted", zero_division=0)
    f1_m = f1_score(y_val_labels, preds, average="macro", zero_division=0)

    # Optimize a combination: mostly weighted F1 but also macro
    return 0.6 * f1_w + 0.4 * f1_m


# ─────────────────── Run optimization ───────────────────────────────

if __name__ == "__main__":
    N_TRIALS = 80
    print(f"\nStarting Optuna optimization with {N_TRIALS} trials...")

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print(f"\n{'=' * 70}")
    print(f"BEST TRIAL: #{study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"{'=' * 70}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # ── Retrain with best params on full training data ──────────────
    print(f"\n{'=' * 70}")
    print("Retraining with best params on FULL training data...")
    print(f"{'=' * 70}")
    bp = study.best_params

    best_tfidf = {
        "ngram_range": [1, bp["tfidf_ngram_max"]],
        "max_df": bp["tfidf_max_df"],
        "min_df": bp["tfidf_min_df"],
        "max_features": bp["tfidf_max_features"],
        "sublinear_tf": bp["tfidf_sublinear_tf"],
        "norm": "l2",
        "stop_words": None,
        "analyzer": bp["tfidf_analyzer"],
    }

    # Full data encode
    X_full, y_full, label_names, tf_encoder = preprocess_and_encode(
        unmatched_train, best_tfidf, fit=True
    )

    # Class weights on full data
    y_full_labels = y_full.values.argmax(axis=1)
    unique_cls = np.unique(y_full_labels)
    balanced_w = compute_class_weight("balanced", classes=unique_cls, y=y_full_labels)
    balanced_dict = {int(c): float(w) for c, w in zip(unique_cls, balanced_w)}
    alpha = bp["weight_alpha"]
    final_weights = {
        c: (1.0 - alpha) * 1.0 + alpha * w for c, w in balanced_dict.items()
    }
    print(
        f"Class weight range: {min(final_weights.values()):.3f} - {max(final_weights.values()):.3f}"
    )

    best_cb = {
        "n_estimators": bp["cb_n_estimators"],
        "max_depth": bp["cb_max_depth"],
        "learning_rate": bp["cb_learning_rate"],
        "l2_leaf_reg": bp["cb_l2_leaf_reg"],
        "random_strength": bp["cb_random_strength"],
        "bagging_temperature": bp["cb_bagging_temp"],
        "random_state": 42,
    }

    model = EnsembleClassifier(
        Classifier=CatBoostClassifier,
        num_classes=y_full.shape[1],
        class_weights=final_weights,
        verbose=0,
        **best_cb,
    )
    model.fit(X_full, y_full_labels)

    # Save model
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_SAVE_DIR / "ensemble_model.joblib"
    joblib.dump(
        {"model": model, "label_names": label_names, "tf_encoder": tf_encoder},
        str(save_path),
    )
    print(f"Model saved to {save_path}")

    # ── Evaluate on test data ───────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Evaluating on TEST data...")
    print(f"{'=' * 70}")

    test_df_raw = pd.read_csv(TEST_PATH)
    matched_test, unmatched_test = deterministic_decider(
        test_df_raw, use_deterministic=True
    )

    # Preprocess unmatched test data with fitted tf_encoder
    test_processor = PreProcessor(unmatched_test, is_training=True)
    test_processor()
    X_test_initial = test_processor.processed_df.drop(
        "gl_code", axis=1, errors="ignore"
    )
    X_test_encoded = tf_encoder.transform(X_test_initial["aggregated_text"])
    X_test_initial_no_text = X_test_initial.drop("aggregated_text", axis=1)
    X_test = sp.hstack(
        [sp.csr_matrix(X_test_initial_no_text.values), X_test_encoded]
    ).tocsr()

    preds_test = np.asarray(model.predict(X_test)).astype(int).ravel()
    unmatched_test_copy = unmatched_test.copy()
    unmatched_test_copy["predicted_label"] = [label_names[p] for p in preds_test]

    result_df = pd.concat([matched_test, unmatched_test_copy]).sort_index()

    # Clean actual labels same way as predictions
    test_df_clean = pd.read_csv(TEST_PATH)
    test_df_clean.columns = test_df_clean.columns.str.strip().str.casefold()
    text_proc = TextProcessor(test_df_clean)
    cleaned = text_proc(["gl code"])
    actual_labels = cleaned["cleaned_gl code"]
    predicted_labels = result_df["predicted_label"]

    common_idx = actual_labels.index.intersection(predicted_labels.index)
    actual = actual_labels.loc[common_idx]
    predicted = predicted_labels.loc[common_idx]

    accuracy = accuracy_score(actual, predicted)
    f1_w = f1_score(actual, predicted, average="weighted", zero_division=0)
    f1_m = f1_score(actual, predicted, average="macro", zero_division=0)

    print(f"\nTest Accuracy:        {accuracy:.4f}")
    print(f"Test F1 (weighted):   {f1_w:.4f}")
    print(f"Test F1 (macro):      {f1_m:.4f}")
    print("\n--- Per-GL-Code Classification Report ---\n")
    print(classification_report(actual, predicted, zero_division=0))

    # Save results
    eval_results = pd.DataFrame(
        {
            "actual_label": actual.values,
            "predicted_label": predicted.values,
            "correct": actual.values == predicted.values,
        }
    )
    output_path = ROOT / "analysis" / "evaluation_results_test.csv"
    eval_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")

    # Save best params
    import json

    params_path = MODEL_SAVE_DIR / "best_params.json"
    with open(params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"Best params saved to {params_path}")
