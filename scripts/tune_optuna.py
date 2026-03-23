import warnings

warnings.filterwarnings("ignore")

import logging
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import spacy
from catboost import CatBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight

from float_categorization.constant import LABEL_COL
from float_categorization.modelling.models.ensemble_model import EnsembleClassifier
from float_categorization.pre_processing.pre_processor import PreProcessor
from float_categorization.pre_processing.text_processor import TextProcessor
from float_categorization.runners.decide_deterministic import deterministic_decider
from float_categorization.text_encoder.tf_idf_encoder import TfIdfEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("catboost").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "Copy of Float Sample Data - Sheet5.csv"
TEST_PATH = ROOT / "data" / "test_data.csv"
MODEL_SAVE_DIR = ROOT / "models_dict_latest"

# Best TF-IDF config from grid search
TFIDF_PARAMS = dict(
    max_features=2000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    smooth_idf=True,
    input="content",
    lowercase=False,
    stop_words=None,
)


def prepare_ml_data(file_path):
    """Read CSV, apply deterministic, filter, preprocess ONCE."""
    df = pd.read_csv(file_path)
    matched, unmatched = deterministic_decider(df, use_deterministic=True)

    if not matched.empty:
        det_labels = set(matched["predicted_label"].unique())
        label_col = LABEL_COL.strip().casefold()
        nlp = spacy.load("en_core_web_sm")
        nlp.disable_pipes("ner", "parser")
        raw_to_cleaned = {}
        for label in unmatched[label_col].dropna().unique():
            doc = nlp(label)
            raw_to_cleaned[label] = " ".join(
                t.lemma_
                for t in doc
                if not t.is_stop and not t.is_punct and t.text.strip()
            )
        remove = {r for r, c in raw_to_cleaned.items() if c in det_labels}
        unmatched = unmatched[~unmatched[label_col].isin(remove)]
        print(
            f"Deterministic: {len(det_labels)} labels, ML data: {len(unmatched)} rows"
        )

    # Preprocess (spacy) ONCE
    processor = PreProcessor(unmatched, is_training=True)
    processor()
    proc_df = processor.processed_df
    agg_text = proc_df["aggregated_text"]
    feat_df = proc_df.drop(columns=["gl_code", "aggregated_text"], errors="ignore")
    y_dummies = pd.get_dummies(proc_df["gl_code"], dtype=int)
    y_labels = y_dummies.values.argmax(axis=1)
    label_names = y_dummies.columns.tolist()

    # TF-IDF encode
    vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
    encoded = vectorizer.fit_transform(agg_text)
    X = sp.hstack([sp.csr_matrix(feat_df.values), encoded]).tocsr()

    print(f"X shape: {X.shape}, Classes: {len(label_names)}")
    return X, y_labels, label_names, vectorizer


def compute_weights(y_labels, strategy="balanced", power=1.0):
    unique = np.unique(y_labels)
    if strategy == "none":
        return None
    weights = compute_class_weight("balanced", classes=unique, y=y_labels)
    if strategy == "dampened":
        weights = np.power(weights, power)
    return {int(c): float(w) for c, w in zip(unique, weights)}


def objective(trial, X, y_labels, n_classes):
    n_est = trial.suggest_int("n_estimators", 200, 600, step=50)
    depth = trial.suggest_int("max_depth", 4, 10)
    lr = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    l2 = trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True)
    rs = trial.suggest_float("random_strength", 0.5, 5.0)

    ws = trial.suggest_categorical("weight_strategy", ["balanced", "dampened", "none"])
    wp = trial.suggest_float("weight_power", 0.15, 0.85) if ws == "dampened" else 1.0
    cw = compute_weights(y_labels, strategy=ws, power=wp)

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []
    for ti, vi in skf.split(X, y_labels):
        m = EnsembleClassifier(
            Classifier=CatBoostClassifier,
            num_classes=n_classes,
            class_weights=cw,
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            l2_leaf_reg=l2,
            random_strength=rs,
            random_state=42,
            verbose=0,
        )
        m.fit(X[ti], y_labels[ti])
        p = np.asarray(m.predict(X[vi])).astype(int).ravel()
        scores.append(f1_score(y_labels[vi], p, average="weighted", zero_division=0))
    return np.mean(scores)


def evaluate_on_test(model, label_names, vectorizer):
    """Evaluate final model on test set."""
    test_df = pd.read_csv(TEST_PATH)
    matched, unmatched = deterministic_decider(test_df, use_deterministic=True)

    processor = PreProcessor(unmatched, is_training=True)
    processor()
    proc_df = processor.processed_df
    agg_text = proc_df["aggregated_text"]
    feat_df = proc_df.drop(columns=["gl_code", "aggregated_text"], errors="ignore")
    encoded = vectorizer.transform(agg_text)
    X_test = sp.hstack([sp.csr_matrix(feat_df.values), encoded]).tocsr()

    preds = np.asarray(model.predict(X_test)).astype(int).ravel()
    unmatched = unmatched.copy()
    unmatched["predicted_label"] = [label_names[p] for p in preds]
    result_df = pd.concat([matched, unmatched]).sort_index()

    # Clean actual labels
    test_clean = pd.read_csv(TEST_PATH)
    test_clean.columns = test_clean.columns.str.strip().str.casefold()
    tp = TextProcessor(test_clean)
    actual = tp(["gl code"])["cleaned_gl code"]
    predicted = result_df["predicted_label"]
    idx = actual.index.intersection(predicted.index)
    actual, predicted = actual.loc[idx], predicted.loc[idx]

    acc = accuracy_score(actual, predicted)
    f1w = f1_score(actual, predicted, average="weighted", zero_division=0)
    f1m = f1_score(actual, predicted, average="macro", zero_division=0)
    print(f"\nAccuracy:        {acc:.4f}")
    print(f"F1 (weighted):   {f1w:.4f}")
    print(f"F1 (macro):      {f1m:.4f}")
    print(f"\n{classification_report(actual, predicted, zero_division=0)}")

    out = ROOT / "analysis" / "evaluation_results_tuned.csv"
    pd.DataFrame(
        {
            "actual_label": actual.values,
            "predicted_label": predicted.values,
            "correct": actual.values == predicted.values,
        }
    ).to_csv(out, index=False)
    print(f"Saved to {out}")
    return acc, f1w, f1m


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Preprocess (spacy runs once)")
    print("=" * 60)
    X, y_labels, label_names, vectorizer = prepare_ml_data(TRAIN_PATH)

    print("\n" + "=" * 60)
    print("STEP 2: Optuna tuning (30 trials, 3-fold CV)")
    print("=" * 60)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(lambda t: objective(t, X, y_labels, len(label_names)), n_trials=30)
    print(f"\nBest CV F1: {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("STEP 3: Train final model on all data")
    print("=" * 60)
    bp = dict(study.best_params)
    ws = bp.pop("weight_strategy")
    wp = bp.pop("weight_power", 1.0)
    cw = compute_weights(y_labels, strategy=ws, power=wp)

    final = EnsembleClassifier(
        Classifier=CatBoostClassifier,
        num_classes=len(label_names),
        class_weights=cw,
        random_state=42,
        verbose=50,
        **bp,
    )
    final.fit(X, y_labels)

    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_SAVE_DIR / "ensemble_model.joblib"
    tf_enc = TfIdfEncoder()
    tf_enc.vectorizer = vectorizer
    joblib.dump(
        {"model": final, "label_names": label_names, "tf_encoder": tf_enc},
        str(save_path),
    )
    print(f"Saved to {save_path}")

    print("\n" + "=" * 60)
    print("STEP 4: Evaluate on test data")
    print("=" * 60)
    evaluate_on_test(final, label_names, vectorizer)
