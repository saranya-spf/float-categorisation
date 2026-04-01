"""Run the full pipeline WITHOUT TF-IDF features (text ablation study)."""

import warnings
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

from float_categorization.core import run_pipeline, run_inference
from float_categorization.pre_processing.text_processor import TextProcessor
from float_categorization.runners.decide_deterministic import deterministic_decider

warnings.filterwarnings("ignore")
default_path = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    TRAIN_PATH = default_path / "data" / "Copy of Float Sample Data - Sheet5.csv"
    TEST_PATH = default_path / "data" / "test_data.csv"
    MODEL_SAVE_DIR = default_path / "models_dict_latest"
    MODEL_PATH = MODEL_SAVE_DIR / "ensemble_model.joblib"

    print("=" * 80)
    print("ABLATION: Training CatBoost model WITHOUT TF-IDF (manual features only)")
    print("=" * 80)
    run_pipeline(
        file_path=TRAIN_PATH,
        use_deterministic=True,
        save_model=True,
        save_dir=str(MODEL_SAVE_DIR),
        use_ensemble=True,
        test_size=0.01,
        Classifier=CatBoostClassifier,
        use_class_weights=False,
        skip_tfidf=True,
        n_estimators=500,
        max_depth=5,
        learning_rate=0.1931240061832569,
        l2_leaf_reg=1.7226169323450076,
        random_strength=3.5471847376618655,
        random_state=42,
    )

    # Step 2: Run inference on held-out test data
    print("\n" + "=" * 80)
    print("STEP 2: Running inference on test data (NO TF-IDF)")
    print("=" * 80)
    result_df = run_inference(
        file_path=str(TEST_PATH),
        model_dict_path=str(MODEL_PATH),
        use_deterministic=True,
        use_ensemble=True,
    )

    # Step 3: Evaluate
    print("\n" + "=" * 80)
    print("STEP 3: Evaluation on Test Data (NO TF-IDF)")
    print("=" * 80)

    test_df = pd.read_csv(TEST_PATH)
    test_df.columns = test_df.columns.str.strip().str.casefold()
    text_proc = TextProcessor(test_df)
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
    print("\n--- Per-GL-Code Classification Report (Precision / Recall / F1) ---\n")
    print(classification_report(actual, predicted, zero_division=0))

    det_matched, det_unmatched = deterministic_decider(
        pd.read_csv(TEST_PATH), use_deterministic=True
    )
    deterministic_indices = set(det_matched.index)
    full_output = test_df.loc[common_idx].copy()
    full_output["predicted_label"] = predicted.values
    full_output["correct"] = actual.values == predicted.values
    full_output["prediction_source"] = [
        "deterministic" if i in deterministic_indices else "ml_model"
        for i in common_idx
    ]

    for src in ["deterministic", "ml_model"]:
        mask = full_output["prediction_source"] == src
        n = mask.sum()
        n_correct = (full_output.loc[mask, "correct"]).sum()
        print(
            f"  {src}: {n_correct}/{n} correct ({n_correct / n:.2%})"
            if n
            else f"  {src}: 0 rows"
        )
