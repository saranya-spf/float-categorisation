import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# from float_categorization.deterministic.fuzzy_rule_classifier import FuzzyRuleClassifier
from float_categorization.pre_processing.text_processor import TextProcessor
from float_categorization.modelling.bert_trainer import BertTrainer
from float_categorization.modelling.models.bert_model import BERTClassifier


# def run_deterministic(df: pd.DataFrame) -> pd.DataFrame:
#     fuzz_classifier = FuzzyRuleClassifier(df)
#     return fuzz_classifier.fit_rules(df, cols=[" Transaction Detail "])


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


def train_ml_model(X_train, X_test, y_train, y_test) -> BertTrainer:
    model = BERTClassifier(fine_tune=True, num_classes=y_train.shape[1])
    trainer = BertTrainer(
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


def run_bert_pipeline(
    file_path: str,
    use_deterministic: bool = False,
    save_model: bool = True,
    save_dir: str = "",
) -> None:
    df = pd.read_csv(file_path)

    # if use_deterministic:
    #     result_df = run_deterministic(df)
    # else:
    result_df = df.copy()
    result_df["predicted_label"] = pd.NA

    X_train, X_test, y_train, y_test, unmatched_idx, label_names = prepare_ml_data(
        result_df
    )
    model = BERTClassifier(fine_tune=True, num_classes=y_train.shape[1])
    trainer = BertTrainer(
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

    if save_model:
        save_path = save_dir if save_dir else trainer.BEST_MODEL_PATH
        trainer.save_model(model, path=save_path)
        print(f"Model saved to {save_path}")


#   To be used for data where there is no label present
#   Purely for inference purposes (prediction of label)

def inference_bert_models(
    file_path: str,
    model_dict_path: str,
    use_deterministic: bool = False,
) -> pd.DataFrame:
    df = pd.read_csv(file_path)

    # if use_deterministic:
    #     result_df = run_deterministic(df)
    # else:
    result_df = df.copy()
    result_df["predicted_label"] = pd.NA

    unmatched = result_df[result_df["predicted_label"].isna()]
    text_processor = TextProcessor(unmatched)
    X, y = text_processor.process_for_training()
    label_names = y.columns.tolist()

    state_dict = torch.load(model_dict_path, map_location="cpu")
    num_classes = state_dict["classifier.2.bias"].shape[0]

    model = BERTClassifier(fine_tune=True, num_classes=num_classes)
    trainer = BertTrainer(X_train=X, y_train=y)

    unmatched_X = result_df.loc[unmatched.index, "aggregated_text"]
    _, preds = trainer.predict(model, model_dict_path, unmatched_X)
    result_df.loc[unmatched.index, "predicted_label"] = [label_names[p] for p in preds]

    return result_df


#   To be used when you want to test the model and
#   calculate metrics such as accuracy etc. for that model
#   Thus, using "evaluate" requires the labels to be present
def evaluate_bert_models(
    file_path: str,
    model_dict_path: str,
    output_path: str,
    use_deterministic: bool = False,
) -> None:
    df = pd.read_csv(file_path)

    # if use_deterministic:
    #     result_df = run_deterministic(df)
    # else:
    result_df = df.copy()
    result_df["predicted_label"] = pd.NA

    X_train, X_test, y_train, y_test, unmatched_idx, label_names = prepare_ml_data(
        result_df
    )

    state_dict = torch.load(model_dict_path, map_location="cpu")
    num_classes = state_dict["classifier.2.bias"].shape[0]

    model = BERTClassifier(fine_tune=True, num_classes=num_classes)
    trainer = BertTrainer(X_train=X_train, y_train=y_train)

    _, all_preds = trainer.predict(model, model_dict_path, X_test)

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

    results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


def fine_tune_model(file_path: str, n_trials: int = 20):
    df = pd.read_csv(file_path)
    return BertTrainer.tune(df, n_trials=n_trials)