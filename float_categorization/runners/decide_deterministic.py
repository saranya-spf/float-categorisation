import pandas as pd
from float_categorization.deterministic.fuzzy_rule_classifier import (
    FuzzyRuleClassifier,
    ManualFeatureTransformer,
)
from float_categorization.pre_processing.feature_processor import FeatureProcessor


def run_deterministic(df: pd.DataFrame) -> pd.DataFrame:
    manual_df = ManualFeatureTransformer(df)()
    feature_df = FeatureProcessor(df)()
    manual_df["amount_parsed"] = feature_df["amount_parsed"]

    fuzz_classifier = FuzzyRuleClassifier()
    return fuzz_classifier.apply_rules(manual_df)


def deterministic_decider(
    df: pd.DataFrame, use_deterministic: bool = True
) -> pd.DataFrame:
    if use_deterministic:
        result_df = run_deterministic(df)
    else:
        result_df = df.copy()
        result_df["predicted_label"] = None

    matched = result_df[result_df["predicted_label"].notna()]
    unmatched = result_df[result_df["predicted_label"].isna()].drop(
        columns=["predicted_label"]
    )
    return matched, unmatched
