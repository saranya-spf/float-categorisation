from __future__ import annotations

import re

import warnings
from pathlib import Path
import pandas as pd

from float_categorization.pre_processing.text_processor import TextProcessor


# 5700 office expense                1595       [ML]
# 5773 software license               855       [ML]
# 5450 hr recruiting                  542       [DONE]
# 5615 facebook advertising           539       [DONE]
# 5785 meal entertainment             277       [ML]
# 5772 computer                       269       [DONE]
# 5617 advertising                    227       [DONE]
# account payable sage                195       [ML]
# 5784 travel entertainment           147       [ML]
# 5780 telephone utility               49       [DONE]
# 1212 accrue receivables              35       [DONE]
# 1351 spring mortgage group           35       [DONE]
# 1060 td cheque account               26       [DONE]
# 4245 interest revenue spring         25       [DONE]
# 5610 accounting legal                18       [DONE]
# 1820 office furniture equipment      17       [ML]
# 1320 prepaid expense                 15       [ML]
# 5410 wage salary                     11       [ML]
# 1071 float usd account                6       [DONE]
# 1358 bloom app inc                    4       [ML]
# 5690 interest bank charge             2       [DONE]
# 1359 1478897                          1       [IGNORE]
# 5626 training development             1       [IGNORE]


class ManualFeatureTransformer:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.casefold()

    def __call__(self) -> pd.DataFrame:
        for col in self.df.select_dtypes(include="object").columns:
            self.df[col] = (
                self.df[col]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                .str.replace(r"\s+", " ", regex=True)
            )
        return self.df


class FuzzyRuleClassifier:
    # Maps field -> { label -> [predicates] }.
    # Fields refer to raw column names (after ManualFeatureTransformer cleaning).
    # Rules are evaluated field-by-field, label-by-label; first match wins.
    RULES = {
        "payee": {
            "5615 facebook advertising": [
                lambda text: bool(re.search(r"facebk", text, re.IGNORECASE)),
            ],
            "5610 accounting legal": [
                lambda text: bool(
                    re.search(
                        r"mag-scc online fees"
                        r"|action process serving lt"
                        r"|cfa institute"
                        r"|chartered professional"
                        r"|court kings bench edm"
                        r"|eldor-wal registrations"
                        r"|financial and consumer se"
                        r"|insurance council of mani"
                        r"|learnformula\(cpd\)"
                        r"|prov of bc- bc registr"
                        r"|sq \*notary2u\.ca & docserv"
                        r"|xai llc",
                        text,
                        re.IGNORECASE,
                    )
                ),
            ],
        },
        "transaction detail": {
            "5772 computer": [
                lambda text: bool(
                    re.search(r"hardware.*hardgoods", text, re.IGNORECASE)
                ),
            ],
            "1071 float usd account": [
                lambda text: bool(
                    re.search(r"cash cad to cash usd", text, re.IGNORECASE)
                ),
            ],
            "5450 hr recruiting": [
                lambda text: bool(
                    re.search(
                        r"General HR/ Recruiting - Indeed, BambooHR, CertN & LinkedIN",
                        text,
                        re.IGNORECASE,
                    )
                )
            ],
            "1351 spring mortgage group": [
                lambda text: bool(
                    re.search(
                        r"Card to charge SMG Appraisals from SMG-AR Channel",
                        text,
                        re.IGNORECASE,
                    )
                )
            ],
        },
        "transaction type": {
            "4245 interest revenue spring": [
                lambda text: text in ("cashback", "interest"),
            ],
        },
        "amount_parsed": {
            "1060 td cheque account": [
                lambda amt: bool(amt > 0.0 and abs(amt) >= 125000.0)
            ]
        },
    }

    # Combined-field rules: concatenate listed fields, then apply predicate.
    # Each entry: label -> (fields_to_concat, [predicates])
    COMBINED_FIELD_RULES = {
        "1212 accrue receivables": (
            ["payee", "transaction detail"],
            [
                lambda text: bool(
                    re.search(
                        r"apple\.com/bill",
                        text,
                        re.IGNORECASE,
                    )
                ),
            ],
        ),
        "5780 telephone utility": (
            ["payee", "transaction detail"],
            [
                lambda text: bool(
                    re.search(
                        r"ccsi myfax|freedom mobile|rogers|shaw telecom"
                        r"|internet and telecom",
                        text,
                        re.IGNORECASE,
                    )
                ),
            ],
        ),
        "5617 advertising": (
            ["payee", "transaction detail"],
            [
                lambda text: bool(
                    re.search(
                        r"microsoft\*ads|snap snap ads|tiktok|capcut|midjourney"
                        r"|bitly\.com|canva|varify\.io|cardconexp"
                        r"|infobip|jotform|udemy|quick digit",
                        text,
                        re.IGNORECASE,
                    )
                ),
            ],
        ),
    }

    # Cross-field rules: ALL conditions must be true (AND logic).
    # Each entry: label -> [(field, predicate), ...]
    MULTI_FIELD_RULES = {
        "5690 interest bank charge": [
            ("transaction type", lambda text: text == "float fee"),
            ("transaction detail", lambda text: text == "float billing & usage fees"),
        ],
    }

    def _match_row(self, row: pd.Series) -> str | None:
        # Check multi-field AND rules first
        for label, conditions in self.MULTI_FIELD_RULES.items():
            if all(
                field in row.index and pred(row[field]) for field, pred in conditions
            ):
                return label

        # Combined-field rules: concat fields then match
        for label, (fields, predicates) in self.COMBINED_FIELD_RULES.items():
            combined = " ".join(str(row[f]) for f in fields if f in row.index)
            if any(pred(combined) for pred in predicates):
                return label

        # Then single-field OR rules
        for field, label_rules in self.RULES.items():
            if field not in row.index:
                continue
            value = row[field]
            for label, predicates in label_rules.items():
                if any(pred(value) for pred in predicates):
                    return label
        return None

    def apply_rules(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply rules on a DataFrame with manually cleaned columns."""
        df["predicted_label"] = df.apply(self._match_row, axis=1)
        return df


warnings.filterwarnings("ignore")
default_path = Path(__file__).parents[2]


def print_label_distribution(file_path: str):
    """Print value_counts of GL Code after TextProcessor cleaning."""
    df = pd.read_csv(file_path)
    processor = TextProcessor(df)
    cleaned = processor(["gl code"])
    print(cleaned["cleaned_gl code"].value_counts().to_string())


# if __name__ == "__main__":
#     FILE_PATH_1 = default_path / "data" / "Copy of Float Sample Data - Sheet5.csv"
#     df = pd.read_csv(FILE_PATH_1)

#     # Simple cleaning for rule input features (strip, lowercase, collapse spaces)
#     manual_transformer = ManualFeatureTransformer(df)
#     manual_df = manual_transformer()

#     # Process numeric features (parses amount, handles nulls, etc.)
#     feature_processor = FeatureProcessor(df)
#     feature_df = feature_processor()

#     # Merge manual-cleaned text cols + feature-processed numeric cols
#     # Keep manual text cols for rules, amount_parsed from feature_df
#     manual_df["amount_parsed"] = feature_df["amount_parsed"]

#     # Clean GL Code only via TextProcessor (for label normalization)
#     label_processor = TextProcessor(df)
#     label_cleaned = label_processor([" GL Code "])
#     manual_df["cleaned_gl_code"] = label_cleaned["cleaned_gl code"].values

#     # Apply deterministic rules
#     fuzz_classifier = FuzzyRuleClassifier()
#     result = fuzz_classifier.apply_rules(manual_df)

#     actual_col = "cleaned_gl_code"
#     pred_col = "predicted_label"

#     total = len(result)
#     predicted_mask = result[pred_col].notna()
#     unpredicted_mask = ~predicted_mask
#     n_predicted = predicted_mask.sum()
#     n_unpredicted = unpredicted_mask.sum()
#     all_classes = set(result[actual_col].dropna().unique())
#     predicted_classes = set(result.loc[predicted_mask, pred_col].unique())
#     classes_with_coverage = all_classes & predicted_classes
#     classes_without_coverage = all_classes - predicted_classes

#     print(f"\n{'=' * 50}")
#     print(f"COVERAGE REPORT")
#     print(f"{'=' * 50}")
#     print(f"Total samples:             {total}")
#     print(
#         f"Predicted:                 {n_predicted} ({n_predicted / total * 100:.1f}%)"
#     )
#     print(
#         f"Left to predict:           {n_unpredicted} ({n_unpredicted / total * 100:.1f}%)"
#     )
#     print(f"\nTotal classes:             {len(all_classes)}")
#     print(f"Classes with rules:        {len(classes_with_coverage)}")
#     print(f"Classes without rules:     {len(classes_without_coverage)}")
#     if classes_without_coverage:
#         for c in sorted(classes_without_coverage):
#             print(f"  - {c}")

#     # Accuracy metrics
#     if n_predicted > 0:
#         from sklearn.metrics import accuracy_score, f1_score, classification_report

#         pred_df = result[predicted_mask]
#         y_true_pred = pred_df[actual_col]
#         y_pred_pred = pred_df[pred_col]

#         acc_pred = accuracy_score(y_true_pred, y_pred_pred)
#         f1_w_pred = f1_score(
#             y_true_pred, y_pred_pred, average="weighted", zero_division=0
#         )
#         f1_m_pred = f1_score(y_true_pred, y_pred_pred, average="macro", zero_division=0)

#         print(f"\n{'=' * 50}")
#         print(f"ACCURACY — predicted only ({n_predicted} samples)")
#         print(f"{'=' * 50}")
#         print(f"Accuracy:                  {acc_pred:.4f}")
#         print(f"F1 (weighted):             {f1_w_pred:.4f}")
#         print(f"F1 (macro):                {f1_m_pred:.4f}")
#         print(f"\n{classification_report(y_true_pred, y_pred_pred, zero_division=0)}")

#         # Overall: treat unpredicted as wrong (label = None)
#         y_true_all = result[actual_col]
#         y_pred_all = result[pred_col].fillna("__unpredicted__")

#         acc_all = accuracy_score(y_true_all, y_pred_all)
#         f1_w_all = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
#         f1_m_all = f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)

#         print(f"{'=' * 50}")
#         print(f"ACCURACY — overall ({total} samples, unpredicted = wrong)")
#         print(f"{'=' * 50}")
#         print(f"Accuracy:                  {acc_all:.4f}")
#         print(f"F1 (weighted):             {f1_w_all:.4f}")
#         print(f"F1 (macro):                {f1_m_all:.4f}")
#     else:
#         print("\nNo predictions made — skipping accuracy metrics.")
