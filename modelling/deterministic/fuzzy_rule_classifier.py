from __future__ import annotations

import re
import fuzzywuzzy as fuzz
from typing import List

import warnings
from pathlib import Path
import pandas as pd

from modelling.pre_processing.text_processor import TextProcessor
from modelling.modelling.constant import TXT_COLS_TEST


class FuzzyRuleClassifier:
    # Maps field -> { label -> [predicates] }.
    # Each field corresponds to a cleaned column (e.g. "transaction detail" -> "cleaned_transaction detail").
    # Rules are evaluated field-by-field, label-by-label; first match wins.
    RULES = {
        "transaction detail": {
            "5615 Facebook Advertising": [
                lambda text: bool(re.search(r"facebk", text, re.IGNORECASE)),
            ],
            "1071 Float USD Account": [
                lambda text: bool(re.search(r"cash cad cash usd", text, re.IGNORECASE)),
            ],
            
        },
        # "payee": {
        #     "SomeLabel": [
        #         lambda text: bool(re.search(r"some_pattern", text, re.IGNORECASE)),
        #     ],
        # },
        "amount": {
            "1060 TD Chequing Account": [
                lambda amt: bool(amt > 0.0 and abs(amt) > 125000.0)
            ]
        }
    }

    def __init__(self, df: pd.DataFrame):
        self.processor = TextProcessor(df)

    def process(self, cols: List[str]) -> pd.DataFrame:
        res = self.processor(cols)
        return res

    def _match_row(self, row: pd.Series) -> str | None:
        for field, label_rules in self.RULES.items():
            col = f"cleaned_{field}"
            if col not in row.index:
                continue
            text = str(row[col])
            for label, predicates in label_rules.items():
                if any(pred(text) for pred in predicates):
                    return label
        return None

    def fit_rules(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Process the given columns, apply rules, and return df with 'predicted_label'."""
        cleaned = self.process(cols)
        merged = df.join(cleaned)
        merged["predicted_label"] = merged.apply(self._match_row, axis=1)
        return merged


warnings.filterwarnings("ignore")
default_path = Path(__file__).parents[2]

if __name__ == "__main__":
    FILE_PATH_1 = default_path / "data" / "Copy of Float Sample Data - Sheet5.csv"
    df = pd.read_csv(FILE_PATH_1)

    fuzz_classifier = FuzzyRuleClassifier(df)
    result = fuzz_classifier.fit_rules(df, cols=[" Transaction Detail "])
    print(result[["predicted_label"]].head(30))
