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

    MULTI_FIELD_RULES = {
        "5690 interest bank charge": [
            ("transaction type", lambda text: text == "float fee"),
            ("transaction detail", lambda text: text == "float billing & usage fees"),
        ],
    }

    def _match_row(self, row: pd.Series) -> str | None:
        for label, conditions in self.MULTI_FIELD_RULES.items():
            if all(
                field in row.index and pred(row[field]) for field, pred in conditions
            ):
                return label

        for label, (fields, predicates) in self.COMBINED_FIELD_RULES.items():
            combined = " ".join(str(row[f]) for f in fields if f in row.index)
            if any(pred(combined) for pred in predicates):
                return label

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
