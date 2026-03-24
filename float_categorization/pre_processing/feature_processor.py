import re
from typing import List

import pandas as pd

from float_categorization.constant import NAME_TO_ID, NAME_UNKNOWN_ID


class FeatureProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.casefold()

    def __call__(self, *args, **kwds):
        cols_to_drop = ["date", "funding account", "currency"]
        self.drop_columns(cols_to_drop)

        name_based_features = ["spender", "approver", "payment authorizer"]
        # self.drop_columns(name_based_features)
        self.encode_name_based_fields(name_based_features)

        self.one_hot_encode_only_for_null_values()
        self.process_further()

        return self.df

    def drop_columns(self, cols_to_drop: List[str]):
        self.df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    def one_hot_encode_only_for_null_values(self):
        cols_to_one_hot = [
            "account",
            "initial amount",
            "initial currency",
            "last 4 digits",
            "invoice number",
            "fx initial currency",
            "fx initial amount",
            "conversion rate",
        ]

        for col in cols_to_one_hot:
            if col in self.df.columns:
                self.df[col + "_is_null"] = self.df[col].isnull().astype(int)

        self.drop_columns(cols_to_one_hot)

    def one_hot_encode(self, cols: List[str]):
        dummies = pd.get_dummies(
            self.df[cols], columns=cols, drop_first=False, dtype=int
        )
        self.df = pd.concat([self.df, dummies], axis=1)

    @staticmethod
    def _normalize_name(name) -> str:
        """Lowercase, collapse whitespace, keep only first and last name."""
        if pd.isna(name):
            return ""
        name = str(name).strip().casefold()
        name = re.sub(r"\s+", " ", name)
        parts = name.split()
        if len(parts) > 2:
            # Keep first + last to unify "hector alfonso pertierra marin" style
            # but check the full name first in the dictionary
            return name  # return full; lookup will try full then first+last
        return name

    @staticmethod
    def _name_to_id(name: str) -> int:
        """Map a single normalized name string to its integer ID."""
        if not name:
            return 0
        if name in NAME_TO_ID:
            return NAME_TO_ID[name]
        # Try first + last only (handles middle-name mismatch)
        parts = name.split()
        if len(parts) > 2:
            short = f"{parts[0]} {parts[-1]}"
            if short in NAME_TO_ID:
                return NAME_TO_ID[short]
        return NAME_UNKNOWN_ID

    def encode_name_based_fields(self, cols: List[str]):
        # First pass: normalize and build a per-row cache so the same person
        # appearing in multiple columns on the same row gets the same ID.
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._normalize_name)

        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self._name_to_id)

        # Ensure integer dtype
        for col in cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(int)

    def process_further(self):
        def parse_amount(val):
            val = str(val).strip()
            if not val or val.lower() == "nan":
                return 0.0
            is_negative = val.startswith("(") and val.endswith(")")
            val = val.strip("()")
            val = val.replace(",", "")
            result = float(val)
            return -result if is_negative else result

        self.df["amount_parsed"] = self.df["amount"].apply(parse_amount)
        self.df.drop("amount", axis=1, inplace=True)

        self.fil_null_values_numer()
        self.fill_null_values_text(col_names=["transaction type"])

    def fil_null_values_numer(self, fill_with: float = 0.0):
        col_names = ["initial amount", "fx initial amount", "conversion rate"]
        for col in col_names:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(fill_with)

    def fill_null_values_text(self, col_names: List[str], fill_with: str = "unknown"):
        for col in col_names:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(fill_with)

        self.one_hot_encode(col_names)
