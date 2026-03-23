from typing import List

import pandas as pd


class FeatureProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.casefold()

    def __call__(self, *args, **kwds):
        cols_to_drop = ["date", "funding account", "currency"]
        self.drop_columns(cols_to_drop)

        optional_cols_to_drop = ["spender", "approver", "payment authorizer"]
        self.drop_columns(optional_cols_to_drop)

        self.one_hot_encode_null_values()
        self.process_further()

        return self.df

    def drop_columns(self, cols_to_drop: List[str]):
        self.df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    def one_hot_encode_null_values(self):
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
