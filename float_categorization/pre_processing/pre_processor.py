from typing import Tuple, List

import pandas as pd
import scipy.sparse as sp

from float_categorization.constant import TXT_COLS_TRAIN, TXT_COLS_TEST, LABEL_COL
from float_categorization.pre_processing.text_processor import TextProcessor
from float_categorization.pre_processing.feature_processor import FeatureProcessor
from float_categorization.text_encoder.tf_idf_encoder import TfIdfEncoder


class PreProcessor:
    def __init__(self, df: pd.DataFrame, is_training: bool = True):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.casefold()
        self.is_training = is_training
        self.tf_encoder = TfIdfEncoder()

    def __call__(self):
        feature_processor = FeatureProcessor(self.df)
        feature_df = feature_processor()

        txt_cols = TXT_COLS_TRAIN if self.is_training else TXT_COLS_TEST
        text_processor = TextProcessor(self.df)
        text_df = text_processor(txt_cols)

        label_series = None
        if self.is_training and LABEL_COL in self.df.columns:
            label_df = text_processor([LABEL_COL])
            label_series = label_df[f"cleaned_{LABEL_COL}"]

        feature_df.drop(columns=txt_cols + [LABEL_COL], inplace=True, errors="ignore")
        self.processed_df_deterministic = pd.concat([feature_df, text_df], axis=1)

        cleaned_cols = [c for c in text_df.columns]
        self.processed_df_deterministic["aggregated_text"] = (
            text_df.agg(" ".join, axis=1).str.split().str.join(" ")
        )
        self.processed_df = self.processed_df_deterministic.drop(columns=cleaned_cols)

        if label_series is not None:
            self.processed_df["gl_code"] = label_series
            self.processed_df_deterministic["gl_code"] = label_series

        return self.processed_df, self.processed_df_deterministic

    def process_for_training(self) -> Tuple[sp.csr_matrix, pd.DataFrame]:
        self.X_initial, self.y = (
            self.processed_df.drop("gl_code", axis=1),
            self.processed_df["gl_code"],
        )

        self.encoded = self.tf_encoder.fit_transform(self.X_initial["aggregated_text"])
        self.X_initial.drop("aggregated_text", axis=1, inplace=True)
        self.X_combined = sp.hstack(
            [sp.csr_matrix(self.X_initial.values), self.encoded]
        ).tocsr()

        return self.X_combined, pd.get_dummies(self.y, dtype=int)

    def process_for_inference(self, tf_encoder: TfIdfEncoder) -> sp.csr_matrix:
        self.X_initial = self.processed_df.drop("gl_code", axis=1, errors="ignore")
        encoded = tf_encoder.transform(self.X_initial["aggregated_text"])
        self.X_initial.drop("aggregated_text", axis=1, inplace=True)
        self.X_combined = sp.hstack(
            [sp.csr_matrix(self.X_initial.values), encoded]
        ).tocsr()
        return self.X_combined


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
    )
    processor = PreProcessor(df, is_training=True)
    processed_df, deterministic_df = processor()
    processed_df.to_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/model_processed.csv",
        index=False,
    )
    deterministic_df.to_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/deterministic_processed.csv",
        index=False,
    )
