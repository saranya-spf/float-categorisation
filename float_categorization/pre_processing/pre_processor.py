import pandas as pd

from float_categorization.constant import TXT_COLS_TRAIN, TXT_COLS_TEST, LABEL_COL
from float_categorization.pre_processing.text_processor import TextProcessor
from float_categorization.pre_processing.feature_processor import FeatureProcessor


class PreProcessor:
    def __init__(self, df: pd.DataFrame, is_training: bool = True):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.casefold()
        self.is_training = is_training

    def __call__(self):
        feature_processor = FeatureProcessor(self.df)
        feature_df = feature_processor()

        txt_cols = TXT_COLS_TRAIN if self.is_training else TXT_COLS_TEST
        text_processor = TextProcessor(self.df)

        #   Process feature text columns
        text_df = text_processor(txt_cols)

        #   Process label column separately (only during training)
        label_series = None
        if self.is_training and LABEL_COL in self.df.columns:
            label_df = text_processor([LABEL_COL])
            label_series = label_df[f"cleaned_{LABEL_COL}"]

        feature_df.drop(columns=txt_cols + [LABEL_COL], inplace=True, errors="ignore")
        processed_df_deterministic = pd.concat([feature_df, text_df], axis=1)

        cleaned_cols = [c for c in text_df.columns]
        processed_df_deterministic["aggregated_text"] = (
            text_df.agg(" ".join, axis=1).str.split().str.join(" ")
        )
        processed_df = processed_df_deterministic.drop(columns=cleaned_cols)

        if label_series is not None:
            processed_df["label"] = label_series
            processed_df_deterministic["label"] = label_series

        return processed_df, processed_df_deterministic


if __name__ == "__main__":
    df = pd.read_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
    )
    processor = PreProcessor(df, is_training=True)
    processed_df, deterministic_df = processor()
    processed_df.to_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/model_processed.csv"
    )
    deterministic_df.to_csv(
        "/Users/saranya.pal/Desktop/Projects/float_categorization/data/deterministic_processed.csv"
    )
