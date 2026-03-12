from typing import Optional, List

import pandas as pd
import spacy


class TextProcessor:
    def __init__(self, df: pd.DataFrame, cols_to_keep: Optional[List] = None):
        self.df = df
        self.df.columns = self.df.columns.str.strip().str.casefold()
        print("Loading spacy text process model...")
        self.nlp = spacy.load("en_core_web_sm")
        print("Spacy model loaded successfully")

        if cols_to_keep is None:
            cols_to_keep = [
                "transaction type",
                "payee",
                "transaction detail",
                "gl code",
            ]
        self.df = self.df[cols_to_keep]

    
    def process_text(self) -> pd.DataFrame:
        text_cols = [col for col in self.df.columns if col != "gl code"]
        for col in text_cols:
            self.df[f"cleaned_{col}"] = self.df[col].astype(str).apply(self.clean_text)

        cleaned_cols = [f"cleaned_{col}" for col in text_cols]
        self.df["aggregated_text"] = self.df[cleaned_cols].agg(" ".join, axis=1)
        return self.df


    def clean_text(self, text: str | None) -> str:
        if not text or pd.isna(text):
            return ""
        doc = self.nlp(text.strip().lower())
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return " ".join(tokens)
