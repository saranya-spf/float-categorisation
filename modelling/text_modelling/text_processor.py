from typing import Optional, List, Tuple

import pandas as pd
import spacy


COLS_TO_KEEP = [
    "transaction type",
    "payee",
    "transaction detail",
    "gl code",
]


class TextProcessor:
    def __init__(self, df: pd.DataFrame, cols_to_keep: Optional[List] = None):
        self.df = df
        self.df.columns = self.df.columns.str.strip().str.casefold()
        print("Loading spacy text processing model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes("ner", "parser")

        print("Spacy model loaded successfully")

        if cols_to_keep is None:
            cols_to_keep = COLS_TO_KEEP
        self.df = self.df[cols_to_keep]


    def process_text(self) -> pd.DataFrame:
        text_cols = [col for col in self.df.columns if col != "gl code"]

        for col in text_cols:
            texts = self.df[col].fillna("").astype(str).str.strip().str.lower().tolist()
            cleaned = [self._clean_doc(doc) for doc in self.nlp.pipe(texts)]
            self.df[f"cleaned_{col}"] = cleaned

        cleaned_cols = [f"cleaned_{col}" for col in text_cols]
        self.df["aggregated_text"] = self.df[cleaned_cols].agg(" ".join, axis=1)

        return self.segregate_features_and_labels()

    
    def _clean_doc(self, doc) -> str:
        """Process an already-tokenized spacy Doc object."""
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return " ".join(tokens)


    def segregate_features_and_labels(self) -> Tuple[pd.Series, pd.DataFrame]:
        self.X = self.df["aggregated_text"]
        gl_codes = self.df["gl code"].fillna("").astype(str).str.strip().str.lower().tolist()
        cleaned_labels = [self._clean_doc(doc) for doc in self.nlp.pipe(gl_codes)]
        self.y = pd.Series(cleaned_labels)
        
        print("Text processed successfully!!")
        
        return self.X, pd.get_dummies(self.y, dtype=int)
