import os
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import spacy


_worker_nlp = None


def _init_worker():
    global _worker_nlp
    _worker_nlp = spacy.load("en_core_web_sm")
    _worker_nlp.disable_pipes("ner", "parser")


def _process_chunk(texts: List[str]) -> List[str]:
    return [
        " ".join(
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        )
        for doc in _worker_nlp.pipe(texts, batch_size=256)
    ]


class TextProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.columns = self.df.columns.str.strip().str.casefold()
        print("Loading spacy text processing model...")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.disable_pipes("ner", "parser")

        print("Spacy model loaded successfully")

    def _parallel_clean(self, all_texts: List[str]) -> List[str]:
        if len(all_texts) < 500:
            return [
                self._clean_doc(doc) for doc in self.nlp.pipe(all_texts, batch_size=256)
            ]

        n_workers = min(os.cpu_count() or 1, 4)
        chunk_size = max(1, len(all_texts) // n_workers)
        chunks = [
            all_texts[i : i + chunk_size] for i in range(0, len(all_texts), chunk_size)
        ]

        with ProcessPoolExecutor(
            max_workers=n_workers, initializer=_init_worker
        ) as executor:
            results = list(executor.map(_process_chunk, chunks))

        return [text for chunk_result in results for text in chunk_result]

    def __call__(self, cols: List[str]) -> pd.DataFrame:
        cols = [c.strip().casefold() for c in cols]
        missing = [c for c in cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Columns not found in DataFrame: {missing}")

        all_texts: List[str] = []
        col_sizes: List[int] = []

        for col in cols:
            texts = self.df[col].fillna("").astype(str).str.strip().str.lower().tolist()
            all_texts.extend(texts)
            col_sizes.append(len(texts))

        all_cleaned = self._parallel_clean(all_texts)

        result = pd.DataFrame(index=self.df.index)
        idx = 0
        for i, col in enumerate(cols):
            size = col_sizes[i]
            result[f"cleaned_{col}"] = all_cleaned[idx : idx + size]
            idx += size

        return result

    def process_for_training(
        self, label_col: str = "gl code"
    ) -> Tuple[pd.Series, pd.DataFrame]:
        label_col = label_col.strip().casefold()
        feature_cols = [col for col in self.df.columns if col != label_col]

        cleaned = self(feature_cols + [label_col])

        cleaned_labels = cleaned[f"cleaned_{label_col}"].tolist()
        cleaned_feature_cols = [f"cleaned_{col}" for col in feature_cols]
        self.df["aggregated_text"] = cleaned[cleaned_feature_cols].agg(" ".join, axis=1)

        return self._segregate_features_and_labels(cleaned_labels)

    def _clean_doc(self, doc) -> str:
        """Process an already-tokenized spacy Doc object."""
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return " ".join(tokens)

    def _segregate_features_and_labels(
        self, cleaned_labels: List[str]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        self.X = self.df["aggregated_text"]
        self.y = pd.Series(cleaned_labels)

        print("Text processed successfully!!")

        return self.X, pd.get_dummies(self.y, dtype=int)
