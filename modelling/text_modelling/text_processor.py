import os
from typing import Optional, List, Tuple
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import spacy


COLS_TO_KEEP = [
    "transaction type",
    "payee",
    "transaction detail",
    "gl code",
]

_worker_nlp = None


def _init_worker():
    """Load a spaCy model once per worker process."""
    global _worker_nlp
    _worker_nlp = spacy.load("en_core_web_sm")
    _worker_nlp.disable_pipes("ner", "parser")


def _process_chunk(texts: List[str]) -> List[str]:
    """Clean a chunk of texts in a worker process."""
    return [
        " ".join(
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        )
        for doc in _worker_nlp.pipe(texts, batch_size=256)
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

    def _parallel_clean(self, all_texts: List[str]) -> List[str]:
        """Clean texts using multiprocessing for row-level parallelism."""
        
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

    def process_text(self) -> pd.DataFrame:
        text_cols = [col for col in self.df.columns if col != "gl code"]

        all_texts: List[str] = []
        col_sizes: List[int] = []

        for col in text_cols:
            texts = self.df[col].fillna("").astype(str).str.strip().str.lower().tolist()
            all_texts.extend(texts)
            col_sizes.append(len(texts))

        gl_codes = (
            self.df["gl code"].fillna("").astype(str).str.strip().str.lower().tolist()
        )
        all_texts.extend(gl_codes)
        col_sizes.append(len(gl_codes))

        all_cleaned = self._parallel_clean(all_texts)

        idx = 0
        for i, col in enumerate(text_cols):
            size = col_sizes[i]
            self.df[f"cleaned_{col}"] = all_cleaned[idx : idx + size]
            idx += size

        cleaned_labels = all_cleaned[idx : idx + col_sizes[-1]]

        cleaned_cols = [f"cleaned_{col}" for col in text_cols]
        self.df["aggregated_text"] = self.df[cleaned_cols].agg(" ".join, axis=1)

        return self.segregate_features_and_labels(cleaned_labels)


    def _clean_doc(self, doc) -> str:
        """Process an already-tokenized spacy Doc object."""
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop and not token.is_punct and token.text.strip()
        ]
        return " ".join(tokens)

    def segregate_features_and_labels(
        self, cleaned_labels: List[str]
    ) -> Tuple[pd.Series, pd.DataFrame]:
        self.X = self.df["aggregated_text"]
        self.y = pd.Series(cleaned_labels)

        print("Text processed successfully!!")

        return self.X, pd.get_dummies(self.y, dtype=int)
