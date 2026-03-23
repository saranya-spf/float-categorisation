import yaml
from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)


class TfIdfEncoder:
    def __init__(self, overrides=None):
        params = dict(config)
        if overrides:
            params.update(overrides)
        self.vectorizer = TfidfVectorizer(
            input="content",
            lowercase=False,
            stop_words=params.get("STOP_WORDS"),
            ngram_range=tuple(params.get("NGRAM_RANGE")),
            max_df=params.get("MAX_DF"),
            min_df=params.get("MIN_DF"),
            max_features=params.get("MAX_FEATURES"),
            sublinear_tf=params.get("SUBLINEAR_TF"),
            norm=params.get("NORM"),
            use_idf=params.get("USE_IDF"),
            smooth_idf=params.get("SMOOTH_IDF"),
        )

    #   For training (since new vocab is created)
    def fit_transform(self, texts: pd.Series) -> sp.csr_matrix:
        return self.vectorizer.fit_transform(texts)

    #   For test data
    def transform(self, texts: pd.Series) -> sp.csr_matrix:
        return self.vectorizer.transform(texts)

