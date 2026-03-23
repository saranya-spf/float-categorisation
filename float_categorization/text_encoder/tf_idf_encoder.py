import yaml
from pathlib import Path

import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

class TfIdfEncoder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            input="content",
            lowercase=False,
            stop_words=config.get("STOP_WORDS"),
            ngram_range=tuple(config.get("NGRAM_RANGE")),
            max_df=config.get("MAX_DF"),
            min_df=config.get("MIN_DF"),
            max_features=config.get("MAX_FEATURES"),
            sublinear_tf=config.get("SUBLINEAR_TF"),
            norm=config.get("NORM"),
            use_idf=config.get("USE_IDF"),
            smooth_idf=config.get("SMOOTH_IDF"),
        )

    #   For training (since new vocab is created)
    def fit_transform(self, texts: pd.Series) -> sp.csr_matrix:
        return self.vectorizer.fit_transform(texts)

    #   For test data
    def transform(self, texts: pd.Series) -> sp.csr_matrix:
        return self.vectorizer.transform(texts)


# if __name__ == "__main__":
#     df = pd.read_csv(
#         "/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv"
#     )
#     processor = PreProcessor(df, is_training=True)
#     processed_df, deterministic_df = processor()

#     tf_idf = TfIdfEncoder()
#     tfidf_matrix = tf_idf.fit_transform(processed_df["aggregated_text"])

#     print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
#     print(f"Vocabulary size: {len(tf_idf.vectorizer.vocabulary_)}")
#     print(f"Sample features: {list(tf_idf.vectorizer.get_feature_names_out()[:20])}")
