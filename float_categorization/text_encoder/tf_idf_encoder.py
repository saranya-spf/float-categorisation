import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfEncoder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            input="content",
            lowercase=False,
            stop_words=None,
            ngram_range=(1, 3),
            max_df=0.95,
            min_df=2,
            max_features=1000,
            sublinear_tf=True,
            norm="l2",
            use_idf=True,
            smooth_idf=True,
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
