import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/preprocessed_training_data.csv")

vectorizer = TfidfVectorizer(
    input="content",
    lowercase=False,
    stop_words=None,
    ngram_range=(1, 3),
    max_df=0.95,
    min_df=2,
    max_features=27,
    sublinear_tf=True,
    norm="l2",
    use_idf=True,
    smooth_idf=True,
)

tfidf_matrix = vectorizer.fit_transform(df["aggregated_text"])
feature_names = vectorizer.get_feature_names_out()
print("TF-IDF feature names:", list(feature_names))
print("Shape:", tfidf_matrix.shape)

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Build: aggregated_text | 27 tfidf features | gl_code
result = pd.concat(
    [
        df[["aggregated_text"]].reset_index(drop=True),
        tfidf_df,
        df[["gl_code"]].reset_index(drop=True),
    ],
    axis=1,
)

print("Result shape:", result.shape)
print("Columns:", list(result.columns))

result.to_csv("data/tfidf_features_with_gl_code.csv", index=False)
print("Saved to data/tfidf_features_with_gl_code.csv")
