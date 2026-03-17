import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from float_categorization.pre_processing.text_processor import TextProcessor

class TfIdfEncoder:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.text_processor = TextProcessor()
        
        
    def __call__(self):
        text_cols_to_process = ["payee", "transaction detail", "gl code"]
        self.text_processed_df = self.text_processor(text_cols_to_process)
        
    
    def tf_idf_encoder(self):
        pass