import os

import pandas as pd

from dotenv import load_dotenv
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForMaskedLM

from modelling.text_modelling.text_processor import TextProcessor


load_dotenv()
ACCESS_TOKEN = os.environ.get("HF_TOKEN")


#   DOCUMENTATION;
#   Refer here:
#   

class TextTokenizer(nn.Module):
    def __init__(
        self, 
        model_name: str = "google-bert/bert-base-uncased",
        padding: str = "max_length",
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            padding = padding,
            truncation = truncation,
            return_tensors = return_tensors,
            token=ACCESS_TOKEN, 
            **kwargs
        )
        # self.model = AutoModelForMaskedLM.from_pretrained(model_name, token=ACCESS_TOKEN)
        
        
    def tokenize(self, text: str):
        return self.tokenizer(text)
        



if __name__ == "__main__":
    df = pd.read_csv("/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv")    
    
    processor = TextProcessor(df)
    processed_text = processor.process_text()
    
    # print([processed_text["aggregated_text"].iloc[0], processed_text["aggregated_text"].iloc[1]])
    
    tokenizer = TextTokenizer()
    texts = processed_text["aggregated_text"].to_list()
    tokens = tokenizer.tokenize(texts)
    
    # print(tokenizer)
    print(tokens)
    
    # output = model(**tokens)
    
    
    # res:
    # {
    #     'input_ids': [[101, 4636, 6410, 2924, 4651, 102], [101, 4636, 6410, 2924, 4651, 102]], 
    #     'token_type_ids': [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]], 
    #     'attention_mask': [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]
    # }