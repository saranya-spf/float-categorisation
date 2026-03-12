import os
from typing import Dict, Any
import math

from dotenv import load_dotenv
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM

from modelling.text_modelling.text_processor import TextProcessor
from modelling.text_modelling.tokenizer import TextTokenizer

load_dotenv()
ACCESS_TOKEN = os.environ.get("HF_TOKEN")

class TextInferencer(nn.Module):
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        device_name: str = "mps" if torch.backends.mps.is_available() else "cpu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.device = torch.device(device_name)
        
        # Load model and immediately move it to the correct device (MPS for Mac)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name,
            token=ACCESS_TOKEN,
        ).to(self.device)
        
        # Set to evaluation mode (important for dropout/batchnorm layers)
        self.model.eval()

    def get_outputs(self, tokens: Dict[str, Any]) -> torch.Tensor:
        # Move all token tensors to the same device as the model
        token_tensors = {k: torch.tensor(v).to(self.device) for k, v in tokens.items()}
        
        # Disable gradient tracking for inference to save massive amounts of memory
        with torch.no_grad():
            outputs = self.model(**token_tensors)
            
        return outputs.logits # or whatever specific output you need


if __name__ == "__main__":
    df = pd.read_csv("/Users/saranya.pal/Desktop/Projects/float_categorization/data/Copy of Float Sample Data - Sheet5.csv") # Your CSV path
    
    processor = TextProcessor(df)
    processed_text = processor.process_text()
    
    tokenizer = TextTokenizer()
    texts = processed_text["aggregated_text"].to_list()
    
    inferencer = TextInferencer()
    
    # Process in BATCHES instead of all at once
    batch_size = 8  # Adjust this depending on your sequence lengths and available RAM
    all_outputs = []
    
    print(f"Processing {len(texts)} rows in batches of {batch_size}...")
    
    for i in range(0, len(texts[:1]), batch_size):
        batch_texts = texts[i : i + batch_size]
        
        # Assuming your TextTokenizer handles proper HuggingFace padding/truncation
        # It should return input_ids and attention_mask
        batch_tokens = tokenizer.tokenize(batch_texts)
        
        batch_output = inferencer.get_outputs(tokens=batch_tokens)
        
        # Move output back to CPU if you plan to store it in a list to prevent GPU/MPS memory build-up
        all_outputs.append(batch_output.cpu())
        
        print(f"Processed batch {i // batch_size + 1}/{math.ceil(len(texts) / batch_size)}")