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
