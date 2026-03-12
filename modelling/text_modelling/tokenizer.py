import os

from dotenv import load_dotenv
import torch
import torch.nn as nn
from transformers import AutoTokenizer

load_dotenv()
ACCESS_TOKEN = os.environ.get("HF_TOKEN")

class TextTokenizer(nn.Module):
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        padding: str = "max_length",
        max_length=128,
        truncation: bool = True,
        return_tensors: str = "pt",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=ACCESS_TOKEN,
        )
        self.padding = padding
        self.truncation = truncation
        self.return_tensors = return_tensors
        self.max_length = max_length

    def tokenize(self, text: str):
        return self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
        )
