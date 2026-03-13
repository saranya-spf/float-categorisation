from typing import List, Dict, Any, Tuple

import torch

from modelling.text_modelling.tokenizer import TextTokenizer


class TextCollator:
    def __init__(self, model_name: str = "google-bert/bert-base-uncased"):
        self.tokenizer = TextTokenizer(model_name=model_name)

    def __call__(
        self, batch: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        texts = [item["text"] for item in batch]
        tokenized = self.tokenizer.tokenize(texts)

        collated = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        if "token_type_ids" in tokenized:
            collated["token_type_ids"] = tokenized["token_type_ids"]

        labels = None
        if "labels" in batch[0]:
            labels = torch.stack([item["labels"] for item in batch])
        return collated, labels
