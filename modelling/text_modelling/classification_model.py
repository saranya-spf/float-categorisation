import os

from dotenv import load_dotenv
import torch
import torch.nn as nn

from transformers import AutoModel

load_dotenv()
ACCESS_TOKEN = os.environ.get("HF_TOKEN")


class BERTClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "google-bert/bert-base-uncased",
        hidden_dim: int = 128,
        num_classes: int = 24,
        fine_tune: bool = False,
    ):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, token=ACCESS_TOKEN)

        for param in self.bert.parameters():
            param.requires_grad = fine_tune

        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        #   Use [CLS] token representation (first token)
        #   This is a good approach when we fine-tune the model
        #   However, some pooling (Avg pool, mean pool etc) can be tested as well
        cls_output = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_output)
        probs = torch.softmax(logits, dim=-1)
        return probs
