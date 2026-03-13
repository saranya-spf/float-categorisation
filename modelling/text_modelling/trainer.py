import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

from modelling.text_modelling.text_processor import TextProcessor
from modelling.text_modelling.text_dataset import TextDataset
from modelling.text_modelling.data_collator import TextCollator
from modelling.text_modelling.classification_model import BERTClassifier


CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"
with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODELS_DIR = Path(__file__).resolve().parents[2] / "models_dict"


class Trainer:
    BEST_MODEL_PATH = str(MODELS_DIR / "best_model.pt")
    LAST_MODEL_PATH = str(MODELS_DIR / "last_model.pt")

    def __init__(
        self,
        X_train: pd.Series,
        y_train: pd.DataFrame,
        X_test: Optional[pd.Series] = None,
        y_test: Optional[pd.DataFrame] = None,
    ):
        self.collator = TextCollator()
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        train_dataset = TextDataset(self.X_train, self.y_train)
        test_dataset = None

        if self.X_test is not None and self.y_test is not None:
            test_dataset = TextDataset(X_test, y_test)

        BATCH_SIZE = config["BATCH_SIZE"]

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=self.collator,
        )

        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=8,
                shuffle=False,
                collate_fn=self.collator,
            )

    def train(self, model: nn.Module, restore_best_weights: bool = True):
        NUM_EPOCHS = config["NUM_EPOCHS"]
        LEARNING_RATE = config["LEARNING_RATE"]

        print(f"Using device: {DEVICE}")
        model.to(DEVICE)
        model.train()
        loss_fn = CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
        losses = []
        best_val_f1 = 0.0
        best_model_state = None
        num_classes = next(iter(self.train_loader))[1].shape[1]

        for epoch in range(NUM_EPOCHS):
            total_loss = 0.0
            all_probs = []
            all_labels = []

            for i, (batch_x, batch_y) in enumerate(self.train_loader):
                batch_x = {k: v.to(DEVICE) for k, v in batch_x.items()}
                batch_y = batch_y.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(**batch_x)
                loss = loss_fn(outputs, batch_y)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()

                all_probs.append(outputs.detach().cpu().numpy())
                all_labels.append(batch_y.argmax(dim=1).cpu().numpy())

                print(
                    f"\r  Batch {i+1}/{len(self.train_loader)}, loss: {loss.item():.3f}",
                    end="",
                    flush=True,
                )

            print()
            losses.append(total_loss)

            all_probs = np.concatenate(all_probs)
            all_preds = all_probs.argmax(axis=1)
            all_labels = np.concatenate(all_labels)

            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS}, epoch loss: {total_loss:0.3f}",
                flush=True,
            )
            print(
                f"Train ROC: {roc_auc_score(all_labels, all_probs, multi_class='ovr', labels=range(num_classes)):0.3f}, "
                f"Train F1-score: {f1_score(all_labels, all_preds, average='weighted'):0.3f}",
                flush=True,
            )

            if self.test_loader is not None:
                val_probs, val_preds, val_labels = self._validate(model)
                val_f1 = f1_score(val_labels, val_preds, average="weighted")
                print(
                    f"Validation ROC: {roc_auc_score(val_labels, val_probs, multi_class='ovr', labels=range(num_classes)):0.3f}, "
                    f"Validation F1-score: {val_f1:0.3f}",
                    flush=True,
                )

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_model_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                    self.save_model(model, path=self.BEST_MODEL_PATH)
                    print(f"  -> Best model saved (F1: {best_val_f1:.3f})")

                model.train()

        self.save_model(model, path=self.LAST_MODEL_PATH)
        print(f"Last epoch model saved.")

        if restore_best_weights and best_model_state is not None:
            model.load_state_dict(best_model_state)
            model.to(DEVICE)
            print(f"Restored best model weights (F1: {best_val_f1:.3f})")
        print(f"Training complete. Best Validation F1: {best_val_f1:.3f}")

    def save_model(
        self,
        model: nn.Module,
        path: str = None,
    ) -> None:
        if path is None:
            path = self.BEST_MODEL_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), path)

    @torch.no_grad()
    def _validate(self, model: nn.Module):
        model.eval()
        all_probs = []
        all_labels = []

        for batch_x, batch_y in self.test_loader:
            batch_x = {k: v.to(DEVICE) for k, v in batch_x.items()}
            batch_y = batch_y.to(DEVICE)

            outputs = model(**batch_x)
            all_probs.append(outputs.cpu().numpy())
            all_labels.append(batch_y.argmax(dim=1).cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_preds = all_probs.argmax(axis=1)
        all_labels = np.concatenate(all_labels)
        return all_probs, all_preds, all_labels

    @torch.no_grad()
    def predict(
        self, 
        model: nn.Module, 
        model_dict_path: str, 
        X: pd.Series
    ):
        model.load_state_dict(torch.load(model_dict_path, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

        dataset = TextDataset(X)
        loader = DataLoader(
            dataset=dataset,
            batch_size=8,
            shuffle=False,
            collate_fn=self.collator,
        )

        all_probs = []
        for batch_x, _ in loader:
            batch_x = {k: v.to(DEVICE) for k, v in batch_x.items()}
            outputs = model(**batch_x)
            all_probs.append(outputs.cpu().numpy())

        all_probs = np.concatenate(all_probs)
        all_preds = all_probs.argmax(axis=1)
        return all_probs, all_preds


def build_datasets(df: pd.DataFrame, test_size: float = 0.2):
    processor = TextProcessor(df)
    X, y = processor.process_text()

    # Stratified split using the original label (before one-hot)
    # Use argmax to get single label index for stratification
    y_labels = y.values.argmax(axis=1)

    #   Filter out classes with fewer than 2 samples (stratification requires at least 2)
    #   Try to remove all classses whose sample <= 15
    #   Even after that, we get a 13-class classification problem
    #   Then predict some using the deterministic predictor

    label_counts = pd.Series(y_labels).value_counts()
    valid_mask = pd.Series(y_labels).isin(label_counts[label_counts >= 2].index).values
    X = X[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)
    y_labels = y_labels[valid_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y_labels, random_state=42
    )

    return X_train, X_test, y_train, y_test
