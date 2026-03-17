import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

from float_categorization.pre_processing.text_processor import TextProcessor
from float_categorization.text_encoder.text_dataset import TextDataset
from float_categorization.text_encoder.data_collator import TextCollator


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
        batch_size: Optional[int] = None,
        imbalance_training: bool = True,
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

        batch_size = batch_size or config["BATCH_SIZE"]

        sampler = None
        self.class_weights = None

        if imbalance_training:
            targets = torch.argmax(torch.tensor(self.y_train.values), dim=1)
            unique_classes = np.unique(targets.numpy())
            class_sample_count = np.array(
                [len(np.where(targets.numpy() == t)[0]) for t in unique_classes]
            )
            weight = 1.0 / class_sample_count
            class_to_weight = dict(zip(unique_classes, weight))
            samples_weight = np.array(
                [class_to_weight[int(t)] for t in targets.numpy()]
            )

            samples_weight = torch.from_numpy(samples_weight)
            sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

            num_classes = self.y_train.shape[1]
            cw = np.zeros(num_classes, dtype=np.float64)
            for cls_idx, w in class_to_weight.items():
                cw[cls_idx] = w
            cw /= cw.sum()
            self.class_weights = torch.tensor(cw, dtype=torch.float32)

        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=self.collator,
        )

        self.test_loader = None
        if test_dataset is not None:
            self.test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.collator,
            )

    def train(
        self,
        model: nn.Module,
        restore_best_weights: bool = True,
        num_epochs: Optional[int] = None,
        learning_rate: Optional[float] = None,
        verbose: bool = True,
    ) -> float:
        NUM_EPOCHS = num_epochs or config["NUM_EPOCHS"]
        LEARNING_RATE = learning_rate or config["LEARNING_RATE"]

        if verbose:
            print(f"Using device: {DEVICE}")
        model.to(DEVICE)
        model.train()
        loss_fn = CrossEntropyLoss(
            weight=(
                self.class_weights.to(DEVICE)
                if self.class_weights is not None
                else None
            )
        )
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

                if verbose:
                    print(
                        f"\r  Batch {i + 1}/{len(self.train_loader)}, loss: {loss.item():.3f}",
                        end="",
                        flush=True,
                    )

            if verbose:
                print()
            losses.append(total_loss)

            all_probs = np.concatenate(all_probs)
            all_preds = all_probs.argmax(axis=1)
            all_labels = np.concatenate(all_labels)

            if verbose:
                print(
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}, epoch loss: {total_loss:0.3f}",
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
                if verbose:
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
                    if verbose:
                        self.save_model(model, path=self.BEST_MODEL_PATH)
                        print(f"  -> Best model saved (F1: {best_val_f1:.3f})")

                model.train()

        if verbose:
            self.save_model(model, path=self.LAST_MODEL_PATH)
            print("Last epoch model saved.")

            if restore_best_weights and best_model_state is not None:
                model.load_state_dict(best_model_state)
                model.to(DEVICE)
                print(f"Restored best model weights (F1: {best_val_f1:.3f})")
            print(f"Training complete. Best Validation F1: {best_val_f1:.3f}")

        return best_val_f1

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
    def predict(self, model: nn.Module, model_dict_path: str, X: pd.Series):
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

    @staticmethod
    def tune(
        df: pd.DataFrame,
        n_trials: int = 20,
        test_size: float = 0.2,
        epoch_range: tuple = (2, 10),
        lr_range: tuple = (1e-5, 1e-3),
        batch_size_choices: tuple = (16, 32, 64, 128),
    ) -> dict:
        """Run Optuna hyperparameter search over num_epochs, learning_rate, and batch_size."""
        from float_categorization.modelling.models.classification_model import BERTClassifier

        X_train, X_test, y_train, y_test = build_datasets(df, test_size=test_size)
        num_classes = y_train.shape[1]

        def objective(trial: optuna.Trial) -> float:
            num_epochs = trial.suggest_int("num_epochs", *epoch_range)
            lr = trial.suggest_float("learning_rate", *lr_range, log=True)
            batch_size = trial.suggest_categorical(
                "batch_size", list(batch_size_choices)
            )

            model = BERTClassifier(fine_tune=True, num_classes=num_classes)
            trainer = Trainer(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                batch_size=batch_size,
            )
            best_val_f1 = trainer.train(
                model,
                num_epochs=num_epochs,
                learning_rate=lr,
                restore_best_weights=False,
                verbose=False,
            )
            return best_val_f1

        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(),
        )
        study.optimize(objective, n_trials=n_trials)

        print(f"\nBest trial F1: {study.best_trial.value:.4f}")
        print(f"Best params: {study.best_trial.params}")
        return study.best_trial.params


def build_datasets(df: pd.DataFrame, test_size: float = 0.2):
    text_processor = TextProcessor(df)
    X, y = text_processor.process_for_training()

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
