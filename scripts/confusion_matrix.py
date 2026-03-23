"""Generate a full confusion matrix from test evaluation results."""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

df = pd.read_csv("analysis/evaluation_results_test.csv")

# Get all unique labels (union of actual and predicted)
all_labels = sorted(
    set(df["actual_label"].dropna().unique())
    | set(df["predicted_label"].dropna().unique())
)

actual = df["actual_label"].fillna("(empty)")
predicted = df["predicted_label"].fillna("(empty)")
all_labels_clean = sorted(set(actual.unique()) | set(predicted.unique()))

cm = confusion_matrix(actual, predicted, labels=all_labels_clean)

# Print as a readable table
print("CONFUSION MATRIX (rows=actual, cols=predicted)")
print(f"{'':>30s}", end="")
# Short column headers
short = [l[:12] for l in all_labels_clean]
for s in short:
    print(f" {s:>12s}", end="")
print("  | Total")
print("-" * (32 + 13 * len(all_labels_clean) + 10))

for i, label in enumerate(all_labels_clean):
    print(f"{label:>30s}", end="")
    for j in range(len(all_labels_clean)):
        val = cm[i][j]
        if val == 0:
            print(f" {'·':>12s}", end="")
        else:
            print(f" {val:>12d}", end="")
    print(f"  | {cm[i].sum():>4d}")

print("-" * (32 + 13 * len(all_labels_clean) + 10))
print(f"{'Predicted total':>30s}", end="")
for j in range(len(all_labels_clean)):
    print(f" {cm[:, j].sum():>12d}", end="")
print()

# Also print a more readable version: only non-zero off-diagonal entries
print("\n\nMISCLASSIFICATION DETAILS (actual → predicted : count)")
print("=" * 60)
for i, actual_label in enumerate(all_labels_clean):
    for j, pred_label in enumerate(all_labels_clean):
        if i != j and cm[i][j] > 0:
            print(f"  {actual_label:>30s} → {pred_label:<30s} : {cm[i][j]}")
