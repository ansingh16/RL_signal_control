"""Evaluation: model metrics on a held-out loader plus simple baselines.

The baselines exist to keep the LSTM honest. On a dataset that is ~98.7%
non-congested, accuracy alone is misleading, so every model is compared against
a majority-class predictor, a rush-hour rule, and a speed threshold.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RUSH_HOURS = [7, 8, 9, 17, 18, 19]
WEEKDAYS = [0, 1, 2, 3, 4]


def predict_loader(model, loader):
    """Run the model over a loader and return (y_true, y_pred) arrays."""
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for features, labels in loader:
            outputs = model(features)
            y_true.extend(labels.numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).numpy())
    return np.array(y_true), np.array(y_pred)


def classification_metrics(y_true, y_pred, pos_label=1):
    """Accuracy plus precision/recall/F1 for the congested class."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def evaluate_model(model, loader):
    """Evaluate a trained model on a loader and return a metrics dict."""
    y_true, y_pred = predict_loader(model, loader)
    return classification_metrics(y_true, y_pred)


def baseline_metrics(df, target_col="congestion", speed_col="Avg mph"):
    """Compute the three reference baselines on a feature frame.

    Returns a dict keyed by baseline name with accuracy and congested-class
    recall, so they can sit alongside the model in a comparison table.
    """
    y_true = df[target_col].values

    results = {}

    # 1. Always predict the majority (non-congested) class.
    majority = np.zeros_like(y_true)
    results["majority_class"] = classification_metrics(y_true, majority)

    # 2. Congestion during weekday rush hours.
    rush = (df["hour"].isin(RUSH_HOURS) & df["dayofweek"].isin(WEEKDAYS)).astype(int).values
    results["rush_hour_rule"] = classification_metrics(y_true, rush)

    # 3. Current average speed below 25 mph.
    speed = (df[speed_col] < 25).astype(int).values
    results["speed_threshold"] = classification_metrics(y_true, speed)

    return results


def print_comparison(model_metrics, baselines):
    """Pretty-print the model against the baselines."""
    rows = [("LSTM", model_metrics)] + list(baselines.items())
    print(f"{'Model':<18}{'Accuracy':>10}{'Recall':>10}{'F1':>10}")
    print("-" * 48)
    for name, m in rows:
        print(f"{name:<18}{m['accuracy']:>10.4f}{m['recall']:>10.4f}{m['f1']:>10.4f}")
