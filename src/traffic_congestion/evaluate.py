"""Evaluation: model metrics on a held-out loader plus simple baselines.

The baselines exist to keep the LSTM honest. On a dataset that is ~98.7%
non-congested, accuracy alone is misleading, so every model is compared against
a majority-class predictor, a rush-hour rule, and a speed threshold.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
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


def predict_proba_loader(model, loader, pos_label=1):
    """Run the model and return (y_true, y_prob) where y_prob is P(congested)."""
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for features, labels in loader:
            probs = F.softmax(model(features), dim=1)[:, pos_label]
            y_true.extend(labels.numpy())
            y_prob.extend(probs.numpy())
    return np.array(y_true), np.array(y_prob)


def per_hour_accuracy(y_true, y_pred, hours):
    """Accuracy and congested-class recall broken down by hour of day.

    Returns a dict ``hour -> {accuracy, recall, n}`` so we can see when the
    model actually helps versus when a clock-based rule would do.
    """
    y_true, y_pred, hours = np.asarray(y_true), np.asarray(y_pred), np.asarray(hours)
    out = {}
    for h in range(24):
        mask = hours == h
        if not mask.any():
            continue
        yt, yp = y_true[mask], y_pred[mask]
        out[h] = {
            "accuracy": accuracy_score(yt, yp),
            "recall": recall_score(yt, yp, pos_label=1, zero_division=0),
            "n": int(mask.sum()),
        }
    return out


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


# --- Plotting -------------------------------------------------------------
# Functions write a figure to ``path`` (matplotlib Agg backend) and return it,
# so they work headless in scripts and CI.

def _import_plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_confusion_matrix(y_true, y_pred, path, labels=("Free-flow", "Congested")):
    plt = _import_plt()
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_pr_curve(y_true, y_prob, path):
    plt = _import_plt()
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = np.mean(y_true)  # precision of a random classifier
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(recall, precision, color="#1f77b4", label=f"LSTM (AP = {ap:.3f})")
    ax.axhline(baseline, ls="--", color="grey", label=f"No-skill ({baseline:.3f})")
    ax.set_xlabel("Recall (congested)")
    ax.set_ylabel("Precision (congested)")
    ax.set_title("Precision-recall curve")
    ax.set_ylim(0, 1.02)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return ap


def plot_per_hour_accuracy(per_hour, path):
    plt = _import_plt()
    hours = sorted(per_hour)
    acc = [per_hour[h]["accuracy"] for h in hours]
    rec = [per_hour[h]["recall"] for h in hours]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, acc, marker="o", label="Accuracy")
    ax.plot(hours, rec, marker="s", label="Congested recall")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Score")
    ax.set_title("Test performance by hour of day")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_baseline_comparison(model_metrics, baselines, path):
    plt = _import_plt()
    rows = [("LSTM", model_metrics)] + list(baselines.items())
    names = [n for n, _ in rows]
    acc = [m["accuracy"] for _, m in rows]
    rec = [m["recall"] for _, m in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - 0.2, acc, 0.4, label="Accuracy", color="#4c72b0")
    ax.bar(x + 0.2, rec, 0.4, label="Congested recall", color="#dd8452")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model vs baselines (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_training_history(train_losses, val_losses, val_accuracies, path):
    """Two-panel training curve: loss (train vs val) and validation accuracy."""
    plt = _import_plt()
    epochs = range(1, len(train_losses) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, train_losses, marker="o", ms=3, label="Train")
    ax_loss.plot(epochs, val_losses, marker="s", ms=3, label="Validation")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training and validation loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    ax_acc.plot(epochs, val_accuracies, marker="o", ms=3, color="#2ca02c")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Validation accuracy")
    ax_acc.set_title("Validation accuracy")
    ax_acc.set_ylim(0, 1.02)
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
