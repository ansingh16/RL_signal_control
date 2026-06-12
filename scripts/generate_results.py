"""Train the congestion LSTM on one checkpoint and write an evaluation report.

Produces, under ``results/``:
  - confusion_matrix.png
  - pr_curve.png
  - per_hour_accuracy.png
  - baseline_comparison.png
  - metrics.md          (table of the model vs the three baselines)

Run from the repo root:

    PYTHONPATH=src python scripts/generate_results.py --site data/raw/df_4374.parquet
"""

import argparse
import os
from collections import Counter

import torch

from traffic_congestion import (
    Config,
    ModelTrainer,
    TrafficDataProcessor,
    TrafficNet,
)
from traffic_congestion.evaluate import (
    baseline_metrics,
    classification_metrics,
    per_hour_accuracy,
    plot_baseline_comparison,
    plot_confusion_matrix,
    plot_per_hour_accuracy,
    plot_pr_curve,
    plot_training_history,
    predict_loader,
    predict_proba_loader,
)

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", default="data/raw/df_4374.parquet")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cfg = Config()

    # --- Data ---
    proc = TrafficDataProcessor(args.site)
    proc.prepare_site_data(
        speed_threshold=cfg.data.speed_threshold,
        lags=cfg.data.lags,
        roll_windows=cfg.data.roll_windows,
    )
    train_loader, val_loader, test_loader, info = proc.create_loaders_from_df(
        seq_len=cfg.data.seq_len,
        splits=cfg.data.splits,
        batch_size=cfg.data.batch_size,
        handle_imbalance=cfg.data.handle_imbalance,
    )

    # --- Class weights from the training portion only ---
    test_start = info["test_rows_range"][0]
    train_labels = proc.df["congestion"].iloc[: info["train_rows_range"][1] + 1]
    counts = Counter(train_labels.tolist())
    total = sum(counts.values())
    class_weights = {c: total / n for c, n in counts.items()}

    # --- Train ---
    model = TrafficNet(
        input_size=len(info["feature_cols"]),
        hidden_size=cfg.model.hidden_size,
        num_layers=cfg.model.num_layers,
        dropout=cfg.model.dropout,
    )
    trainer = ModelTrainer(
        model,
        class_weights=class_weights,
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
        checkpoint_path=cfg.train.checkpoint_path,
    )
    trainer.train_model(
        train_loader, val_loader,
        epochs=args.epochs,
        early_stopping_patience=cfg.train.early_stopping_patience,
    )

    # --- Evaluate on the held-out test set ---
    y_true, y_pred = predict_loader(model, test_loader)
    _, y_prob = predict_proba_loader(model, test_loader)
    model_metrics = classification_metrics(y_true, y_pred)

    # Baselines on the exact rows the model predicts (the labels after each window)
    seq_len = cfg.data.seq_len
    test_label_df = proc.df.iloc[test_start + seq_len :]
    baselines = baseline_metrics(test_label_df)

    hours = test_label_df["hour"].values
    by_hour = per_hour_accuracy(y_true, y_pred, hours)

    # --- Plots ---
    plot_training_history(
        trainer.train_losses, trainer.val_losses, trainer.val_accuracies,
        f"{RESULTS_DIR}/training_history.png",
    )
    plot_confusion_matrix(y_true, y_pred, f"{RESULTS_DIR}/confusion_matrix.png")
    ap = plot_pr_curve(y_true, y_prob, f"{RESULTS_DIR}/pr_curve.png")
    plot_per_hour_accuracy(by_hour, f"{RESULTS_DIR}/per_hour_accuracy.png")
    plot_baseline_comparison(model_metrics, baselines, f"{RESULTS_DIR}/baseline_comparison.png")

    # --- Metrics table ---
    write_metrics_md(args.site, model_metrics, ap, baselines, info)
    print("\nWrote results to", RESULTS_DIR)


def write_metrics_md(site, model_metrics, ap, baselines, info):
    rows = [("LSTM", model_metrics)] + list(baselines.items())
    lines = [
        "# Evaluation report",
        "",
        f"Site: `{site}`  |  Test samples: {info['n_test_samples']:,}  "
        f"|  Features: {len(info['feature_cols'])}",
        "",
        "One-step-ahead congestion forecast (next 15-minute interval), evaluated on a "
        "chronologically held-out test set.",
        "",
        "| Model | Accuracy | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
    ]
    pretty = {
        "LSTM": "**LSTM (forecast)**",
        "majority_class": "Majority class",
        "rush_hour_rule": "Rush-hour rule",
        "speed_threshold": "Speed threshold*",
    }
    for name, m in rows:
        lines.append(
            f"| {pretty.get(name, name)} | {m['accuracy']:.3f} | {m['precision']:.3f} "
            f"| {m['recall']:.3f} | {m['f1']:.3f} |"
        )
    lines += [
        "",
        f"LSTM average precision (PR-AUC): **{ap:.3f}** vs no-skill baseline "
        f"of {model_metrics_pos_rate(baselines):.3f}.",
        "",
        "\\* The speed-threshold baseline predicts congestion from the *current* interval's "
        "average speed, which is what defines the label. It is a sanity check on labelling, "
        "not a forecasting competitor: the LSTM and the time-based baselines only use "
        "information available before the predicted interval.",
    ]
    with open(f"{RESULTS_DIR}/metrics.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def model_metrics_pos_rate(baselines):
    # The no-skill PR baseline equals the positive (congested) rate, which is
    # 1 - the majority-class accuracy.
    return 1 - baselines["majority_class"]["accuracy"]


if __name__ == "__main__":
    main()
