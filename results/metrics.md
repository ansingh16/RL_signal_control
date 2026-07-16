# Evaluation report

Site: `data/raw/df_4374.parquet`  |  Test samples: 15,731  |  Features: 22

One-step-ahead congestion forecast (next 15-minute interval), evaluated on a chronologically held-out test set.

Congested share by split: train 5.5% | val 18.8% | **test 25.6%**. The split is chronological and congestion rises over the period, so the figures below are measured against the test prevalence rather than the site-wide average.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **LSTM (forecast)** | 0.846 | 0.628 | 0.977 | 0.765 |
| Majority class | 0.744 | 0.000 | 0.000 | 0.000 |
| Rush-hour rule | 0.740 | 0.490 | 0.344 | 0.404 |
| Speed threshold* | 0.975 | 1.000 | 0.901 | 0.948 |

LSTM average precision (PR-AUC): **0.893** vs no-skill baseline of 0.256.

\* The speed-threshold baseline predicts congestion from the *current* interval's average speed, which is what defines the label. It is a sanity check on labelling, not a forecasting competitor: the LSTM and the time-based baselines only use information available before the predicted interval.
