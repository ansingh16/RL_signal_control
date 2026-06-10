"""Tests for the feature-engineering pipeline and sequence batching."""

import numpy as np
import torch

from traffic_congestion import TrafficDataProcessor, TrafficDataset

# 2 base signals + 3 lags x 2 + 3 windows x 4 (mean/std for volume & speed) + 2 time
EXPECTED_FEATURES = 22


def test_prepare_site_data_feature_set(raw_parquet):
    proc = TrafficDataProcessor(raw_parquet)
    df = proc.prepare_site_data()

    # Feature order is frozen at build time and excludes the target.
    assert proc.feature_cols is not None
    assert len(proc.feature_cols) == EXPECTED_FEATURES
    assert "congestion" not in proc.feature_cols
    # No NaNs survive the lag/rolling warm-up region.
    assert not df.isna().any().any()


def test_congestion_target_matches_speed_threshold(raw_parquet):
    proc = TrafficDataProcessor(raw_parquet)
    df = proc.prepare_site_data(speed_threshold=30)

    expected = (df["Avg mph"] < 30).astype(int)
    assert (df["congestion"] == expected).all()
    # The synthetic data congests every third interval, so both classes appear.
    assert df["congestion"].nunique() == 2


def test_lag_features_use_correct_offsets(raw_parquet):
    proc = TrafficDataProcessor(raw_parquet)
    df = proc.prepare_site_data(lags=(1, 2, 4))

    # Rows are contiguous in time after dropna, so an interior row's lag-k value
    # equals the Total Volume k positions earlier.
    i = 20
    assert df["volume_lag1"].iloc[i] == df["Total Volume"].iloc[i - 1]
    assert df["volume_lag2"].iloc[i] == df["Total Volume"].iloc[i - 2]
    assert df["volume_lag4"].iloc[i] == df["Total Volume"].iloc[i - 4]
    assert df["speed_lag1"].iloc[i] == df["Avg mph"].iloc[i - 1]


def test_rolling_features_match_manual_window(raw_parquet):
    proc = TrafficDataProcessor(raw_parquet)
    df = proc.prepare_site_data(roll_windows=(4,))

    i = 20
    window = df["Total Volume"].iloc[i - 3 : i + 1]
    assert np.isclose(df["volume_roll_mean_4"].iloc[i], window.mean())
    assert np.isclose(df["volume_roll_std_4"].iloc[i], window.std())


def test_dataset_window_shapes_and_label_offset(raw_parquet):
    proc = TrafficDataProcessor(raw_parquet)
    df = proc.prepare_site_data()

    seq_len = 8
    ds = TrafficDataset(df, proc.feature_cols, "congestion", seq_len=seq_len)
    assert len(ds) == len(df) - seq_len

    X, y = ds[0]
    assert X.shape == (seq_len, EXPECTED_FEATURES)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long
    # The label is the interval immediately after the window (one-step-ahead).
    assert int(y) == int(df["congestion"].iloc[seq_len])


def test_create_loaders_chronological_split(raw_parquet):
    proc = TrafficDataProcessor(raw_parquet)
    proc.prepare_site_data()
    train_loader, val_loader, test_loader, info = proc.create_loaders_from_df(
        seq_len=8, batch_size=16
    )

    # Every window-able sample lands in exactly one split.
    assert (
        info["n_train_samples"] + info["n_val_samples"] + info["n_test_samples"]
        == info["n_samples"]
    )
    assert info["train_rows_range"][1] < info["test_rows_range"][0]
    assert len(info["feature_cols"]) == EXPECTED_FEATURES

    xb, yb = next(iter(train_loader))
    assert xb.shape[1:] == (8, EXPECTED_FEATURES)
    assert yb.ndim == 1


def test_create_loaders_requires_prepared_data(raw_parquet):
    proc = TrafficDataProcessor(raw_parquet)
    try:
        proc.create_loaders_from_df()
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError when prepare_site_data() was skipped")
