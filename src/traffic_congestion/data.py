"""Data loading, feature engineering, and sequence batching for the TfL M25
congestion dataset.

The raw data is one parquet file per motorway checkpoint, recorded at 15-minute
intervals. ``TrafficDataProcessor`` turns a single site's parquet into a feature
frame and chronological train/val/test ``DataLoader``s. ``TrafficDataset`` wraps
the feature frame into sliding-window sequences for the LSTM.
"""

from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Columns we never feed to the model: identifiers and the raw timestamp inputs.
_DROP_COLUMNS = ["Report Date", "Time Period Ending", "Time Interval", "Site Name", "site_id"]


class TrafficDataset(Dataset):
    """Sliding-window sequences over a feature frame.

    Each sample is ``seq_len`` consecutive rows of features; its label is the
    congestion flag of the row immediately after the window, so the model
    predicts the next interval from the recent history.
    """

    def __init__(self, df, feature_cols, target_col, seq_len=16):
        self.seq_len = seq_len
        self.features = df[feature_cols].values.astype(np.float32)
        self.targets = df[target_col].values.astype(int)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        X = self.features[idx : idx + self.seq_len]
        y = self.targets[idx + self.seq_len]
        return torch.from_numpy(X), torch.tensor(y, dtype=torch.long)


class TrafficDataProcessor:
    """Load and feature-engineer a single checkpoint's traffic time series."""

    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.info = None

    @staticmethod
    def _replace_empty(col):
        """Coerce a column to float, mapping empty/non-numeric entries to 0."""
        return pd.to_numeric(col, errors="coerce").fillna(0.0)

    def prepare_site_data(self, speed_threshold=30, lags=(1, 2, 4), roll_windows=(4, 8, 16)):
        """Build the feature frame for one site.

        Parameters
        ----------
        speed_threshold : int
            Average speed (mph) below which an interval is labelled congested.
        lags : tuple of int
            Lag offsets in 15-minute steps (1=15min, 2=30min, 4=1h).
        roll_windows : tuple of int
            Rolling-window sizes in 15-minute steps (4=1h, 8=2h, 16=4h).
        """
        print(f"Preprocessing data for site {self.filepath}")

        df = pd.read_parquet(self.filepath)

        # Build a datetime index from the report date and interval-ending time.
        df["time"] = pd.to_datetime(
            df["Report Date"].str.split("T").str[0] + df["Time Period Ending"],
            format="%Y-%m-%d%H:%M:%S",
        )
        df = df.set_index("time").sort_index()
        df = df.drop(columns=[c for c in _DROP_COLUMNS if c in df.columns])
        df = df.apply(self._replace_empty)

        # Keep the two core signals; everything else is derived from them.
        df = df[["Total Volume", "Avg mph"]].copy()

        # Sensor dropouts show up as 0; treat them as missing and interpolate.
        df["Total Volume"] = df["Total Volume"].replace(0, np.nan).interpolate()
        df["Avg mph"] = df["Avg mph"].replace(0, np.nan).interpolate()

        # Lag features (recent volume/speed).
        for lag in lags:
            df[f"volume_lag{lag}"] = df["Total Volume"].shift(lag)
            df[f"speed_lag{lag}"] = df["Avg mph"].shift(lag)

        # Rolling mean/std over the recent past.
        for window in roll_windows:
            df[f"volume_roll_mean_{window}"] = df["Total Volume"].rolling(window).mean()
            df[f"volume_roll_std_{window}"] = df["Total Volume"].rolling(window).std()
            df[f"speed_roll_mean_{window}"] = df["Avg mph"].rolling(window).mean()
            df[f"speed_roll_std_{window}"] = df["Avg mph"].rolling(window).std()

        # Temporal context.
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek

        # Target: congestion flag for the current interval.
        df["congestion"] = (df["Avg mph"] < speed_threshold).astype(int)

        # Lags and rolling windows leave NaNs at the head of the series.
        df = df.dropna()

        self.df = df
        print(f"Data processed. Shape: {df.shape}")
        print(f"Congestion distribution: {Counter(df['congestion'])}")
        return df

    def create_loaders_from_df(
        self,
        target_col="congestion",
        seq_len=48,
        splits=(0.7, 0.15, 0.15),
        batch_size=64,
        shuffle_train=True,
        num_workers=0,
        handle_imbalance=True,
    ):
        """Chronologically split into train/val/test ``DataLoader``s.

        The split is by time (no shuffling across the boundary) to avoid leaking
        future intervals into training. When ``handle_imbalance`` is set, the
        training loader draws samples with a ``WeightedRandomSampler`` weighted
        by inverse class frequency.
        """
        if self.df is None:
            raise RuntimeError("Call prepare_site_data() before create_loaders_from_df().")

        feature_cols = [c for c in self.df.columns if c != target_col]

        assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
        n_rows = len(self.df)
        if n_rows <= seq_len:
            raise ValueError("Not enough rows for the given seq_len")

        # Sliding windows reduce the usable sample count by seq_len.
        n_samples = n_rows - seq_len
        n_train = int(n_samples * splits[0])
        n_val = int(n_samples * splits[1])
        n_test = n_samples - n_train - n_val
        if n_train <= 0 or n_val < 0 or n_test < 0:
            raise ValueError("Split sizes too small; adjust seq_len or splits")

        # Each split carries seq_len rows of lead-in so its first window is valid.
        train_end_row = n_train + seq_len
        val_start_row = n_train
        val_end_row = n_train + n_val + seq_len
        test_start_row = n_train + n_val

        df_train = self.df.iloc[0:train_end_row].copy()
        df_val = self.df.iloc[val_start_row:val_end_row].copy()
        df_test = self.df.iloc[test_start_row:n_rows].copy()

        train_ds = TrafficDataset(df_train, feature_cols, target_col, seq_len)
        val_ds = TrafficDataset(df_val, feature_cols, target_col, seq_len)
        test_ds = TrafficDataset(df_test, feature_cols, target_col, seq_len)

        if handle_imbalance:
            train_labels = train_ds.targets[seq_len:]
            class_counts = Counter(train_labels.tolist())
            total = sum(class_counts.values())
            class_weights = {cls: total / count for cls, count in class_counts.items()}
            print(f"Class distribution in training: {class_counts}")
            print(f"Class weights: {class_weights}")

            sample_weights = [class_weights[label] for label in train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights, num_samples=len(sample_weights), replacement=True
            )
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers
            )
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers
            )

        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

        info = {
            "n_rows": n_rows,
            "n_samples": n_samples,
            "n_train_samples": len(train_ds),
            "n_val_samples": len(val_ds),
            "n_test_samples": len(test_ds),
            "train_rows_range": (0, train_end_row - 1),
            "val_rows_range": (val_start_row, val_end_row - 1),
            "test_rows_range": (test_start_row, n_rows - 1),
            "feature_cols": feature_cols,
        }

        self.train_loader, self.val_loader, self.test_loader, self.info = (
            train_loader,
            val_loader,
            test_loader,
            info,
        )
        return train_loader, val_loader, test_loader, info
