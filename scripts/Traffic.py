import torch
from torch.utils.data import Dataset, DataLoader

class TrafficDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_length=32):
        """
        df: preprocessed dataframe (chronologically ordered, indexed by datetime)
        feature_cols: list of feature column names
        target_col: column to predict (e.g., "congestion")
        seq_length: how many past steps to use for prediction
        """
        self.features = df[feature_cols].values
        self.targets = df[target_col].values
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        X = self.features[idx:idx+self.seq_length]
        y = self.targets[idx+self.seq_length]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ---------------------------
# Helper to create splits and dataloaders using the class
# ---------------------------
def create_loaders_from_df(df, feature_cols, target_col="congestion",
                           seq_len=48, splits=(0.7, 0.15, 0.15),
                           batch_size=64, shuffle_train=True, num_workers=0):
    """
    Chronologically split df into train/val/test *by sequence samples* and return DataLoaders
    - df must be sorted by datetime index ascending
    - splits are fractions summing roughly to 1.0 (train, val, test)
    """
    # sanity
    assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
    n_rows = len(df)
    if n_rows <= seq_len:
        raise ValueError("Not enough rows for the given seq_len")

    # number of available sequence samples
    n_samples = n_rows - seq_len

    # compute number of sequence samples per split
    n_train = int(n_samples * splits[0])
    n_val = int(n_samples * splits[1])
    n_test = n_samples - n_train - n_val

    if n_train <= 0 or n_val < 0 or n_test < 0:
        raise ValueError("Split sizes too small; adjust seq_len or splits")

    # Convert sequence counts to row slice endpoints (we must include seq_len extra rows)
    # Train rows cover indices [0 : n_train + seq_len)
    train_end_row = n_train + seq_len
    val_start_row = n_train
    val_end_row = n_train + n_val + seq_len
    test_start_row = n_train + n_val
    test_end_row = n_rows  # include to the end

    df_train = df.iloc[0:train_end_row].copy()
    df_val = df.iloc[val_start_row:val_end_row].copy()
    df_test = df.iloc[test_start_row:test_end_row].copy()

    # Build Datasets using the class
    train_ds = TrafficDataset(df_train, feature_cols, target_col=target_col, seq_length=seq_len)
    val_ds = TrafficDataset(df_val, feature_cols, target_col=target_col, seq_length=seq_len)
    test_ds = TrafficDataset(df_test, feature_cols, target_col=target_col, seq_length=seq_len)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    info = {
        "n_rows": n_rows,
        "n_samples": n_samples,
        "n_train_samples": len(train_ds),
        "n_val_samples": len(val_ds),
        "n_test_samples": len(test_ds),
        "train_rows_range": (0, train_end_row-1),
        "val_rows_range": (val_start_row, val_end_row-1),
        "test_rows_range": (test_start_row, test_end_row-1)
    }

    return train_loader, val_loader, test_loader, info

