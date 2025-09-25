import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix


class TrafficDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=16):
        self.seq_len = seq_len
        self.features = df[feature_cols].values
        self.targets = df[target_col].values.astype(int)  # ensure int for classification

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Sequence of features
        X = self.features[idx:idx+self.seq_len]
        # Label is the target after the sequence
        y = self.targets[idx+self.seq_len]

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)  # (scalar, not vector)

        return X, y


class TrafficNet(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super(TrafficNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout
        )

        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)               # [batch, seq_len, hidden_size]
        out = out[:, -1, :]                 # last timestep → [batch, hidden_size]
        out = self.fc(out)                  # → [batch, num_classes]
        return out



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

 
    train_end_row = n_train + seq_len
    val_start_row = n_train
    val_end_row = n_train + n_val + seq_len
    test_start_row = n_train + n_val
    test_end_row = n_rows  # include to the end

    df_train = df.iloc[0:train_end_row].copy()
    df_val = df.iloc[val_start_row:val_end_row].copy()
    df_test = df.iloc[test_start_row:test_end_row].copy()

    # Build Datasets using the class
    train_ds = TrafficDataset(df_train, feature_cols, target_col=target_col, seq_len=seq_len)
    val_ds = TrafficDataset(df_val, feature_cols, target_col=target_col, seq_len=seq_len)
    test_ds = TrafficDataset(df_test, feature_cols, target_col=target_col, seq_len=seq_len)

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



def train_model(model, train_loader, val_loader, epochs=20):

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()  

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            y_true.extend(labels.numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).detach().numpy())

        train_acc = accuracy_score(y_true, y_pred)

        # Validation
        model.eval()
        val_true, val_pred = [], []
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                val_true.extend(labels.numpy())
                val_pred.extend(torch.argmax(outputs, dim=1).numpy())

        val_acc = accuracy_score(val_true, val_pred)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} "
              f"| Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    return model

