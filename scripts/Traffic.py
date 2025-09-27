import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd 
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

class TrafficDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, seq_len=16):
        self.seq_len = seq_len
        self.features = df[feature_cols].values
        self.targets = df[target_col].values.astype(int)

    def __len__(self):
        return len(self.features) - self.seq_len

    def __getitem__(self, idx):
        # Sequence of features
        X = self.features[idx:idx+self.seq_len]
        # Label is the target after the sequence
        y = self.targets[idx+self.seq_len]

        # Convert to tensors
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return X, y

class TrafficDataProcessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        
    def _replace_empty(self, x):
        x = x.replace('', 0).astype(float)
        x = x.infer_objects(copy=False)
        return x

    def prepare_site_data(self, speed_threshold=30):
        """
        Prepares traffic dataset for LSTM training for one site.
        """
        print(f"Preprocessing data for site {self.filepath}")

        # Load data
        df = pd.read_parquet(self.filepath)
            
        # add time
        df['time'] = pd.to_datetime(df['Report Date'].str.split('T').str[0] + df['Time Period Ending'], format='%Y-%m-%d%H:%M:%S')
        # set time as index
        df.set_index('time', inplace=True)
        # drop columns
        df.drop(columns=['Report Date', 'Time Period Ending', 'Time Interval'], inplace=True)

        # drop site columns
        df.drop(columns=['Site Name','site_id'], inplace=True)

        df = df.apply(self._replace_empty)

        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Report Date" in df.columns and "Time Period Ending" in df.columns:
                df.index = pd.to_datetime(
                    df["Report Date"].astype(str) + " " + df["Time Period Ending"].astype(str)
                )
            else:
                raise ValueError("No datetime index or date/time columns found.")
        
        df = df.sort_index()
        
        # Keep main features
        df = df[["Total Volume", "Avg mph"]].copy()
        
        # Handle anomalies (0 → NaN → interpolate)
        df["Total Volume"] = df["Total Volume"].replace(0, np.nan).interpolate()
        df["Avg mph"] = df["Avg mph"].replace(0, np.nan).interpolate()
        
        # Create lag features (15min, 30min, 1h)
        for lag in [1, 2, 4]:  # 1=15min, 2=30min, 4=1h
            df[f"volume_lag{lag}"] = df["Total Volume"].shift(lag)
            df[f"speed_lag{lag}"] = df["Avg mph"].shift(lag)
        
        # Rolling features (1h=4, 2h=8, 4h=16)
        for window in [4, 8, 16]:
            df[f"volume_roll_mean_{window}"] = df["Total Volume"].rolling(window).mean()
            df[f"volume_roll_std_{window}"] = df["Total Volume"].rolling(window).std()
            df[f"speed_roll_mean_{window}"] = df["Avg mph"].rolling(window).mean()
            df[f"speed_roll_std_{window}"] = df["Avg mph"].rolling(window).std()
        
        # Time features
        df["hour"] = df.index.hour
        df["dayofweek"] = df.index.dayofweek
        
        # Target: congestion flag
        df["congestion"] = (df["Avg mph"] < speed_threshold).astype(int)
        
        # Drop NaNs from lags/rolls
        df = df.dropna()
        
        self.df = df
        print(f"Data processed. Shape: {df.shape}")
        print(f"Congestion distribution: {Counter(df['congestion'])}")
        
    def create_loaders_from_df(self, target_col="congestion", seq_len=48, 
                              splits=(0.7, 0.15, 0.15), batch_size=64, 
                              shuffle_train=True, num_workers=0, handle_imbalance=True):
        """
        Chronologically split df into train/val/test and return DataLoaders with imbalance handling
        """
        # Features and target
        feature_cols = [c for c in self.df.columns if c != target_col]

        # sanity check
        assert abs(sum(splits) - 1.0) < 1e-6, "splits must sum to 1.0"
        n_rows = len(self.df)
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
        test_end_row = n_rows

        df_train = self.df.iloc[0:train_end_row].copy()
        df_val = self.df.iloc[val_start_row:val_end_row].copy()
        df_test = self.df.iloc[test_start_row:test_end_row].copy()

        # Build Datasets
        train_ds = TrafficDataset(df_train, feature_cols, target_col=target_col, seq_len=seq_len)
        val_ds = TrafficDataset(df_val, feature_cols, target_col=target_col, seq_len=seq_len)
        test_ds = TrafficDataset(df_test, feature_cols, target_col=target_col, seq_len=seq_len)

        # Handle class imbalance for training data
        train_loader = None
        if handle_imbalance:
            # Get labels for training samples
            train_labels = []
            for i in range(len(train_ds)):
                _, label = train_ds[i]
                train_labels.append(label.item())
            
            # Calculate class weights
            class_counts = Counter(train_labels)
            total_samples = sum(class_counts.values())
            class_weights = {cls: total_samples / count for cls, count in class_counts.items()}
            
            print(f"Class distribution in training: {class_counts}")
            print(f"Class weights: {class_weights}")
            
            # Create sample weights
            sample_weights = [class_weights[label] for label in train_labels]
            sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
            
            train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                    sampler=sampler, num_workers=num_workers)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size, 
                                    shuffle=shuffle_train, num_workers=num_workers)

        # Validation and test loaders (no sampling needed)
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
            "test_rows_range": (test_start_row, test_end_row-1),
            "feature_cols": feature_cols
        }

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.info = info
        
        return train_loader, val_loader, test_loader, info


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
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)               # [batch, seq_len, hidden_size]
        out = out[:, -1, :]                 # last timestep → [batch, hidden_size]
        out = self.relu(self.fc1(out))      # → [batch, hidden_size//2]
        out = self.dropout(out)
        out = self.fc2(out)                 # → [batch, num_classes]
        return out



class ModelTrainer:
    def __init__(self, model, class_weights=None):
        self.model = model
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.train_f1_scores = []
        self.val_f1_scores = []
        
        # Handle class imbalance with weighted loss
        if class_weights is not None:
            weight_tensor = torch.tensor([class_weights[0], class_weights[1]], dtype=torch.float32)
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5)

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        y_true, y_pred = [], []

        for features, labels in train_loader:
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            y_true.extend(labels.numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).detach().numpy())

        avg_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return avg_loss, accuracy, f1

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                y_true.extend(labels.numpy())
                y_pred.extend(torch.argmax(outputs, dim=1).numpy())

        avg_loss = running_loss / len(val_loader)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return avg_loss, accuracy, f1

    def train_model(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting training...")
        print(f"Epochs: {epochs}, Early stopping patience: {early_stopping_patience}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.train_f1_scores.append(train_f1)
            self.val_f1_scores.append(val_f1)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_traffic_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_traffic_model.pth'))
        print("Training completed. Best model loaded.")
        
        return self.model
    




class ResultsPlotter:
    def __init__(self, trainer, test_loader=None):
        self.trainer = trainer
        self.test_loader = test_loader
    
    def plot_training_history(self, figsize=(15, 5)):
        """Plot training history"""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss plot
        axes[0].plot(self.trainer.train_losses, label='Train Loss', color='blue')
        axes[0].plot(self.trainer.val_losses, label='Val Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.trainer.train_accuracies, label='Train Accuracy', color='blue')
        axes[1].plot(self.trainer.val_accuracies, label='Val Accuracy', color='red')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # F1 Score plot
        axes[2].plot(self.trainer.train_f1_scores, label='Train F1', color='blue')
        axes[2].plot(self.trainer.val_f1_scores, label='Val F1', color='red')
        axes[2].set_title('Training and Validation F1 Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()

        fig.savefig('../Plots/TrainingHistory.png')
        plt.show()