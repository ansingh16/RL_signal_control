"""Training loop for the congestion LSTM.

``ModelTrainer`` wraps the optimiser, scheduler, and weighted loss, records
per-epoch metrics, and checkpoints the best model by validation loss with early
stopping.
"""

import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score


class ModelTrainer:
    def __init__(
        self,
        model,
        class_weights=None,
        lr=1e-4,
        weight_decay=1e-4,
        scheduler_patience=5,
        checkpoint_path="models/best_traffic_model.pth",
    ):
        self.model = model
        self.checkpoint_path = checkpoint_path

        self.train_losses, self.val_losses = [], []
        self.train_accuracies, self.val_accuracies = [], []
        self.train_f1_scores, self.val_f1_scores = [], []

        if class_weights is not None:
            weight_tensor = torch.tensor(
                [class_weights[0], class_weights[1]], dtype=torch.float32
            )
            self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, "min", patience=scheduler_patience
        )

    def _run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        running_loss = 0.0
        y_true, y_pred = [], []

        torch.set_grad_enabled(train)
        for features, labels in loader:
            if train:
                self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            if train:
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            y_true.extend(labels.numpy())
            y_pred.extend(torch.argmax(outputs, dim=1).detach().numpy())
        torch.set_grad_enabled(True)

        avg_loss = running_loss / len(loader)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="weighted")
        return avg_loss, accuracy, f1

    def train_epoch(self, train_loader):
        return self._run_epoch(train_loader, train=True)

    def validate(self, val_loader):
        return self._run_epoch(val_loader, train=False)

    def train_model(self, train_loader, val_loader, epochs=50, early_stopping_patience=10):
        best_val_loss = float("inf")
        patience_counter = 0
        ckpt_dir = os.path.dirname(self.checkpoint_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)

        print(f"Starting training... Epochs: {epochs}, patience: {early_stopping_patience}")
        for epoch in range(epochs):
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self.validate(val_loader)
            self.scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.train_f1_scores.append(train_f1)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.val_f1_scores.append(val_f1)

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
            else:
                patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        self.model.load_state_dict(torch.load(self.checkpoint_path))
        print("Training completed. Best model loaded.")
        return self.model
