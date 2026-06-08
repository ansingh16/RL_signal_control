"""LSTM classifier for next-interval congestion prediction."""

import torch.nn as nn


class TrafficNet(nn.Module):
    """LSTM over a window of traffic features, classifying the next interval.

    The final LSTM hidden state is passed through a two-layer head with ReLU and
    dropout to produce class logits (congested / not congested by default).
    """

    def __init__(self, input_size, hidden_size=64, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, seq_len, input_size]
        out, _ = self.lstm(x)          # [batch, seq_len, hidden_size]
        out = out[:, -1, :]            # last timestep -> [batch, hidden_size]
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)            # [batch, num_classes]
        return out
