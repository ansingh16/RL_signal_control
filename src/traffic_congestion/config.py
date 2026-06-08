"""Typed configuration for the data pipeline, model, and training loop.

Defaults reproduce the behaviour the modules used before they were
parameterised, so ``DataConfig()``/``ModelConfig()``/``TrainConfig()`` give a
working baseline out of the box.
"""

from dataclasses import dataclass, field


@dataclass
class DataConfig:
    speed_threshold: int = 30           # mph below which an interval is congested
    seq_len: int = 16                   # window length fed to the LSTM (16 = 4h)
    batch_size: int = 64
    splits: tuple = (0.7, 0.15, 0.15)   # chronological train/val/test fractions
    handle_imbalance: bool = True       # weighted sampling on the train loader
    lags: tuple = (1, 2, 4)             # lag offsets in 15-min steps
    roll_windows: tuple = (4, 8, 16)    # rolling-window sizes in 15-min steps


@dataclass
class ModelConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 2


@dataclass
class TrainConfig:
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 50
    early_stopping_patience: int = 10
    scheduler_patience: int = 5
    checkpoint_path: str = "models/best_traffic_model.pth"
    use_class_weights: bool = True      # weighted cross-entropy loss


@dataclass
class Config:
    """Bundle of the three sub-configs for convenience."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
