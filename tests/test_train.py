"""Smoke test: the training loop runs end-to-end for a couple of epochs."""

import math
import os

import torch

from traffic_congestion import ModelTrainer, TrafficDataProcessor, TrafficNet


def test_train_two_epochs_runs(raw_parquet, tmp_path):
    proc = TrafficDataProcessor(raw_parquet)
    proc.prepare_site_data()
    train_loader, val_loader, _, info = proc.create_loaders_from_df(seq_len=8, batch_size=16)

    model = TrafficNet(input_size=len(info["feature_cols"]))
    ckpt = tmp_path / "best.pth"
    trainer = ModelTrainer(model, checkpoint_path=str(ckpt))

    returned = trainer.train_model(
        train_loader, val_loader, epochs=2, early_stopping_patience=5
    )

    # Two epochs of metrics were recorded and the loop returned the model.
    assert len(trainer.train_losses) == 2
    assert len(trainer.val_losses) == 2
    assert isinstance(returned, TrafficNet)

    # Losses are finite real numbers.
    for loss in trainer.train_losses + trainer.val_losses:
        assert math.isfinite(loss)

    # The best checkpoint was written and reloads into the model.
    assert os.path.exists(ckpt)
    model.load_state_dict(torch.load(ckpt))


def test_class_weights_build_weighted_loss(raw_parquet, tmp_path):
    proc = TrafficDataProcessor(raw_parquet)
    proc.prepare_site_data()
    _, _, _, info = proc.create_loaders_from_df(seq_len=8, batch_size=16)

    model = TrafficNet(input_size=len(info["feature_cols"]))
    trainer = ModelTrainer(
        model, class_weights={0: 1.0, 1: 5.0}, checkpoint_path=str(tmp_path / "m.pth")
    )
    assert trainer.criterion.weight is not None
    assert torch.allclose(trainer.criterion.weight, torch.tensor([1.0, 5.0]))
