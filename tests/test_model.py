"""Tests for the TrafficNet LSTM forward pass."""

import torch

from traffic_congestion import TrafficNet


def test_forward_output_shape():
    batch, seq_len, input_size = 8, 16, 22
    model = TrafficNet(input_size=input_size)
    out = model(torch.randn(batch, seq_len, input_size))
    assert out.shape == (batch, 2)


def test_forward_respects_num_classes():
    model = TrafficNet(input_size=10, hidden_size=32, num_layers=1, num_classes=3)
    out = model(torch.randn(4, 6, 10))
    assert out.shape == (4, 3)


def test_forward_handles_single_sample():
    model = TrafficNet(input_size=22)
    out = model(torch.randn(1, 16, 22))
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()


def test_backward_pass_produces_gradients():
    model = TrafficNet(input_size=22)
    out = model(torch.randn(4, 16, 22))
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
