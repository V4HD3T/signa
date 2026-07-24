"""Shape and wiring tests for the three classifiers.

No training -- these pin the contract every model shares: a batch of fixed-length
landmark frames goes in, one logit per class comes out, gradients flow, and a
single clip works as well as a batch (the demo and report both call the model on
one clip at a time). A model that silently produced the wrong output width would
train and even reach a plausible loss before anything looked wrong.
"""

import pytest
import torch

from signa import models
from signa.config import FRAME_DIM, Config


@pytest.mark.parametrize("name", ["bilstm", "transformer", "tcn"])
def test_model_maps_a_batch_to_one_logit_per_class(name):
    model = models.build(Config(model=name), num_classes=26)
    out = model(torch.randn(4, 48, FRAME_DIM))
    assert out.shape == (4, 26)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["bilstm", "transformer", "tcn"])
def test_model_runs_on_a_single_clip(name):
    # demo.classify and report both feed one clip with a batch dim of 1.
    model = models.build(Config(model=name), num_classes=10).eval()
    out = model(torch.randn(1, 48, FRAME_DIM))
    assert out.shape == (1, 10)


@pytest.mark.parametrize("name", ["bilstm", "transformer", "tcn"])
def test_gradients_reach_every_parameter(name):
    model = models.build(Config(model=name), num_classes=5)
    out = model(torch.randn(2, 48, FRAME_DIM))
    out.sum().backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"{name}: no gradient for {missing}"


@pytest.mark.parametrize("name", ["bilstm", "transformer", "tcn"])
def test_model_tolerates_a_variable_clip_length(name):
    # frames is configurable; a model must not hard-code 48.
    model = models.build(Config(model=name), num_classes=8).eval()
    assert model(torch.randn(1, 30, FRAME_DIM)).shape == (1, 8)
    assert model(torch.randn(1, 64, FRAME_DIM)).shape == (1, 8)


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        models.build(Config(model="lstm-typo"), num_classes=3)
