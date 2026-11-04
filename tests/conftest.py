"""Configuration for the pytest test suite."""

import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Seed torch before every test.

    pytest-randomly reseeds `random` and `numpy` but not torch, so without this
    the random initialisations in the NMF models differ between runs.
    """
    torch.manual_seed(1234)
