"""Tests for the `utils` module."""

import io

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from batch_nmf.nmf.batch import batchNMF
from batch_nmf.nmf.plain import plainNMF
from batch_nmf.utils import LOG_COLUMNS, batch_balanced_weights


def test_batch_balanced_weights():
    # batch 0 has 5 samples, batch 1 has 1 sample
    B = torch.tensor([[1.0, 0.0]] * 5 + [[0.0, 1.0]], dtype=torch.double)
    w = batch_balanced_weights(B).numpy()
    assert_allclose(w, [0.2] * 5 + [1.0])
    # every batch contributes the same total weight
    assert_allclose(B.numpy().T @ w, [1.0, 1.0])


def _data():
    W = np.array([[0.9, 0.1], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
    H = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]])
    H = H / H.sum(1, keepdims=True)
    return W @ H


@pytest.mark.parametrize(
    "make_model",
    [
        lambda X: plainNMF(X, 2),
        # scaler_regularization > 0 makes the penalty a tensor
        lambda X: batchNMF(X, np.array([[1, 0]] * 3 + [[0, 1]] * 2), 2, scaler_regularization=1.0),
    ],
)
def test_logfile_header_matches_rows(make_model):
    logfile = io.StringIO()
    make_model(_data()).fit(n_epochs=3, restarts=0, logfile=logfile)
    lines = [line.split("\t") for line in logfile.getvalue().splitlines()]
    header, rows = lines[0], lines[1:]
    assert header == list(LOG_COLUMNS)
    # rows are written at epoch 1, 5001, ... and at the last epoch (here epoch 2)
    assert [(row[header.index("run")], row[header.index("epoch")]) for row in rows] == [("final", "1"), ("final", "2")]
    for row in rows:
        assert len(row) == len(header)
        for col in ("loss", "loss_no_penalty", "best_loss", "best_loss_no_penalty", "penalty", "current_lr", "time"):
            float(row[header.index(col)])  # plain numbers, not "tensor(...)" or "[...]"


def test_weighted_loss_runs():
    X = _data()
    B = np.array([[1, 0]] * 4 + [[0, 1]])
    batchNMF(X, B, 2).fit(n_epochs=3, restarts=0, weighted_loss=True)


def test_max_parameter_change():
    from batch_nmf.utils import max_parameter_change

    old = [np.array([0.2, 0.5]), np.array([10.0])]
    new = [np.array([0.25, 0.5]), np.array([10.5])]
    # absolute for values <= 1 (0.05), relative for larger values (0.5 / 10 = 0.05)
    assert max_parameter_change(old, new) == pytest.approx(0.05)
    assert max_parameter_change(old, [a.copy() for a in old]) == 0.0


def test_get_best_model_stops_when_converged():
    from batch_nmf.utils import get_best_model, make_loss_fn

    m = plainNMF(_data(), 2)
    epochs = []
    get_best_model(m, m.X, 1e30, 1e30, None, make_loss_fn(m), "final", n_epochs=50000, lr=0.1,
                   verbosity=0, tol=1e-4, check_every=100, callback=lambda epoch, **_: epochs.append(epoch))
    assert epochs[-1] + 1 < 50000


def test_get_best_model_tol_zero_runs_all_epochs():
    from batch_nmf.utils import get_best_model, make_loss_fn

    m = plainNMF(_data(), 2)
    epochs = []
    get_best_model(m, m.X, 1e30, 1e30, None, make_loss_fn(m), "final", n_epochs=500, lr=0.1,
                   verbosity=0, tol=0, check_every=100, callback=lambda epoch, **_: epochs.append(epoch))
    assert [e + 1 for e in epochs] == [100, 200, 300, 400, 500]


@pytest.mark.parametrize(("restarts", "expected_final_lr"), [(3, 0.01), (0, 0.5)])
def test_train_model_epoch_caps_and_final_lr(monkeypatch, restarts, expected_final_lr):
    """Restarts get n_epochs_restart, the final run n_epochs, and the final run continues
    at the learning rate where the best restart stopped."""
    from copy import deepcopy

    from batch_nmf import utils

    results = {"initial_0": (5.0, 0.1), "initial_1": (3.0, 0.01), "initial_2": (4.0, 0.2), "final": (3.0, 0.001)}
    calls = []

    def fake_get_best_model(model, target, best_loss, best_loss_np, best_model, loss_fn, traintype, **kwargs):
        calls.append((traintype, kwargs["lr"], kwargs["n_epochs"]))
        loss, end_lr = results[traintype]
        if loss < best_loss:
            return deepcopy(model.state_dict()), loss, loss, end_lr
        return best_model, best_loss, best_loss_np, end_lr

    monkeypatch.setattr(utils, "get_best_model", fake_get_best_model)
    m = plainNMF(_data(), 2)
    utils.train_model(m, m.X, n_epochs=7, restarts=restarts, n_epochs_restart=11, lr=0.5, verbosity=0)
    assert calls == [(f"initial_{i}", 0.5, 11) for i in range(restarts)] + [("final", expected_final_lr, 7)]


def test_log_has_row_for_last_epoch():
    logfile = io.StringIO()
    plainNMF(_data(), 2).fit(n_epochs=5, n_epochs_restart=8, restarts=1, tol=0, logfile=logfile)
    rows = [line.split("\t") for line in logfile.getvalue().splitlines()[1:]]
    last_epoch = {}
    for row in rows:
        last_epoch[row[LOG_COLUMNS.index("run")]] = int(row[LOG_COLUMNS.index("epoch")])
    assert last_epoch == {"initial_0": 7, "final": 4}  # epochs are counted from 0
