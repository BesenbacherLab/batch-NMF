import numpy as np
from numpy.testing import assert_allclose
from batch_nmf.nmf.plain import plainNMF
from batch_nmf.nmf.batch import batchNMF
from batch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF

import pytest


def test_nmf_true_reconstruction():
    # Test that the fit is not too far away from an exact solution
    # (by construction)

    W = np.array([[0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
    H = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]])
    H = H/H.sum(1, keepdims=True)
    X = W @ H

    m = plainNMF(X, W.shape[1])
    res, loss, loss_np, penalty = m.fit(n_epochs=10000)
    W_est, H_est = res
    X_est = W_est @ H_est

    assert_allclose(X, X_est)


def test_nmf_predict():
    # Test that the fit is not too far away from an exact solution
    # (by construction)

    W = np.array([[0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
    H = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]])
    H = H/H.sum(1, keepdims=True)
    X = W @ H

    m = plainNMF(X, W.shape[1], H)
    res, loss, loss_np, penalty = m.fit(n_epochs=10000)

    W_est = res
    X_est = W_est @ H

    assert_allclose(W, W_est)
    assert_allclose(X, X_est)


def test_batchNMF_true_reconstruction():
    # Test that the fit is not too far away from an exact solution
    # (by construction)

    W = np.array([[0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9], [0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
    H = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]])
    B = np.array([[1, 0], [1, 0], [1, 0],[ 1, 0], [1, 0], [0, 1], [0, 1],[0, 1],[0, 1],[0, 1]])
    F = np.array([[0.9, 0.97, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95],
                 [0.1, 0.3, 0.6, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8]])
    H = H/H.sum(1, keepdims=True)
    X = W @ H
    BF = B @ F
    Y = X * BF
    Y = Y/Y.sum(1, keepdims=True)

    m = batchNMF(Y, B, W.shape[1])
    res, loss, loss_np, penalty = m.fit(n_epochs=50000)
    W_est, H_est, F_est = res
    X_est = W_est @ H_est
    Y_est = X_est * (B @ F_est)

    W_est= W_est/W_est.sum(1, keepdims=True)
    Y_est = Y_est/Y_est.sum(1, keepdims=True)

    assert_allclose(Y, Y_est)


def test_batchNMF_predict():
    # Test that the fit is not too far away from an exact solution
    # (by construction)

    W = np.array([[0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
    H = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]])
    B = np.array([[1], [1], [1],[ 1], [1]])
    F = np.array([[0.1, 0.3, 0.6, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8]])
    H = H/H.sum(1, keepdims=True)
    X = W @ H
    BF = B @ F
    Y = X * BF
    Y = Y/Y.sum(1, keepdims=True)

    m = batchNMF(Y, B, W.shape[1], H)
    res, loss, loss_np, penalty = m.fit(n_epochs=10000)
    W_est, F_est = res
    X_est = W_est @ H
    Y_est = X_est * (B @ F_est)

    W_est = W_est/W_est.sum(1, keepdims=True)
    Y_est = Y_est/Y_est.sum(1, keepdims=True)

    assert_allclose(Y, Y_est, rtol=0.01)


def test_sigmoidfilterNMF_lengths():
    X = np.random.default_rng(0).random((4, 12)) + 0.1
    X = X / X.sum(1, keepdims=True)
    B = np.ones((4, 1))
    # the lengths are required
    with pytest.raises(ValueError, match="fragment lengths"):
        sigmoidfilterNMF(X, B, 2)
    # min_length: consecutive lengths starting there
    assert_allclose(sigmoidfilterNMF(X, B, 2, min_length=100).bp.numpy(), np.arange(100, 112))
    # explicit lengths win over min_length
    lengths = np.arange(100, 124, 2)
    m = sigmoidfilterNMF(X, B, 2, min_length=30, lengths=lengths)
    assert_allclose(m.bp.numpy(), lengths)
    assert m.get_F().shape == (1, 12)
    with pytest.raises(AssertionError):
        sigmoidfilterNMF(X, B, 2, lengths=np.arange(100, 105))


@pytest.mark.parametrize("option", ["filter_regularization", "scaler_regularization"])
def test_sigmoidfilterNMF_ignores_bnmf_regularization(option):
    """The filter and scaler penalties of bNMF have no effect for sfNMF: they give a warning
    and fitting still works."""
    X = np.random.default_rng(0).random((6, 12)) + 0.1
    X = X / X.sum(1, keepdims=True)
    B = np.array([[1, 0]] * 3 + [[0, 1]] * 3)
    with pytest.warns(UserWarning, match=f"{option} has no effect for sfNMF"):
        m = sigmoidfilterNMF(X, B, 2, lengths=np.arange(100, 112), **{option: 1.0})
    m.fit(n_epochs=3, restarts=0)
    assert float(m.get_penalty()) == 0.0


def test_fit_defaults_match_command_line():
    """The Python API and the batch-nmf command line tool use the same fitting defaults."""
    import inspect

    from batch_nmf.cli import get_parser
    from batch_nmf.nmf.base import baseNMF

    fit = {k: p.default for k, p in inspect.signature(baseNMF.fit).parameters.items()}
    cli = vars(get_parser().parse_args(["fit", "--input_matrix", "x", "--signature_file", "s", "--exposure_file", "e"]))
    pairs = {"n_epochs": "n_epochs", "lr": "learning_rate", "restarts": "n_restarts", "loss": "loss",
             "min_lr_rate": "min_learning_rate", "lr_decrease_factor": "lr_decrease_factor",
             "n_epochs_restart": "n_epochs_restart", "tol": "tol"}
    assert {k: fit[k] for k in pairs} == {k: cli[v] for k, v in pairs.items()}
