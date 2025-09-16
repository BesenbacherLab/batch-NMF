import numpy as np
from numpy.testing import assert_allclose
from batch_nmf.nmf.plain import plainNMF
from batch_nmf.nmf.batch import batchNMF
from batch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF
from batch_nmf.nmf.scipy_nmf import fit_scipy_nmf

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
    #res, loss, penalty = libname.fit_nmf(X, W.shape[1], n_epochs=10000)
    W_est, H_est = res
    X_est = W_est @ H_est
    
    print('============= PLAIN W ==============')
    print(W)
    print(W_est)
    print(W_est.sum(axis=0))
    print(W_est.sum(axis=1))
    print('============= PLAIN H ==============')
    print(H)
    print(H_est)
    print(H_est.sum(axis=0))
    print(H_est.sum(axis=1))
    print('============= PLAIN X ==============')
    print(X)
    print(X_est)
    print(X_est.sum(axis=0))
    print(X_est.sum(axis=1))
    
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
    
    print('============= PLAIN W ==============')
    print(W)
    print(W_est)
    print(W_est.sum(axis=0))
    print(W_est.sum(axis=1))

    assert_allclose(W, W_est)


    print('============= PLAIN X ==============')
    print(X)
    print(X_est)
    print(X_est.sum(axis=0))
    print(X_est.sum(axis=1))
    
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
    print(BF)
    Y = X * BF
    Y = Y/Y.sum(1, keepdims=True)

    m = batchNMF(Y, B, W.shape[1])
    res, loss, loss_np, penalty = m.fit(n_epochs=50000)
    print(res)
    print(loss)
    W_est, H_est, F_est = res
    X_est = W_est @ H_est
    Y_est = X_est * (B @ F_est)
    
    W_est= W_est/W_est.sum(1, keepdims=True)
    Y_est = Y_est/Y_est.sum(1, keepdims=True)

    print('=============  NMF Filter W ==============')
    print(W)
    print(W_est)
    print(W_est.sum(axis=0))
    print(W_est.sum(axis=1))
    print(W_est.sum(axis=0))
    print(W_est.sum(axis=1))

    print('=============  NMF Filter H ==============')
    print(H)
    print(H_est)
    print(H_est.sum(axis=0))
    print(H_est.sum(axis=1))
    print('=============  NMF Filter F ==============')
    print(F)
    print(F_est)
    print('=============  NMF Filter X ==============')
    print(X)
    print(X_est)
    print('=============  NMF Filter Y ==============')
    print(Y)
    print(Y_est)
    
    #assert_allclose(X, X_est)
    assert_allclose(Y, Y_est)


# def test_sigmoidfilterNMF_true_reconstruction():
#     # Test that the fit is not too far away from an exact solution
#     # (by construction)

#     W = np.array([[0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9], [0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
#     H = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]])
#     B = np.array([[1, 0], [1, 0], [1, 0],[ 1, 0], [1, 0], [0, 1], [0, 1],[0, 1],[0, 1],[0, 1]])
#     F = np.array([[0.9, 0.97, 0.99, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95],
#                  [0.1, 0.3, 0.6, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8]])

#     H = H/H.sum(1, keepdims=True)
#     X = W @ H
#     BF = B @ F
#     print(BF)
#     Y = X * BF
#     Y = Y/Y.sum(1, keepdims=True)

#     m = sigmoidfilterNMF(Y, B, W.shape[1], min_length=80)
#     print(m.DS1.mean())
#     print(m.DS2.mean())
#     print(m.IS1.mean())
#     print(m.IS2.mean())
#     print(m.H.mean())
#     print(m.W.mean())

#     res, loss, loss_np, penalty = m.fit(n_epochs=100000, lr=0.5, restarts=10, loss = 'kullback-leibler') #min_lr_rate = 1e-5
#     print(m.DS1.mean())
#     print(m.DS2.mean())
#     print(m.IS1.mean())
#     print(m.IS2.mean())
#     print(m.H.mean())
#     print(m.W.mean())

#     print(res)
#     print(loss)
#     W_est, H_est, F_est = res
#     X_est = W_est @ H_est
#     Y_est = X_est * (B @ F_est)
    
#     W_est= W_est/W_est.sum(1, keepdims=True)
#     Y_est = Y_est/Y_est.sum(1, keepdims=True)

#     print('=============  NMF Filter W ==============')
#     print(W)
#     print(W_est)
#     print(W_est.sum(axis=0))
#     print(W_est.sum(axis=1))

#     print('=============  NMF Filter H ==============')
#     print(H)
#     print(H_est)
#     print(H_est.sum(axis=0))
#     print(H_est.sum(axis=1))
#     print('=============  NMF Filter F ==============')
#     print(F)
#     print(F_est)
#     print('=============  NMF Filter X ==============')
#     print(X)
#     print(X_est)
#     print('=============  NMF Filter Y ==============')
#     print(Y)
#     print(Y_est)
    
#     #assert_allclose(X, X_est)
#     assert_allclose(Y, Y_est)



def test_batchNMF_predict():
    # Test that the fit is not too far away from an exact solution
    # (by construction)

    W = np.array([[0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
    H = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]])
    #B = np.array([[1, 0], [1, 0], [1, 0],[ 1, 0], [1, 0], [0, 1], [0, 1],[0, 1],[0, 1],[0, 1]])
    B = np.array([[1], [1], [1],[ 1], [1]])
    F = np.array([[0.1, 0.3, 0.6, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8]])
    H = H/H.sum(1, keepdims=True)
    X = W @ H
    BF = B @ F
    print(BF)
    Y = X * BF
    Y = Y/Y.sum(1, keepdims=True)

    m = batchNMF(Y, B, W.shape[1], H)
    res, loss, loss_np, penalty = m.fit(n_epochs=10000)
    print(loss)
    W_est, F_est = res
    X_est = W_est @ H
    Y_est = X_est * (B @ F_est)

    print(B)
    W_est = W_est/W_est.sum(1, keepdims=True)
    Y_est = Y_est/Y_est.sum(1, keepdims=True)

    print('=============  NMF Filter W ==============')
    print(W)
    print(W_est)
    print(W_est.sum(axis=0))
    print(W_est.sum(axis=1))
    
    print('=============  NMF Filter F ==============')
    print(F)
    print(F_est)
    print('=============  NMF Filter X ==============')
    print(X)
    print(X_est)
    print('=============  NMF Filter Y ==============')
    print(Y)
    print(Y_est)
    
    #assert_allclose(X, X_est)
    assert_allclose(Y, Y_est, rtol=0.01)


# def test_sigmoid_filter_nmf_predict():
#     # Test that the fit is not too far away from an exact solution
#     # (by construction)

#     W = np.array([[0.9, 0.1], [0.5,0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9]])
#     H = np.array([[1,2,3,4,5,6,7,8,9,10,11,12],[12,11,10,9,8,7,6,5,4,3,2,1]])
#     #B = np.array([[1, 0], [1, 0], [1, 0],[ 1, 0], [1, 0], [0, 1], [0, 1],[0, 1],[0, 1],[0, 1]])
#     B = np.array([[1], [1], [1],[ 1], [1]])
#     F = np.array([[0.1, 0.3, 0.6, 0.9, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.8]])
#     H = H/H.sum(1, keepdims=True)
#     X = W @ H
#     BF = B @ F
#     print(BF)
#     Y = X * BF
#     Y = Y/Y.sum(1, keepdims=True)
#     m = sigmoidfilterNMF(Y, B, W.shape[1], H)
#     res, loss, loss_np, penalty = m.fit(n_epochs=10000)
#     print(loss)
#     W_est, F_est = res
#     X_est = W_est @ H
#     Y_est = X_est * (B @ F_est)

#     W_est = W_est/W_est.sum(1, keepdims=True)
#     Y_est = Y_est/Y_est.sum(1, keepdims=True)

#     print('=============  NMF Filter W ==============')
#     print(W)
#     print(W_est)
#     print(W_est.sum(axis=0))
#     print(W_est.sum(axis=1))
    
#     print('=============  NMF Filter F ==============')
#     print(F)
#     print(F_est)
#     print('=============  NMF Filter X ==============')
#     print(X)
#     print(X_est)
#     print('=============  NMF Filter Y ==============')
#     print(Y)
#     print(Y_est)
    
#     #assert_allclose(X, X_est)
#     assert_allclose(Y, Y_est)


