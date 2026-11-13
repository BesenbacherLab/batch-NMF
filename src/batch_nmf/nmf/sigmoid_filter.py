import warnings

import torch
import numpy as np
import torch.nn as nn
from batch_nmf.nmf.base import baseNMF

class sigmoidfilterNMF(baseNMF):
    """Sigmoid filter NMF (sfNMF): batch filters made of two sigmoid functions.

    The same model as bNMF (see `batchNMF`), but the filter of each batch is the product of an
    increasing sigmoid (loss of short fragments) and a decreasing sigmoid (loss of long
    fragments) of the fragment length `L`:

        f(L) = L^n_i / (h_i^n_i + L^n_i) × h_d^n_d / (h_d^n_d + L^n_d)

    with four parameters per batch: the lengths `h_i` and `h_d` where the two sigmoids are 0.5,
    and their steepness `n_i` and `n_d`. Every batch has its own filter (there is no reference
    batch), and the filters are always below 1.

    With `H` given, the signatures are fixed and only the weights and filters are fitted, as in
    `batch-nmf predict`.

    Args:
        target: Input matrix with one row per sample, rows summing to 1.
        B: One-hot matrix of the batch of each sample (samples × batches), for example from
            `read_batch_file`.
        n_signatures: Number of signatures.
        H: Fixed signatures (signatures × lengths). If given, only the weights and filters are fitted.
        F: Fixed filters (batches × lengths). If given, they are not fitted.
        filter_regularization: Not used. The smoothness penalty of bNMF has no effect on sigmoid
            filters, which are smooth already; a positive value gives a warning.
        scaler_regularization: Not used. The penalty of bNMF fixes the overall level of the
            filters; sigmoid filters are always below 1, so it would always be 0. A positive
            value gives a warning.
        volume_regularization: If positive, the strength of a penalty on the volume of the
            signatures, `log det(H Hᵀ + I)`.
        min_length: Fragment length of the first column. The columns are then taken to be the
            consecutive lengths `min_length`, `min_length + 1`, ... Ignored if `lengths` is given.
        lengths: Fragment length of each column of `target`, for example from
            `get_fragment_lengths`. One of `lengths` and `min_length` is required.
    """

    def __init__(
        self,
        target: np.ndarray,
        B: np.ndarray,
        n_signatures: int = 2,
        H: np.ndarray | None = None,
        F: np.ndarray | None = None,
        filter_regularization: float = -1,
        scaler_regularization: float = -1,
        volume_regularization: float = -1,
        min_length: int | None = None,
        lengths: np.ndarray | None = None,
    ):
        super().__init__(target, n_signatures, H, volume_regularization=volume_regularization)
        self.name = "sigmoid_filter_NMF"
        if filter_regularization > 0:
            warnings.warn(
                "filter_regularization has no effect for sfNMF (the sigmoid filters are already smooth) "
                "and is ignored.",
                UserWarning,
                stacklevel=2,
            )
        if scaler_regularization > 0:
            warnings.warn(
                "scaler_regularization has no effect for sfNMF (the sigmoid filters are always below 1) "
                "and is ignored.",
                UserWarning,
                stacklevel=2,
            )
        self.B = torch.tensor(B, dtype=self.dtype, device=self.device)
        self.n_batch = B.shape[1]
        if lengths is None:
            if min_length is None:
                raise ValueError("sigmoidfilterNMF needs the fragment lengths: give `lengths` or `min_length`")
            lengths = np.arange(min_length, min_length + self.n_types)
        lengths = np.asarray(lengths, dtype=float)
        assert lengths.shape == (self.n_types,), \
            f"lengths must have one entry per column of the input ({self.n_types}), got shape {lengths.shape}"
        assert (lengths > 0).all(), "fragment lengths must be positive"
        self.bp = torch.tensor(lengths, dtype=self.dtype, device=self.device)

        if not F is None:
            self.knownF = True
            self.F = torch.tensor(F, dtype=self.dtype, device=self.device)
        else:
            self.knownF = False
            self.F_shape = (self.n_batch, self.n_types)
            
            self.DS_shape = (self.n_batch, 1)
            self.DS_trans = nn.Softplus()
            
            self.IS_shape = (self.n_batch, 1)
            self.IS_trans = nn.Softplus()

        self.rand_init()
        
    def rand_init(self):
        super().rand_init()
        if not self.knownF:
            self.DS1 = self.init_par(self.DS_shape, 50.0)
            self.DS2 = self.init_par(self.DS_shape, 10.0)
            self.IS1 = self.init_par(self.IS_shape, 5.0)
            self.IS2 = self.init_par(self.IS_shape, 10.0)

    def get_F(self):
        """The filters as a torch tensor: one row per batch."""
        if self.knownF:
            return self.F
        else:
            F = torch.ones(self.F_shape, dtype=self.dtype, device=self.device)
            bp = self.bp
            
            #apply decreasing sigmoid
            H = self.DS_trans(self.DS1)*10
            N = self.DS_trans(self.DS2/5)
            DS2_F = (H**N)/((H**N) + (bp**N))
            F *= DS2_F
            
            #apply increasing sigmoid
            H = self.IS_trans(self.IS1)*10
            N = - self.IS_trans(self.IS2/5)
            IS2_F = (H**N)/((H**N) + (bp**N))
            F *= IS2_F

            return F

    def forward(self):
        X = (self.get_W() @ self.get_H()) * (self.B @ self.get_F())
        return X / X.sum(1, keepdims=True)

    def get_parameters(self):
        """The fitted parameters as numpy arrays: `(W, H, F)`, or `(W, F)` when `H` was given."""
        if self.predict:
            return (self.get_W().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())
        else:
            return (self.get_W().cpu().detach().numpy(), self.get_H().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())

