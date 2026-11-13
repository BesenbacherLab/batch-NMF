import numpy as np
import torch
from batch_nmf.nmf.base import baseNMF

class batchNMF(baseNMF):
    """Batch NMF (bNMF): NMF with a free filter for each batch.

    Models the normalised fragment length profiles `X` (one row per sample) as

        X ≈ rownorm((W H) ∘ (B F))

    where `W` holds the weights of the signatures in each sample (rows sum to 1), `H` the
    signatures (rows sum to 1), `B` the one-hot matrix of the batch of each sample, and `F` one
    filter per batch: a positive factor for every fragment length. `∘` is the element-wise
    product and `rownorm` scales each row to sum to 1. The first batch is the reference: its
    filter is fixed at 1, and the other filters describe how each batch differs from it.

    With `H` given, the signatures are fixed and only the weights and filters are fitted, as in
    `batch-nmf predict`. Every batch then gets a free filter.

    Args:
        target: Input matrix with one row per sample, rows summing to 1.
        B: One-hot matrix of the batch of each sample (samples × batches), for example from
            `read_batch_file`.
        n_signatures: Number of signatures.
        H: Fixed signatures (signatures × lengths). If given, only the weights and filters are fitted.
        F: Fixed filters (batches × lengths). If given, they are not fitted.
        filter_regularization: If positive, the strength of a penalty on the differences between
            the filter values of neighbouring lengths, which makes the filters smoother.
        scaler_regularization: If positive, the strength of a penalty that keeps the row sums of the
            unnormalised reconstruction close to 1. This fixes the overall level of each filter,
            which the data do not determine.
        volume_regularization: If positive, the strength of a penalty on the volume of the
            signatures, `log det(H Hᵀ + I)`.
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
    ):
        super().__init__(target, n_signatures, H, volume_regularization=volume_regularization)
        self.name = "batch_NMF"
        self.filter_regularization = filter_regularization
        self.scaler_regularization = scaler_regularization
        self.B = torch.tensor(B, dtype=self.dtype, device=self.device)
        self.n_batch = B.shape[1]
        if not F is None:
            self.knownF = True
            self.F = torch.tensor(F, dtype=self.dtype, device=self.device)
        else:
            self.knownF = False
        if self.predict:
            self.F_shape = (self.n_batch, self.n_types)
        else:
            self.F_shape = (self.n_batch - 1, self.n_types)
            self.first_row_F = torch.ones((1, self.n_types), dtype=self.dtype, device=self.device)
        self.F_trans = torch.exp
        self.rand_init()
        
    def rand_init(self):
        super().rand_init()
        if not self.knownF:  # inits W and H (if not fixed)
            self.F = self.init_par(self.F_shape, 0.0)

    def get_F(self):
        """The filters as a torch tensor: one row per batch (the reference batch's row is 1)."""
        if self.knownF:
            return self.F
        else:
            if self.predict:
                return self.F_trans(self.F)
            else:
                return torch.cat((self.first_row_F, self.F_trans(self.F)), dim=0)

    def get_penalty(self):
        penalty = super().get_penalty()
        if self.scaler_regularization > 0:
            penalty += self.scaler_regularization * ((self.row_sum-1.0)**2).mean()
        if self.filter_regularization > 0:
            penalty += self.filter_regularization * (torch.diff(self.F_trans(self.F), dim=1, n=1) ** 2).mean()
        return penalty

    def forward(self):
        X = (self.get_W() @ self.get_H()) * (self.B @ self.get_F())
        self.row_sum = X.sum(1, keepdims=True)
        return X / self.row_sum

    def get_parameters(self):
        """The fitted parameters as numpy arrays: `(W, H, F)`, or `(W, F)` when `H` was given."""
        if self.predict:
            return (self.get_W().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())
        else:
            return (self.get_W().cpu().detach().numpy(), self.get_H().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())

