import warnings

import torch
import numpy as np
import torch.nn as nn
from batch_nmf.nmf.base import baseNMF

class sigmoidfilterNMF(baseNMF):
    def __init__(
        self,
        target,
        B,
        n_signatures=2,
        H=None,
        F=None,
        filter_regularization=-1,
        scaler_regularization=-1,
        volume_regularization=-1,
        min_length=80,
        lengths=None,
    ):
        """
        Arguments:
            filter_regularization: Not used. The smoothness penalty of bNMF has no effect on
                sigmoid filters, which are smooth already; a positive value gives a warning.
            min_length: Fragment length of the first column. The columns are assumed to be
                consecutive lengths min_length, min_length+1, ... Ignored if `lengths` is given.
            lengths: Fragment length of each column of `target` (overrides `min_length`).
        """
        super().__init__(target, n_signatures, H, volume_regularization=volume_regularization)
        self.name = f"sigmoid_filter_NMF"
        if filter_regularization > 0:
            warnings.warn(
                "filter_regularization has no effect for sfNMF (the sigmoid filters are already smooth) "
                "and is ignored.",
                UserWarning,
                stacklevel=2,
            )
        self.scaler_regularization = scaler_regularization
        self.B = torch.tensor(B, dtype=self.dtype, device=self.device)
        self.n_batch = B.shape[1]
        self.min_length = min_length
        if lengths is None:
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

    def get_penalty(self):
        penalty = super().get_penalty()
        if self.scaler_regularization > 0:
            penalty += self.scaler_regularization * torch.nn.functional.relu(self.row_sum.max() - 1.0)
        return penalty

    def forward(self):
        X = (self.get_W() @ self.get_H()) * (self.B @ self.get_F())
        self.row_sum = X.sum(1, keepdims=True)
        return X / self.row_sum

    def get_parameters(self):
        if self.predict:
            return (self.get_W().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())
        else:
            return (self.get_W().cpu().detach().numpy(), self.get_H().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())

