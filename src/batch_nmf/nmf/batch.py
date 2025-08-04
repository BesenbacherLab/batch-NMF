import torch
import numpy as np
import torch.nn as nn
from batch_nmf.nmf.base import baseNMF

class batchNMF(baseNMF):
    def __init__(
        self,
        target,
        B,
        n_signatures=2,
        H=None,
        F=None,
        filter_regularization=0.0,
        scaler_regularization=-1,
        volume_regularization=-1,
    ):
        super().__init__(target, n_signatures, H, volume_regularization=volume_regularization)
        self.name = f"batch_NMF"
        self.filter_regularization = filter_regularization
        self.scaler_regularization = scaler_regularization
        self.B = torch.tensor(B, dtype=self.dtype, device=self.device)
        self.n_batch = B.shape[1]
        if not F is None:
            self.knownF = True
            self.F = torch.tensor(F, dtype=self.dtype, device=self.device)
        else:
            self.knownF = False

        self.F_shape = (self.n_batch - 1, self.n_types)
        self.first_row_F = torch.ones((1, self.n_types), dtype=self.dtype, device=self.device)
        self.F_trans = torch.exp
        self.rand_init()
        
    def rand_init(self):
        super().rand_init()
        if not self.knownF:  # inits W and H (if not fixed)
            self.F = self.init_par(self.F_shape, 0.0)

    def get_F(self):
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
        if self.predict:
            return (self.get_W().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())
        else:
            return (self.get_W().cpu().detach().numpy(), self.get_H().cpu().detach().numpy(), self.get_F().cpu().detach().numpy())

