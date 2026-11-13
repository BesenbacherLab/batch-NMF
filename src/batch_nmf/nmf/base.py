from typing import TextIO

import torch
import torch.nn as nn
from batch_nmf.utils import (
    train_model, write_log_header, MAX_SD, DEFAULT_EPOCHS, DEFAULT_RESTART_EPOCHS, DEFAULT_RESTARTS, DEFAULT_LR,
    DEFAULT_LR_FACTOR, DEFAULT_MIN_LR, DEFAULT_LOSS, DEFAULT_TOL, DEFAULT_CHECK_EVERY,
)

class baseNMF(nn.Module):
    """Common code of the NMF models (fitting, weights and signatures)."""

    def __init__(self, target, n_signatures=2, H=None, dtype=torch.double, volume_regularization=-1):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = dtype
        n_samples, n_types = target.shape
        self.n_samples = n_samples
        self.n_types = n_types
        self.n_signatures = n_signatures
        self.X = torch.tensor(target, dtype=dtype, device=self.device)
        if H is None:
            self.H_shape = (n_signatures, n_types)
            self.H_trans = torch.nn.Softmax(dim=1)
            self.predict = False
        else:
            self.H = torch.tensor(H, dtype=dtype, device=self.device)
            self.predict = True
        self.W_trans = torch.nn.Softmax(dim=1)
        self.W_shape = (n_samples, n_signatures)
        self.volume_regularization = volume_regularization

    def init_par(self, shape, add=0.0, mult=1.0):
        return nn.Parameter(
            torch.clamp(torch.randn(shape, dtype=self.dtype, requires_grad=True, device=self.device), min=-MAX_SD, max=MAX_SD)*mult + add
        )

    def rand_init(self):
        self.W = self.init_par(self.W_shape, -1.0)
        if not self.predict:
            self.H = self.init_par(self.H_shape,  1.0)

    def get_W(self):
        """The weights (exposures) as a torch tensor: one row per sample, rows sum to 1."""
        return self.W_trans(self.W)

    def get_H(self):
        """The signatures as a torch tensor: one row per signature, rows sum to 1.

        If the signatures were given (predict mode), these are returned unchanged.
        """
        if self.predict:
            return self.H
        else:
            return self.H_trans(self.H)

    
    def get_penalty(self):
        if self.volume_regularization > 0.0 and not self.predict:
            H = self.get_H()
            delta = 1.0
            vol = (torch.logdet(H@H.t() + torch.eye(self.n_signatures, dtype=self.dtype, device=self.device)*delta))
            assert vol > 0.0
            return self.volume_regularization * vol
        else:
            return torch.tensor(0.0, device=self.device)

    def fit(self, n_epochs: int = DEFAULT_EPOCHS, logfile: TextIO | None = None, lr: float = DEFAULT_LR,
            restarts: int = DEFAULT_RESTARTS, loss: str = DEFAULT_LOSS, weighted_loss: bool = False,
            min_lr_rate: float = DEFAULT_MIN_LR, lr_decrease_factor: float = DEFAULT_LR_FACTOR,
            n_epochs_restart: int = DEFAULT_RESTART_EPOCHS, tol: float = DEFAULT_TOL,
            check_every: int = DEFAULT_CHECK_EVERY, vrgrid: bool = False) -> tuple:
        """Fit the model with the Adam optimizer.

        Runs `restarts` random starts of at most `n_epochs_restart` epochs each, and then a
        final run of at most `n_epochs` epochs that continues from the best start. The learning
        rate is multiplied by `lr_decrease_factor` whenever the loss has not improved for 1000
        epochs. A run stops at its maximum number of epochs, when the learning rate falls below
        `min_lr_rate`, or when no fitted value has changed by more than `tol` over the last
        `check_every` epochs. The defaults are the same as those of the `batch-nmf` command line
        tool. Call `torch.manual_seed` before creating the model to make a fit repeatable.

        Args:
            n_epochs: Maximum number of epochs of the final run.
            logfile: Open text file for the training log (tab-separated, one row every 5000
                epochs and at the end of each run), or None.
            lr: Initial learning rate.
            restarts: Number of random starts before the final run.
            loss: `'kullback-leibler'` or `'frobenius'`.
            weighted_loss: Weight the loss so that every batch contributes equally.
            min_lr_rate: A run stops when the learning rate falls below this.
            lr_decrease_factor: Factor applied to the learning rate when the loss stops improving.
            n_epochs_restart: Maximum number of epochs of each random start.
            tol: Convergence tolerance (0 turns the convergence check off). Changes are absolute
                for values up to 1 and relative for larger values.
            check_every: Number of epochs between convergence checks.
            vrgrid: EXPERIMENTAL. Choose the volume regularization by a grid search (see `fit_grid`).

        Returns:
            A tuple `(parameters, loss, loss_without_penalty, penalty)` for the best state found:
                `parameters` as returned by `get_parameters` (for example `(W, H, F)` for bNMF),
                the loss with and without the regularization penalties, and the penalty.
        """
        options = dict(n_epochs=n_epochs, logfile=logfile, lr=lr, restarts=restarts, loss=loss,
                       weighted_loss=weighted_loss, min_lr_rate=min_lr_rate, lr_decrease_factor=lr_decrease_factor,
                       n_epochs_restart=n_epochs_restart, tol=tol, check_every=check_every)
        if vrgrid:
            return self.fit_grid(**options)
        write_log_header(logfile)
        return train_model(self, self.X, **options)

    def fit_grid(self, n_epochs=DEFAULT_EPOCHS, logfile=None, lr=DEFAULT_LR, restarts=DEFAULT_RESTARTS, vr_grid=None,
                 loss=DEFAULT_LOSS, weighted_loss=False, rel_cutoff=1.001, min_lr_rate=DEFAULT_MIN_LR,
                 lr_decrease_factor=DEFAULT_LR_FACTOR, n_epochs_restart=DEFAULT_RESTART_EPOCHS, tol=DEFAULT_TOL,
                 check_every=DEFAULT_CHECK_EVERY):
        """EXPERIMENTAL: choose the volume regularization by a grid search. Not tested.

        Fits the model without volume regularization, then with increasing values: `vr_grid`, or
        by default 11 values that start at loss / (50 * volume) and double each time. Each value
        continues from the previous fit, with the learning rate divided by 50. The search stops at
        the first value whose loss without the penalty is more than `rel_cutoff` times the loss of
        the unregularized fit, and the fit with the previous value is returned.
        """
        write_log_header(logfile)
        schedule = dict(min_lr_rate=min_lr_rate, lr_decrease_factor=lr_decrease_factor, tol=tol, check_every=check_every)
        self.volume_regularization = 0.0
        vr0_res, vr0_loss, vr0_loss_np, vr0_penalty = train_model(
            self, self.X, n_epochs=n_epochs, logfile=logfile, lr=lr, restarts=restarts, verbosity=0, loss=loss,
            weighted_loss=weighted_loss, n_epochs_restart=n_epochs_restart, **schedule,
        )

        best_res = None
        best_vr_val = 0.0
        print('##### VRGRID #####', 0.0, vr0_loss, vr0_loss_np, vr0_penalty)
        if vr_grid is None:
            grid_dist = 2.0
            grid_len = 10
            H = self.get_H()
            delta = 1.0
            vol = (torch.logdet(H@H.t() + torch.eye(self.n_signatures, dtype=self.dtype, device=self.device)*delta))
            val = vr0_loss / (50*vol.cpu().detach().numpy())
            vr_grid = [val]
            for _ in range(grid_len):
                val = val * grid_dist
                vr_grid.append(val)
        else:
            vr_grid = sorted(x for x in vr_grid if x > 1e-30)

        print(vr_grid)
        print('max:', rel_cutoff * vr0_loss_np)
        for vr_val in vr_grid:
            self.volume_regularization = vr_val
            res, vr_loss, vr_loss_np, penalty = train_model(
                self, self.X, n_epochs=max(n_epochs, 40000), logfile=logfile, lr=lr/50, restarts=0, verbosity=0,
                loss=loss, weighted_loss=weighted_loss, **schedule,
            )
            print('##### VRGRID #####', vr_val, vr_loss, vr_loss_np, penalty)
            if vr_loss_np < (rel_cutoff * vr0_loss_np):
                best_loss = vr_loss
                best_loss_np = vr_loss_np
                best_res = res
                best_penalty = penalty
                best_vr_val = vr_val
            else:
                break

        if best_res is None:
            best_res = vr0_res
            best_loss = vr0_loss
            best_loss_np = vr0_loss_np
            best_penalty = vr0_penalty
        print('##### VRBEST #####', best_vr_val, best_loss, best_loss_np, best_penalty)
        return best_res, best_loss, best_loss_np, best_penalty

