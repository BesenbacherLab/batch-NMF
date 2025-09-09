import torch
import numpy as np
import torch.nn as nn
from batch_nmf.utils import train_model, MAX_SD

class baseNMF(nn.Module):
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
        self.zero = torch.tensor(0.0, device=self.device)
        self.W_trans = torch.nn.Softmax(dim=1)
        self.W_shape = (n_samples, n_signatures)
        self.volume_regularization = volume_regularization

    def init_par(self, shape, add=0.0, mult=1.0):
        return nn.Parameter(
            torch.clamp(torch.randn(shape, dtype=self.dtype, requires_grad=True, device=self.device), min=-MAX_SD, max=MAX_SD)*mult + add
        )

    def rand_init(self):
        #self.current_vr = self.volume_regularization
        self.W = self.init_par(self.W_shape, -1.0)
        if not self.predict:
            self.H = self.init_par(self.H_shape,  1.0)

    def get_W(self):
        return self.W_trans(self.W)

    def get_H(self):
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
            return torch.tensor(0.0, device=self.device) #0.0 #self.zero

    def fit(self, n_epochs=40000, logfile=None, lr=0.1, restarts = 4, vrgrid=False, loss='frobenius', weighted_loss=False, min_lr_rate=1e-5):
        if not vrgrid:
            return train_model(
                self, self.X, n_epochs=n_epochs, logfile=logfile, lr=lr, restarts=restarts, loss=loss, weighted_loss=weighted_loss, min_lr_rate=min_lr_rate,
            )
        else:
            return self.fit_grid(n_epochs = n_epochs, logfile = logfile, lr =lr, restarts = restarts, loss=loss, weighted_loss=weighted_loss)
    
    def fit_grid(self, n_epochs=40000, logfile=None, lr=0.1, restarts = 1, vr_grid =None, loss='frobenius', weighted_loss=False, rel_cutoff=1.001):
        self.volume_regularization = 0.0
        vr0_res, vr0_loss, vr0_loss_np, vr0_penalty = train_model(
            self, self.X, n_epochs=n_epochs, logfile=logfile, lr=lr, restarts=restarts, verbosity=0, loss=loss, weighted_loss=weighted_loss
        )

        best_loss_np = 1e30
        best_res = None
        best_vr_val = 0.0
        print('##### VRGRID #####', 0.0, vr0_loss, vr0_loss_np, vr0_penalty)
        if vr_grid is None:
            #grid_dist = 1.5
            grid_dist = 2.0
            grid_len = 10
            H = self.get_H()
            delta = 1.0
            vol = (torch.logdet(H@H.t() + torch.eye(self.n_signatures, dtype=self.dtype, device=self.device)*delta))
            val = vr0_loss / (50*vol.cpu().detach().numpy())
            vr_grid = [val]
            for i in range(grid_len):
                val = val * grid_dist
                vr_grid.append(val)
        else:
            vr_grid.sort()
            vr_grid = [x for x in vr_grid if x>1e-30]

        print(vr_grid)
        print('max:',rel_cutoff * vr0_loss_np)
        for vr_val in vr_grid:
            self.volume_regularization = vr_val
            n_epochs2 = max(n_epochs, 40000)
            
            res, vr_loss, vr_loss_np, penalty = train_model(
                self, self.X, n_epochs=n_epochs2, logfile=logfile, lr=lr/50, restarts=0, verbosity=0, loss=loss, weighted_loss=weighted_loss
            )
            print('##### VRGRID #####', vr_val, vr_loss, vr_loss_np, penalty)
            print('vr_loss_np', vr_loss_np)
            if vr_loss_np < (rel_cutoff * vr0_loss_np):# and loss_np <= best_loss_np:
                #print('better')
                best_loss = vr_loss
                best_loss_np = vr_loss_np
                best_res = res
                best_penalty = penalty
                best_vr_val = vr_val
            else:
                #print('not_better')
                break
         
        if best_res is None:
            best_res = vr0_res
            best_loss = vr0_loss
            best_loss_np = vr0_loss_np
            best_penalty = vr0_penalty
        print('##### VRBEST #####', best_vr_val, best_loss, best_loss_np, best_penalty)
        return best_res, best_loss, best_loss_np, best_penalty

