import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from copy import deepcopy

MAX_SD = 2
DEFAULT_EPOCHS = 100000
DEFAULT_N_LOWERING = 4

def print_vec_diff(name, X, Y):
    print('====', name, '====')
    #print(np.mean(X), np.mean(Y), np.mean(X)-np.mean(Y))
    #print(np.min(Y), np.max(Y), np.max(Y)-np.min(Y))
    q75, q50, q25 = np.percentile(X, [75 ,50, 25])
    print(q25, q50, q75, q75-q25)
    q75, q50, q25 = np.percentile(Y, [75 ,50, 25])
    print(q25, q50, q75, q75-q25)

def get_best_model(
    model, 
    target, 
    best_loss_wp, #best loss sofar with penalty
    best_loss_np, #best loss sofar without penalty
    best_model,
    loss_fn,
    traintype,
    n_epochs = DEFAULT_EPOCHS,
    lr = 0.1, 
    logfile=None, 
    verbosity = 2,
    min_lr_rate = 1e-5,
    log_input = False):
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    patience = 1000
    factor = 0.90
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = factor, verbose=True)

    if not logfile is None:
        start_time = time.time()
    for epoch in range(n_epochs):
        model.zero_grad(set_to_none=True)
        Y_ = model()
        if log_input:
            loss = loss_fn(Y_.log(), target)
        else:
            loss = loss_fn(Y_, target)

        loss_np = loss.item() # loss without penalty
        #loss_np = loss.cpu().detach().numpy()
        
        #if loss_np < 0:
        #    print("Loss is negative, resetting to 0.0")
        #    print_vec_diff("Y_", Y_.cpu().detach().numpy(), target.cpu().detach().numpy())
        #    print(Y_.cpu().detach().numpy().sum(1, keepdims=True))
        #    print(target.cpu().detach().numpy().sum(1, keepdims=True))
        #    assert loss_np >= 0, "Loss should not be negative"

        if hasattr(model, "get_penalty") and not model.predict:
            penalty = model.get_penalty()
            loss += penalty
        else:
            penalty = 0.0

        loss_wp = loss.item() # loss with penalty
        if loss_wp < best_loss_wp:
            best_model = deepcopy(model.state_dict())
            best_loss_wp = loss_wp
            best_loss_np = loss_np

        if(epoch%5000==1) and verbosity>1:
            print(epoch, loss_wp, loss_np, best_loss_wp, best_loss_np, penalty, scheduler._last_lr)

        if(epoch%5000==1) and not logfile is None:
            current_time = time.time() - start_time
            print(model.name, lr, traintype, epoch, loss_wp, loss_np, best_loss_wp, best_loss_np, penalty, scheduler._last_lr, current_time, sep="\t", file=logfile)
            logfile.flush()

        loss.backward(retain_graph=False)
        optimizer.step()
        scheduler.step(loss)
        if scheduler._last_lr[0] < min_lr_rate:
            break

    return best_model, best_loss_wp, best_loss_np

def make_weighted_loss(weights, criterion):
    weights /= weights.mean()
    def new_loss(y_hat, y):
        loss = criterion(y_hat.T, y.T)
        loss = loss * weights
        return loss.mean()
    return new_loss


def train_model(
    model, 
    target, 
    n_epochs = DEFAULT_EPOCHS,
    restarts = 5, 
    lr = 0.1, 
    logfile=None,
    verbosity = 2,
    min_lr_rate = 1e-4,
    loss = 'frobenius',
    weighted_loss = False):

    print(lr)
    if loss == 'frobenius':
        criterion = nn.MSELoss
    elif loss == 'kullback-leibler':
        criterion = nn.KLDivLoss    

    if weighted_loss:
        assert not model.B is None, "model needs batch matrix for weighted_loss to work"
        weights = model.B @ model.B.sum(axis=0)    
    else:
        weights = model.B.sum(axis=1)

    loss_fn = make_weighted_loss(weights, criterion(reduction='none')) 

    best_loss = 1e30
    best_loss_np = 1e30 
    best_model = None

    if not logfile is None:
        print("name", "lr", "run", "epoch", "best_loss", "loss", "penalty", "time", "current_lr", sep="\t", file=logfile)
    
    for i in range(restarts):
        best_model, best_loss, best_loss_np = \
            get_best_model(
                model, 
                target, 
                best_loss,
                best_loss_np,
                best_model,
                loss_fn,
                f'initial_{i}',
                n_epochs = 40000,
                lr = lr, 
                logfile = logfile, 
                verbosity = verbosity,
                min_lr_rate = min_lr_rate,
                log_input = loss == 'kullback-leibler')
        model.rand_init()

    if not best_model is None:
        model.load_state_dict(best_model)

    best_model, best_loss, best_loss_np = \
        get_best_model(
            model, 
            target, 
            best_loss,
            best_loss_np,
            best_model,
            loss_fn,
            'final',
            n_epochs = n_epochs,
            lr = lr, 
            logfile = logfile, 
            verbosity = verbosity,
            min_lr_rate = min_lr_rate,
            log_input = loss == 'kullback-leibler')

    model.load_state_dict(best_model)

    best_parameters = model.get_parameters()
    if hasattr(model, "get_penalty"):
        model()
        penalty = model.get_penalty().cpu().detach().numpy()
    else:
        penalty = 0.0
    return best_parameters, best_loss, best_loss_np, penalty

