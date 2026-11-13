import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
from copy import deepcopy

MAX_SD = 2

# Default fitting options, shared by the Python API (baseNMF.fit, train_model,
# get_best_model) and the command line tool. The epoch counts are upper limits: a run
# stops earlier once no fitted value changes by more than DEFAULT_TOL over
# DEFAULT_CHECK_EVERY epochs (see get_best_model).
DEFAULT_LOSS = 'kullback-leibler'
"""Loss function (`--loss`)."""
DEFAULT_RESTARTS = 4
"""Number of random starts before the final run (`--n_restarts`)."""
DEFAULT_LR = 0.5
"""Initial learning rate of the Adam optimizer (`--learning_rate`)."""
DEFAULT_LR_FACTOR = 0.8
"""Factor applied to the learning rate when the loss has not improved for 1000 epochs (`--lr_decrease_factor`)."""
DEFAULT_MIN_LR = 1e-4
"""A run stops when the learning rate falls below this (`--min_learning_rate`)."""
DEFAULT_EPOCHS = 50000
"""Maximum number of epochs of the final run (`--n_epochs`)."""
DEFAULT_RESTART_EPOCHS = 40000
"""Maximum number of epochs of each random start (`--n_epochs_restart`)."""
DEFAULT_TOL = 1e-4
"""Convergence tolerance (`--tol`)."""
DEFAULT_CHECK_EVERY = 1000
"""Number of epochs between convergence checks."""

# Losses within this relative distance of the best loss count as ties (floating-point noise).
# On a tie the best state is replaced by the newer one. Without this, a run that reaches the
# optimal loss early keeps returning that early state while the parameters keep moving along
# directions where the loss is flat to ~1e-15, so the returned exposures depended on where
# each start first hit the optimum (up to 0.07 apart on test_data) instead of where it converged.
LOSS_TIE = 1e-12

# Columns of the training log written by get_best_model (one row per 5000 epochs).
LOG_COLUMNS = (
    "name",                  # model name
    "lr",                    # initial learning rate
    "run",                   # initial_<i> (restart i) or final
    "epoch",
    "loss",                  # loss incl. penalty
    "loss_no_penalty",
    "best_loss",             # best loss so far incl. penalty
    "best_loss_no_penalty",
    "penalty",
    "current_lr",
    "time",                  # seconds since the start of this run
)


def write_log_header(logfile):
    """Write the header line of the training log (see LOG_COLUMNS)."""
    if logfile is not None:
        print(*LOG_COLUMNS, sep="\t", file=logfile)

def parameter_snapshot(model):
    """Copy of the fitted (transformed) parameters returned by `model.get_parameters()`.

    Returns a list of numpy arrays, e.g. [W, H, F] for bNMF/sfNMF when fitting.
    """
    with torch.no_grad():
        params = model.get_parameters()
    if not isinstance(params, tuple):
        params = (params,)
    return [np.array(p, copy=True) for p in params]


def max_parameter_change(old, new):
    """Largest change between two parameter snapshots.

    The change is absolute for values up to 1 (exposures and signatures lie in [0, 1])
    and relative for larger values (e.g. bNMF filters can be > 1).
    """
    return max(
        (float(np.max(np.abs(n - o) / np.maximum(1.0, np.abs(o)))) for o, n in zip(old, new) if o.size),
        default=0.0,
    )


def _write_log_row(logfile, model, lr, traintype, epoch, loss_wp, loss_np, best_loss_wp, best_loss_np,
                   penalty, current_lr, elapsed):
    penalty_value = penalty.item() if torch.is_tensor(penalty) else penalty
    # Keep in the same order as LOG_COLUMNS.
    print(model.name, lr, traintype, epoch, loss_wp, loss_np, best_loss_wp, best_loss_np,
          penalty_value, current_lr, elapsed, sep="\t", file=logfile)
    logfile.flush()


def get_best_model(
    model, 
    target, 
    best_loss_wp, #best loss sofar with penalty
    best_loss_np, #best loss sofar without penalty
    best_model,
    loss_fn,
    traintype,
    n_epochs = DEFAULT_EPOCHS,
    lr = DEFAULT_LR,
    logfile=None, 
    verbosity = 2,
    min_lr_rate = DEFAULT_MIN_LR,
    lr_decrease_factor = DEFAULT_LR_FACTOR,
    log_input = False,
    tol = DEFAULT_TOL,
    check_every = DEFAULT_CHECK_EVERY,
    callback = None):
    """Train `model` for at most `n_epochs` epochs and keep track of the best state.

    A run stops when the first of these happens:

    - `n_epochs` epochs have been run,
    - the learning rate has been lowered below `min_lr_rate`,
    - (if `tol` > 0) no fitted value (exposures, signatures, filters; see
      `max_parameter_change`) changed by more than `tol` over the last `check_every` epochs.

    `callback(epoch=..., model=..., best_loss=..., lr=...)`, if given, is called every
    `check_every` epochs (used by scripts/convergence_benchmark.py).

    Returns:
        (best_model_state, best_loss, best_loss_without_penalty, learning_rate_at_stop)
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    patience = 1000
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = patience, factor = lr_decrease_factor)

    start_time = time.time()
    last_snapshot = parameter_snapshot(model) if tol > 0 else None
    stop_reason = "max_epochs"
    epoch = -1
    for epoch in range(n_epochs):
        model.zero_grad(set_to_none=True)
        Y_ = model()
        if log_input:
            loss = loss_fn(Y_.log(), target)
        else:
            loss = loss_fn(Y_, target)

        loss_np = loss.item() # loss without penalty

        if hasattr(model, "get_penalty") and not model.predict:
            penalty = model.get_penalty()
            loss += penalty
        else:
            penalty = 0.0

        loss_wp = loss.item() # loss with penalty
        if loss_wp <= best_loss_wp + LOSS_TIE * abs(best_loss_wp):
            best_model = deepcopy(model.state_dict())
            if loss_wp < best_loss_wp:
                best_loss_wp = loss_wp
                best_loss_np = loss_np

        if(epoch%5000==1) and verbosity>1:
            print(epoch, loss_wp, loss_np, best_loss_wp, best_loss_np, penalty, scheduler.get_last_lr())

        if(epoch%5000==1) and not logfile is None:
            _write_log_row(logfile, model, lr, traintype, epoch, loss_wp, loss_np, best_loss_wp, best_loss_np,
                           penalty, scheduler.get_last_lr()[0], time.time() - start_time)

        loss.backward(retain_graph=False)
        optimizer.step()
        # Pass a plain float: float() on a tensor that requires grad raises a
        # UserWarning in recent torch versions (an error under pytest's filterwarnings=error).
        scheduler.step(loss_wp)
        if scheduler.get_last_lr()[0] < min_lr_rate:
            stop_reason = "min_lr"
            break

        if (epoch + 1) % check_every == 0:
            if callback is not None:
                callback(epoch=epoch, model=model, best_loss=best_loss_wp, lr=scheduler.get_last_lr()[0])
            if tol > 0:
                snapshot = parameter_snapshot(model)
                if max_parameter_change(last_snapshot, snapshot) < tol:
                    stop_reason = "converged"
                    break
                last_snapshot = snapshot

    if epoch >= 0:
        if verbosity > 1:
            print(f"{traintype}: stopped after {epoch + 1} epochs ({stop_reason})")
        if logfile is not None and epoch % 5000 != 1:
            # Always log the last epoch so the log shows where each run stopped.
            _write_log_row(logfile, model, lr, traintype, epoch, loss_wp, loss_np, best_loss_wp, best_loss_np,
                           penalty, scheduler.get_last_lr()[0], time.time() - start_time)

    return best_model, best_loss_wp, best_loss_np, scheduler.get_last_lr()[0]

def batch_balanced_weights(B):
    """Per-sample loss weights that make every batch contribute equally to the loss.

    Each sample gets weight 1 / (number of samples in its batch), so the weights of
    the samples in a batch sum to 1. `make_weighted_loss` rescales them to mean 1.

    Arguments:
        B: One-hot batch matrix of shape (n_samples, n_batches).
    """
    batch_sizes = B @ B.sum(axis=0)  # size of each sample's batch
    return 1.0 / batch_sizes


def make_loss_fn(model, loss=DEFAULT_LOSS, weighted_loss=False):
    """Return the training loss function for `model`.

    For 'kullback-leibler' the returned function expects log-probabilities as its first argument.
    """
    if loss == 'frobenius':
        criterion = nn.MSELoss
    elif loss == 'kullback-leibler':
        criterion = nn.KLDivLoss
    else:
        raise ValueError(f"Unknown loss {loss!r}, use 'frobenius' or 'kullback-leibler'")

    if weighted_loss:
        assert getattr(model, "B", None) is not None, "model needs batch matrix for weighted_loss to work"
        weights = batch_balanced_weights(model.B)
    else:
        weights = torch.ones(model.n_samples, dtype=model.dtype, device=model.device)

    return make_weighted_loss(weights, criterion(reduction='none'))


def make_weighted_loss(weights, criterion):
    weights = weights / weights.mean()
    def new_loss(y_hat, y):
        loss = criterion(y_hat.T, y.T)
        loss = loss * weights
        return loss.mean()
    return new_loss


def train_model(
    model, 
    target, 
    n_epochs = DEFAULT_EPOCHS,
    restarts = DEFAULT_RESTARTS,
    lr = DEFAULT_LR,
    logfile=None,
    verbosity = 2,
    min_lr_rate = DEFAULT_MIN_LR,
    lr_decrease_factor = DEFAULT_LR_FACTOR,
    loss = DEFAULT_LOSS,
    weighted_loss = False,
    n_epochs_restart = DEFAULT_RESTART_EPOCHS,
    tol = DEFAULT_TOL,
    check_every = DEFAULT_CHECK_EVERY):
    """Fit `model`: `restarts` random starts of at most `n_epochs_restart` epochs each,
    then a final run of at most `n_epochs` epochs from the best start.

    The final run continues at the learning rate where the best start stopped (`lr` if
    `restarts` is 0). Restarting it at the initial learning rate would first push an
    already converged solution away from the optimum and then spend many epochs re-converging.

    See `get_best_model` for the stopping rules (`tol`, `check_every`, `min_lr_rate`).
    """
    loss_fn = make_loss_fn(model, loss, weighted_loss)

    best_loss = 1e30
    best_loss_np = 1e30 
    best_model = None

    # The log header is written once per fit by baseNMF.fit / fit_grid (fit_grid calls
    # train_model several times and should not repeat the header).

    final_lr = lr
    for i in range(restarts):
        previous_best_model = best_model
        best_model, best_loss, best_loss_np, end_lr = \
            get_best_model(
                model, 
                target, 
                best_loss,
                best_loss_np,
                best_model,
                loss_fn,
                f'initial_{i}',
                n_epochs = n_epochs_restart,
                lr = lr, 
                logfile = logfile, 
                verbosity = verbosity,
                min_lr_rate = min_lr_rate,
                lr_decrease_factor = lr_decrease_factor,
                log_input = loss == 'kullback-leibler',
                tol = tol,
                check_every = check_every)
        if best_model is not previous_best_model:
            final_lr = end_lr  # the best state now comes from this start
        model.rand_init()

    if not best_model is None:
        model.load_state_dict(best_model)

    best_model, best_loss, best_loss_np, _ = \
        get_best_model(
            model, 
            target, 
            best_loss,
            best_loss_np,
            best_model,
            loss_fn,
            'final',
            n_epochs = n_epochs,
            lr = final_lr, 
            logfile = logfile, 
            verbosity = verbosity,
            min_lr_rate = min_lr_rate,
            lr_decrease_factor = lr_decrease_factor,
            log_input = loss == 'kullback-leibler',
            tol = tol,
            check_every = check_every)

    model.load_state_dict(best_model)

    best_parameters = model.get_parameters()
    if hasattr(model, "get_penalty"):
        model()
        penalty = model.get_penalty().cpu().detach().numpy()
    else:
        penalty = 0.0
    return best_parameters, best_loss, best_loss_np, penalty

