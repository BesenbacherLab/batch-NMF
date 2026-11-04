#!/usr/bin/env python3
"""Measure how many epochs batch-nmf needs on a dataset.

Runs several random starts of bNMF or sfNMF for a fixed, long number of epochs (no early
stopping) and records the fitted parameters every --check_every epochs. From these
trajectories it reports:

1. For a range of tolerances: when the convergence check (`--tol` in `batch-nmf`) would
   have stopped each start, and how far the exposures/signatures/filters at that point are
   from where the start ends up after all epochs. Use this to choose --tol.
2. How many starts end in the best solution, and from which epoch on the loss already tells
   the good starts from the bad ones. Use this to choose --n_epochs_restart and --n_restarts.
3. How much starts that reach the same loss disagree about exposures, signatures and
   filter shapes. Filters are compared after dividing each batch's filter by its mean: the
   overall level of a batch's filter cancels in the row normalisation, so for bNMF it is not
   determined by the data (unless --scaler_regularization is used).

Example (run from the repository root):

    uv run python scripts/convergence_benchmark.py fit \\
        --input_matrix test_data/length_data.tsv --batch_file test_data/batch_file.tsv \\
        --n_starts 8 --n_epochs 100000 --out benchmark.tsv

The options mirror the `batch-nmf fit` options. Run time is roughly
n_starts * n_epochs * (time per epoch); the time per epoch is printed after the first start.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from batch_nmf.io_utils import get_fragment_lengths, read_batch_file, read_matrix
from batch_nmf.nmf.batch import batchNMF
from batch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF
from batch_nmf.utils import get_best_model, make_loss_fn, max_parameter_change, parameter_snapshot

TIE = 1e-9  # starts whose final losses differ by less than this (relative) count as equally good


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convergence_benchmark",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("command", choices=["fit"], help="Only 'fit' is supported.")
    parser.add_argument("--input_matrix", required=True)
    parser.add_argument("--batch_file")
    parser.add_argument("--all_separate_batch", action="store_true")
    parser.add_argument("--method", choices=["bNMF", "sfNMF"], default="bNMF")
    parser.add_argument("--n_signatures", type=int, default=2)
    parser.add_argument("--pseudo_count", type=float, default=1.0)
    parser.add_argument("--loss", choices=["frobenius", "kullback-leibler"], default="kullback-leibler")
    parser.add_argument("--weighted_loss", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=0.5)
    parser.add_argument("--lr_decrease_factor", type=float, default=0.80)
    parser.add_argument("--min_learning_rate", type=float, default=1e-4,
                        help="Only used to report when the learning-rate stop would have fired.")
    parser.add_argument("--min_length", type=int, default=None, help="sfNMF only, see batch-nmf -h.")
    parser.add_argument("--filter_regularization", type=float, default=-1)
    parser.add_argument("--scaler_regularization", type=float, default=-1)
    parser.add_argument("--volume_regularization", type=float, default=-1)
    parser.add_argument("--n_starts", type=int, default=8, help="Number of random starts.")
    parser.add_argument("--n_epochs", type=int, default=100000, help="Epochs per start (no early stopping).")
    parser.add_argument("--check_every", type=int, default=1000,
                        help="Record the parameters every this many epochs (the convergence-check window).")
    parser.add_argument("--tols", type=str, default="1e-3,1e-4,1e-5,1e-6",
                        help="Comma-separated tolerances to evaluate.")
    parser.add_argument("--seed", type=int, default=1, help="Start i uses torch seed SEED + i.")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional TSV file with the recorded trajectory of every start.")
    return parser


def load_data(opts: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, list]:
    """Read and normalise the input matrix the same way as `batch-nmf`."""
    X, sample_names, col_names = read_matrix(opts.input_matrix)
    X = X + opts.pseudo_count
    X = X / X.sum(1, keepdims=True)
    if opts.all_separate_batch:
        B = np.identity(len(sample_names))
    elif opts.batch_file is not None:
        B, _ = read_batch_file(opts.batch_file, sample_names)
    else:
        sys.exit("convergence_benchmark: --batch_file or --all_separate_batch is required")
    return X, B, list(col_names)


def make_model(opts: argparse.Namespace, X: np.ndarray, B: np.ndarray, col_names: list):
    kwargs = {
        "filter_regularization": opts.filter_regularization,
        "scaler_regularization": opts.scaler_regularization,
        "volume_regularization": opts.volume_regularization,
    }
    if opts.method == "bNMF":
        return batchNMF(X, B, opts.n_signatures, **kwargs)
    return sigmoidfilterNMF(X, B, opts.n_signatures, lengths=get_fragment_lengths(col_names, opts.min_length), **kwargs)


def run_start(opts: argparse.Namespace, X, B, col_names, seed: int) -> dict:
    """Run one random start for opts.n_epochs epochs and record its trajectory."""
    torch.manual_seed(seed)
    model = make_model(opts, X, B, col_names)
    loss_fn = make_loss_fn(model, opts.loss, opts.weighted_loss)
    rec = {"seed": seed, "epoch": [0], "best_loss": [np.inf], "lr": [opts.learning_rate],
           "elapsed": [0.0], "params": [parameter_snapshot(model)]}
    start = time.time()

    def callback(epoch, model, best_loss, lr):
        rec["epoch"].append(epoch + 1)
        rec["best_loss"].append(best_loss)
        rec["lr"].append(lr)
        rec["elapsed"].append(time.time() - start)
        rec["params"].append(parameter_snapshot(model))

    get_best_model(
        model, model.X, 1e30, 1e30, None, loss_fn, f"start_{seed}",
        n_epochs=opts.n_epochs, lr=opts.learning_rate, verbosity=0,
        min_lr_rate=0.0,  # run all epochs
        lr_decrease_factor=opts.lr_decrease_factor,
        log_input=opts.loss == "kullback-leibler",
        tol=0.0, check_every=opts.check_every, callback=callback,
    )
    rec["changes"] = [np.nan] + [
        max_parameter_change(a, b) for a, b in zip(rec["params"][:-1], rec["params"][1:])
    ]
    return rec


def align_to(params: list, ref: list) -> list:
    """Reorder the signatures in `params` ([W, H, F]) to best match `ref`."""
    W, H = params[0], params[1]
    cost = np.abs(ref[1][:, None, :] - H[None, :, :]).sum(axis=2)
    _, perm = linear_sum_assignment(cost)
    return [W[:, perm], H[perm], *params[2:]]


def fmt(x: float) -> str:
    return "-" if x is None or not np.isfinite(x) else f"{x:.1e}"


def report(opts: argparse.Namespace, runs: list) -> None:
    tols = [float(t) for t in opts.tols.split(",")]
    n_checks = len(runs[0]["epoch"])
    final_losses = np.array([r["best_loss"][-1] for r in runs])
    best = final_losses.min()
    gaps = (final_losses - best) / abs(best)
    secs_per_epoch = np.mean([r["elapsed"][-1] / r["epoch"][-1] for r in runs])

    print(f"\n{len(runs)} starts x {opts.n_epochs} epochs, {secs_per_epoch * 1e3:.2f} ms per epoch")

    print("\n1) Convergence check (--tol): epoch at which each start would stop, and the largest change in")
    print("   any exposure/signature/filter value between that epoch and the end of the run")
    print("   (absolute for values <= 1, relative above).")
    header = "start  " + "  ".join(f"{'tol ' + str(t):>20s}" for t in tols)
    print(header)
    for r in runs:
        cells = []
        for t in tols:
            idx = next((i for i in range(1, n_checks) if r["changes"][i] < t), None)
            if idx is None:
                cells.append(f"{'not reached':>20s}")
            else:
                err = max_parameter_change(r["params"][idx], r["params"][-1])
                cells.append(f"{r['epoch'][idx]:>9d} ep, {fmt(err):>7s}")
        print(f"{r['seed']:5d}  " + "  ".join(cells))
    below = [next((e for e, lr in zip(r["epoch"], r["lr"]) if lr < opts.min_learning_rate), None) for r in runs]
    print(f"   For comparison, the learning-rate stop (< {opts.min_learning_rate}) fires at epoch: "
          + ", ".join("never" if b is None else str(b) for b in below))

    print(f"\n2) Restarts: final loss of each start relative to the best start (ties: < {TIE:g})")
    good = gaps < TIE
    for r, g in zip(runs, gaps):
        print(f"   start {r['seed']}: {g:.2e}{'  (best solution)' if g < TIE else ''}")
    print(f"   {good.sum()} of {len(runs)} starts reach the best solution.")
    if good.all():
        print("   All starts end equally good, so on this dataset restarts only cost time.")
    else:
        settled = None
        for i in range(n_checks - 1, -1, -1):
            losses = np.array([r["best_loss"][i] for r in runs])
            if losses[good].max() < losses[~good].min():
                settled = runs[0]["epoch"][i]
            else:
                break
        if settled is None:
            print("   The good and bad starts can only be told apart at the very end; use more epochs per restart.")
        else:
            print(f"   From epoch {settled} on, every start that ends in the best solution has a lower loss than every")
            print("   start that does not, so --n_epochs_restart around this value is enough to pick the right start.")

    print("\n3) Agreement between the equally good starts (after matching signatures):")
    ref_idx = int(np.argmin(final_losses))
    ref = runs[ref_idx]["params"][-1]
    names = ["exposures (W)", "signatures (H)", "filter shapes (F / row mean)"]

    def shape(F):
        return F / F.mean(axis=1, keepdims=True)

    for r, g in zip(runs, gaps):
        if g >= TIE or r is runs[ref_idx]:
            continue
        p = align_to(r["params"][-1], ref)
        p[2], q = shape(p[2]), [ref[0], ref[1], shape(ref[2])]
        diffs = ", ".join(f"{names[k]} {np.abs(p[k] - q[k]).max():.1e}" for k in range(3))
        print(f"   start {r['seed']} vs start {runs[ref_idx]['seed']}: max abs difference in {diffs}")
    if good.sum() < 2:
        print("   (fewer than two equally good starts)")


def write_tsv(path: str, runs: list) -> None:
    with open(path, "w", encoding="utf-8") as f:
        print("seed", "epoch", "best_loss", "max_change", "lr", "elapsed_s", sep="\t", file=f)
        for r in runs:
            for i in range(len(r["epoch"])):
                print(r["seed"], r["epoch"][i], r["best_loss"][i], r["changes"][i], r["lr"][i], r["elapsed"][i],
                      sep="\t", file=f)


def main(args: list[str] | None = None) -> int:
    opts = get_parser().parse_args(args)
    X, B, col_names = load_data(opts)
    runs = []
    for i in range(opts.n_starts):
        runs.append(run_start(opts, X, B, col_names, opts.seed + i))
        print(f"start {opts.seed + i} done: best loss {runs[-1]['best_loss'][-1]:.10g}, "
              f"{runs[-1]['elapsed'][-1]:.0f} s", flush=True)
    if opts.out is not None:
        write_tsv(opts.out, runs)
    report(opts, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
