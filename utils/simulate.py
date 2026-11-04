#!/usr/bin/env python3
"""Simulation study for bNMF and sfNMF.

Generates fragment-length data with known signatures, weights and batch filters, fits
standard NMF, Adam-fitted NMF, bNMF, sfNMF and an oracle, and evaluates how well each
method recovers the truth.

Subcommands
-----------
simulate  Generate one dataset. --npz writes everything `run --data` needs to one file;
          --out_dir writes batch-nmf input files plus the ground truth as TSV files.
run       Fit the methods to one dataset (simulated from the options, or read with --data)
          and write one TSV of evaluation metrics.

The full simulation design (all experiments and replicates) is the gwf workflow
utils/simulation_workflow.py, which calls `simulate --npz` and `run --data`.

Examples (from the repository root)::

    uv run python utils/simulate.py run --replicate 1 --out results/base_rep1.tsv
    uv run python utils/simulate.py simulate --replicate 1 --npz sim/base_rep1.npz
    uv run python utils/simulate.py run --data sim/base_rep1.npz --methods sfnmf --out results/base_rep1_sfnmf.tsv
    uv run python utils/simulate.py simulate --replicate 1 --out_dir sim_data

Generative model
----------------
- Fragment lengths L = min_length..max_length (default 50-400 bp).
- Signatures H (C x M), rows summing to 1. By default these are parametric cfDNA-like
  profiles: an asymmetric mono-nucleosome peak with 10.4-bp periodicity below the peak, a
  di-nucleosome peak and a short-fragment tail. Signature 0 is "healthy" (peak at 167 bp),
  signature `--tumor_signature` (default 1) is "tumour" (shorter peak, stronger periodicity),
  and the others are random. Use `--signatures FILE` to use signatures estimated from real
  data instead (for example the --signature_file written by `batch-nmf fit`).
- Weights W (N x C), rows summing to 1. Cases get a tumour-signature weight from
  Beta(--tumor_fraction), controls from Beta(--control_tumor_fraction). The remaining weight
  is split over the other signatures by Dirichlet(--alpha); a small alpha gives samples close
  to a single signature, which makes the factorisation better identified.
- Batch filters F_k (one per batch, length M). Training batch 0 is the reference and is
  unfiltered (F_0 = 1). The other training batches and the unseen test batch get filters from
  `--filter_family`:
    sigmoid          increasing x decreasing Hill functions (the family sfNMF assumes)
    short_depletion  increasing Hill function only, strong loss of short fragments
    smooth           log F is a smooth Gaussian-process draw (non-monotone, multimodal)
    periodic         sigmoid x (1 + a sin(2 pi L / 10.4 + phase))
    noisy            sigmoid x exp(iid N(0, sd)) per length
    empirical        rows of --filters FILE (e.g. filters estimated on real data)
  `--filter_strength` s raises the filter to the power s (0 = no batch effect), and
  `--steepness` multiplies the Hill exponents. `--sample_filter_sd` adds a smooth per-sample
  deviation, so samples in a batch are not filtered identically.
- Observed data: p_i = rownorm((W H)_i * F_i). Counts ~ Multinomial(depth, p_i), optionally
  with Dirichlet-multinomial overdispersion (`--dm_concentration`). The input matrix is then
  rownorm(counts + pseudo_count), as in the batch-nmf command line tool. `--depth inf` gives
  noiseless data.

Methods (`--methods`)
---------------------
nmf     scikit-learn NMF (multiplicative updates, KL), as in the manuscript
plain   NMF with softmax-normalised W and H fitted with Adam (batch_nmf plainNMF)
bnmf    batch NMF
sfnmf   sigmoid filter NMF (given the true fragment lengths)
oracle  bNMF with the true batch filters given (only W and H are fitted); an upper bound

Each method is fitted `--n_starts` times from different random initialisations and the start
with the lowest loss is evaluated. The signatures are then fixed and the weights (and, for the
batch methods, a new filter) are inferred for the unseen test batch, which is split into
chunks of each size in `--test_chunk_sizes` that are analysed independently (like Fig. 7;
the default analyses the whole test batch at once, the `test_batch_size` experiment of the
workflow uses 50, 20, 10, 5, 2 and 1).

Run time: one run of the base scenario takes roughly 15-30 minutes on 2 CPU cores
(sfNMF is the slowest method); small test chunks add a few minutes per method.

Evaluation (one output row per method and data set)
---------------------------------------------------
Methods may place a filter shared by all batches differently (bNMF folds the reference
batch's filter into H; sfNMF has no reference batch). Estimates and truth are therefore
compared in the view of the reference batch: H -> rownorm(H * F_0) and
W_ic -> W_ic * sum_l H_cl F_0l (rows renormalised), using each method's own estimate of F_0.
Estimated signatures are matched to the true ones by the Hungarian algorithm on cosine
similarity.

kl_loss          mean KL(x_i || reconstruction_i) on the evaluated samples
w_mae, w_rmse    error of the weights
w_r2             1 - SSE / SST of the weights (SST around the column means)
tumor_w_mae      error of the tumour-signature weight
tumor_auc        AUC of the tumour-signature weight for cases vs controls
h_cos, h_jsd     mean cosine similarity / Jensen-Shannon divergence of matched signatures
h_rmse           RMSE of the signatures
filter_log_rmse  RMSE of log(F_k / F_0) vs the truth, after removing each batch's overall
                 level (not identified) and weighting lengths by fragment mass. For nmf and
                 plain (no filters) this is the error of assuming there is no batch effect.
corrected_jsd    mean JSD between the observed profile mapped into the reference batch
                 (x_i / F_k * F_0, renormalised) and the true unfiltered profile W_i H
n_tied_starts    number of starts whose loss is within --tie_tol (relative) of the best
spread_w_mae     largest weight MAE between the best start and a tied start
spread_h_cos     smallest signature cosine similarity between the best start and a tied start
runtime_s        seconds used by the fits

W and H metrics are left empty when the number of fitted signatures differs from the truth.

Note: w_mae also contains error from non-unique factorisations. In the base scenario the four
non-tumour signatures are fairly similar, and even noiseless data fitted perfectly with the true
filters (oracle) gives w_mae of about 0.017, while tumor_w_mae is about 0.005. Look at
tumor_w_mae, tumor_auc and the spread_* columns alongside w_mae, and use the identifiability
experiment (alpha) to show how this depends on the data.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.special import expit, xlogy
from sklearn.decomposition import NMF, non_negative_factorization
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from batch_nmf.io_utils import get_fragment_lengths, read_matrix
from batch_nmf.nmf.batch import batchNMF
from batch_nmf.nmf.plain import plainNMF
from batch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF

PERIOD = 10.4  # bp per helical turn
ALL_METHODS = ("nmf", "plain", "bnmf", "sfnmf", "oracle")
FILTER_FAMILIES = ("sigmoid", "short_depletion", "smooth", "periodic", "noisy", "empirical")


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def add_scenario_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("design")
    g.add_argument("--replicate", type=int, default=1,
                   help="Replicate number; seeds the random draws. The same replicate uses the same "
                        "random numbers in every scenario, so scenarios can be compared pairwise.")
    g.add_argument("--n_batches", type=int, default=3, help="Number of training batches (batch 0 is the reference).")
    g.add_argument("--n_per_batch", type=str, default="50",
                   help="Samples per training batch: one number, or one per batch separated by commas.")
    g.add_argument("--n_test", type=int, default=50, help="Samples in the unseen test batch (0 to skip testing).")
    g.add_argument("--case_fraction", type=str, default="0.5",
                   help="Fraction of cases per training batch: one number or one per batch. The test batch uses 0.5.")
    g = p.add_argument_group("signatures and weights")
    g.add_argument("--min_length", type=int, default=50)
    g.add_argument("--max_length", type=int, default=400)
    g.add_argument("--n_signatures", type=int, default=5, help="True number of signatures (ignored with --signatures).")
    g.add_argument("--signatures", type=str, default=None,
                   help="TSV with true signatures (rows) over fragment lengths (columns named e.g. length_100).")
    g.add_argument("--signature_seed", type=int, default=0, help="Seed for the parametric signatures (fixed across replicates).")
    g.add_argument("--tumor_signature", type=int, default=1, help="Index (0-based) of the tumour signature.")
    g.add_argument("--tumor_periodicity", type=float, default=0.3,
                   help="Amplitude of the 10.4-bp periodicity in the parametric tumour signature.")
    g.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration for the non-tumour weights.")
    g.add_argument("--tumor_fraction", type=str, default="2,3", help="Beta(a,b) for the tumour weight of cases.")
    g.add_argument("--control_tumor_fraction", type=str, default="1,50", help="Beta(a,b) for the tumour weight of controls.")
    g = p.add_argument_group("batch filters")
    g.add_argument("--filter_family", choices=FILTER_FAMILIES, default="sigmoid")
    g.add_argument("--filters", type=str, default=None, help="TSV with filters (rows) for --filter_family empirical.")
    g.add_argument("--filter_strength", type=float, default=1.0, help="Exponent applied to the filters (0 = no batch effect).")
    g.add_argument("--steepness", type=float, default=1.0, help="Multiplier for the Hill exponents of sigmoid filters.")
    g.add_argument("--gp_length_scale", type=float, default=40.0, help="Length scale (bp) of smooth random filters.")
    g.add_argument("--gp_sd", type=float, default=0.7, help="SD of log F for smooth random filters.")
    g.add_argument("--periodic_amplitude", type=float, default=0.2, help="Amplitude for --filter_family periodic.")
    g.add_argument("--noise_sd", type=float, default=0.1, help="Per-length SD of log F for --filter_family noisy.")
    g.add_argument("--sample_filter_sd", type=float, default=0.0,
                   help="SD of a smooth per-sample deviation of log F (0 = identical filter within a batch).")
    g = p.add_argument_group("counts")
    g.add_argument("--depth", type=str, default="1e6",
                   help="Fragments per sample: a number, 'a:b' for log-uniform between a and b, or 'inf' for noiseless data.")
    g.add_argument("--dm_concentration", type=float, default=0.0,
                   help="Dirichlet-multinomial concentration for overdispersion (0 = plain multinomial).")
    g.add_argument("--pseudo_count", type=float, default=1.0, help="Added to the counts before normalising (as in batch-nmf).")


def add_fit_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("fitting")
    g.add_argument("--methods", type=str, default=",".join(ALL_METHODS), help=f"Comma-separated subset of {ALL_METHODS}.")
    g.add_argument("--n_signatures_fit", type=int, default=None, help="Signatures to fit (default: the true number).")
    g.add_argument("--n_starts", type=int, default=5, help="Random starts per method.")
    g.add_argument("--n_epochs", type=int, default=50000, help="Maximum epochs per start (torch methods).")
    g.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance (see batch-nmf --tol).")
    g.add_argument("--learning_rate", type=float, default=0.5)
    g.add_argument("--lr_decrease_factor", type=float, default=0.8)
    g.add_argument("--min_learning_rate", type=float, default=1e-4)
    g.add_argument("--volume_regularization", type=float, default=-1, help="Passed to plain/bnmf/sfnmf.")
    g.add_argument("--sklearn_max_iter", type=int, default=100000)
    g.add_argument("--sklearn_tol", type=float, default=1e-9)
    g.add_argument("--tie_tol", type=float, default=1e-6, help="Relative loss difference counted as a tie between starts.")
    g.add_argument("--test_chunk_sizes", type=str, default="50",
                   help="The test batch is split into chunks of these sizes; each chunk is analysed independently.")
    g.add_argument("--max_test_chunks", type=int, default=0,
                   help="Maximum number of chunks evaluated per chunk size (0 = all, so every chunk size is "
                        "evaluated on the same test samples).")
    g.add_argument("--predict_restarts", type=int, default=0, help="Random restarts when inferring test weights.")
    g.add_argument("--predict_n_epochs", type=int, default=20000, help="Maximum epochs when inferring test weights.")
    g.add_argument("--verbose", action="store_true", help="Show the output of the fitting routines.")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="simulate", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p_sim = sub.add_parser("simulate", help="Write one simulated dataset and its ground truth.",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_scenario_args(p_sim)
    p_sim.add_argument("--npz", type=str, default=None, help="Write the dataset to this .npz file (input for run --data).")
    p_sim.add_argument("--out_dir", type=str, default=None,
                       help="Write batch-nmf input files and the ground truth as TSV files to this directory.")
    p_run = sub.add_parser("run", help="Fit and evaluate the methods on one dataset; write one TSV.",
                           formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_scenario_args(p_run)
    add_fit_args(p_run)
    p_run.add_argument("--data", type=str, default=None,
                       help="Dataset written by `simulate --npz`. If given, the design, signature, filter and "
                            "count options are ignored and taken from the file.")
    p_run.add_argument("--out", type=str, required=True, help="Output TSV.")
    return parser


def _floats(text: str, n: int, name: str) -> list[float]:
    values = [float(v) for v in str(text).split(",")]
    if len(values) == 1:
        values = values * n
    if len(values) != n:
        raise ValueError(f"--{name} needs 1 or {n} values, got {text!r}")
    return values


def _beta(text: str) -> tuple[float, float]:
    a, b = (float(v) for v in text.split(","))
    return a, b


# ---------------------------------------------------------------------------
# Generative model
# ---------------------------------------------------------------------------

def rownorm(A: np.ndarray) -> np.ndarray:
    return A / A.sum(axis=-1, keepdims=True)


def cfdna_profile(L, mono_mode, mono_sd, periodicity, di_mode, di_sd, di_weight, short_weight, short_scale):
    """A cfDNA-like length profile: asymmetric mono-nucleosome peak (wider below the mode,
    with 10.4-bp periodicity there), a di-nucleosome peak and a short-fragment tail."""
    L = np.asarray(L, dtype=float)
    sd = np.where(L < mono_mode, 1.6 * mono_sd, 0.6 * mono_sd)
    mono = np.exp(-0.5 * ((L - mono_mode) / sd) ** 2)
    wave = 1 + periodicity * np.cos(2 * np.pi * (L - mono_mode) / PERIOD)
    mono = np.where(L < mono_mode, mono * wave, mono)
    di = di_weight * np.exp(-0.5 * ((L - di_mode) / di_sd) ** 2)
    short = short_weight * np.exp(-(L - L.min()) / short_scale)
    return mono + di + short + 1e-6


def parametric_signatures(L, n_signatures, tumor_idx, tumor_periodicity, rng) -> np.ndarray:
    healthy = dict(mono_mode=167, mono_sd=10, periodicity=0.1, di_mode=332, di_sd=25, di_weight=0.06,
                   short_weight=0.01, short_scale=30)
    tumor = dict(mono_mode=150, mono_sd=14, periodicity=tumor_periodicity, di_mode=310, di_sd=25,
                 di_weight=0.04, short_weight=0.05, short_scale=40)
    sigs = []
    for c in range(n_signatures):
        if c == tumor_idx:
            params = tumor
        elif c == 0:
            params = healthy
        else:
            params = dict(mono_mode=rng.uniform(155, 178), mono_sd=rng.uniform(8, 16),
                          periodicity=rng.uniform(0, 0.25), di_mode=rng.uniform(300, 345),
                          di_sd=rng.uniform(18, 35), di_weight=rng.uniform(0.01, 0.12),
                          short_weight=rng.uniform(0, 0.03), short_scale=rng.uniform(20, 50))
        sigs.append(cfdna_profile(L, **params))
    return rownorm(np.array(sigs))


def _read_rows_over_lengths(fname: str, L: np.ndarray) -> np.ndarray:
    matrix, _, cols = read_matrix(fname)
    lengths = get_fragment_lengths(cols)
    index = {int(length): j for j, length in enumerate(lengths)}
    missing = [int(x) for x in L if int(x) not in index]
    if missing:
        raise ValueError(f"{fname} has no columns for lengths {missing[:5]}...")
    return matrix[:, [index[int(x)] for x in L]]


def gp_draws(L, n, length_scale, sd, rng) -> np.ndarray:
    """n smooth draws from a zero-mean Gaussian process with squared-exponential covariance."""
    d = L[:, None] - L[None, :]
    K = sd**2 * np.exp(-0.5 * (d / length_scale) ** 2)
    vals, vecs = np.linalg.eigh(K)
    root = vecs * np.sqrt(np.clip(vals, 0, None))
    return rng.standard_normal((n, len(L))) @ root.T


def hill_increasing(L, h, n):
    return expit(n * (np.log(L) - np.log(h)))


def hill_decreasing(L, h, n):
    return expit(n * (np.log(h) - np.log(L)))


def draw_filter(opts, L, rng, empirical=None) -> np.ndarray:
    """One batch filter, scaled to a maximum of 1."""
    fam = opts.filter_family
    s = opts.steepness
    inc = hill_increasing(L, rng.uniform(60, 120), rng.uniform(4, 12) * s)
    dec = hill_decreasing(L, rng.uniform(250, 450), rng.uniform(3, 8) * s)
    if fam == "sigmoid":
        F = inc * dec
    elif fam == "short_depletion":
        F = hill_increasing(L, rng.uniform(100, 140), rng.uniform(6, 10) * s)
    elif fam == "smooth":
        F = np.exp(gp_draws(L, 1, opts.gp_length_scale, opts.gp_sd, rng)[0])
    elif fam == "periodic":
        F = inc * dec * (1 + opts.periodic_amplitude * np.sin(2 * np.pi * L / PERIOD + rng.uniform(0, 2 * np.pi)))
    elif fam == "noisy":
        F = inc * dec * np.exp(rng.normal(0, opts.noise_sd, len(L)))
    elif fam == "empirical":
        F = empirical[rng.integers(len(empirical))]
    else:
        raise ValueError(fam)
    F = np.clip(F, 1e-300, None) ** opts.filter_strength
    return F / F.max()


def simulate_weights(n, n_signatures, case_fraction, opts, tumor_idx, rng):
    n_cases = int(round(n * case_fraction))
    is_case = rng.permutation(np.r_[np.ones(n_cases, bool), np.zeros(n - n_cases, bool)])
    tf = np.where(is_case, rng.beta(*_beta(opts.tumor_fraction), n), rng.beta(*_beta(opts.control_tumor_fraction), n))
    others = [c for c in range(n_signatures) if c != tumor_idx]
    W = np.zeros((n, n_signatures))
    W[:, tumor_idx] = tf
    W[:, others] = rng.dirichlet(np.full(len(others), opts.alpha), size=n) * (1 - tf)[:, None]
    return W, is_case


def _depths(spec: str, n: int, rng) -> np.ndarray | None:
    if spec in ("inf", "Inf", "0"):
        return None
    if ":" in spec:
        a, b = (float(v) for v in spec.split(":"))
        return np.exp(rng.uniform(np.log(a), np.log(b), n)).astype(np.int64)
    return np.full(n, int(float(spec)))


def observe(P, opts, rng):
    """Draw counts from the true probabilities P and normalise like batch-nmf.
    Returns (input matrix X, counts or None)."""
    depths = _depths(opts.depth, len(P), rng)
    if depths is None:
        return P.copy(), None
    counts = np.empty(P.shape, dtype=np.int64)
    for i, (p, d) in enumerate(zip(P, depths)):
        if opts.dm_concentration > 0:
            p = rng.gamma(np.maximum(opts.dm_concentration * p, 1e-12))
            p = p / p.sum()
        counts[i] = rng.multinomial(d, p / p.sum())
    return rownorm(counts + opts.pseudo_count), counts


def simulate(opts) -> dict:
    """Simulate one dataset. Separate random streams keep e.g. the weights identical when only
    the filters or the depth change between scenarios, and the test batch identical when only
    the training design changes (common random numbers)."""
    rep = opts.replicate
    rng_sig = np.random.default_rng([opts.signature_seed, 0])

    def streams(part):  # weights, counts, per-sample filter deviations
        return [np.random.default_rng([rep, part, i]) for i in range(3)]

    L = np.arange(opts.min_length, opts.max_length + 1, dtype=float)
    if opts.signatures:
        H = rownorm(_read_rows_over_lengths(opts.signatures, L))
    else:
        H = parametric_signatures(L, opts.n_signatures, opts.tumor_signature, opts.tumor_periodicity, rng_sig)
    C = H.shape[0]
    K = opts.n_batches
    sizes = [int(v) for v in _floats(opts.n_per_batch, K, "n_per_batch")]
    case_fr = _floats(opts.case_fraction, K, "case_fraction")
    empirical = _read_rows_over_lengths(opts.filters, L) if opts.filter_family == "empirical" else None

    # filters: reference batch unfiltered, then the other training batches, then the test batch
    F = np.ones((K + 1, len(L)))
    rng_f = np.random.default_rng([rep, 2])
    for k in range(1, K):
        F[k] = draw_filter(opts, L, rng_f, empirical)
    F[K] = draw_filter(opts, L, np.random.default_rng([rep, 3]), empirical)

    def make_set(batch_ids, case_fractions_per_sample, rng_w, rng_x, rng_h):
        Ws, cases = [], []
        for size, cf in case_fractions_per_sample:
            W_b, c_b = simulate_weights(size, C, cf, opts, opts.tumor_signature, rng_w)
            Ws.append(W_b)
            cases.append(c_b)
        W = np.vstack(Ws)
        is_case = np.concatenate(cases)
        F_samples = F[batch_ids].copy()
        if opts.sample_filter_sd > 0:
            F_samples *= np.exp(gp_draws(L, len(batch_ids), 50.0, opts.sample_filter_sd, rng_h))
        P = rownorm((W @ H) * F_samples)
        X, counts = observe(P, opts, rng_x)
        return dict(W=W, is_case=is_case, P=P, X=X, counts=counts, batch=np.asarray(batch_ids))

    train_batch = np.repeat(np.arange(K), sizes)
    train = make_set(train_batch, list(zip(sizes, case_fr)), *streams(1))
    test = make_set(np.full(opts.n_test, K), [(opts.n_test, 0.5)], *streams(4)) if opts.n_test > 0 else None
    return dict(L=L, H=H, F=F, train=train, test=test, n_batches=K, tumor_idx=opts.tumor_signature)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def _quiet(verbose):
    if verbose:
        return contextlib.nullcontext()
    return contextlib.redirect_stdout(io.StringIO())


def kl_loss(X, W, H, F_rows) -> float:
    """Mean over samples of KL(x_i || rownorm((W H)_i * F_i))."""
    P_hat = rownorm((W @ H) * F_rows)
    return float(np.mean(np.sum(xlogy(X, X) - xlogy(X, P_hat), axis=1)))


def _normalise_sklearn(W, H):
    s = H.sum(axis=1)
    s = np.where(s > 0, s, 1.0)
    return rownorm(W * s + 1e-300), H / s[:, None]


def _torch_kwargs(opts, n_epochs):
    return dict(n_epochs=n_epochs, lr=opts.learning_rate, restarts=0, loss="kullback-leibler",
                min_lr_rate=opts.min_learning_rate, lr_decrease_factor=opts.lr_decrease_factor, tol=opts.tol)


def fit_train(method, sim, opts, n_fit, seed) -> dict:
    """One start of `method` on the training data. Returns W, H and F (one row per training batch)."""
    X, batch, K, L = sim["train"]["X"], sim["train"]["batch"], sim["n_batches"], sim["L"]
    B = np.eye(K)[batch]
    torch.manual_seed(seed)
    vr = opts.volume_regularization
    with _quiet(opts.verbose), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if method == "nmf":
            model = NMF(n_components=n_fit, init="random", solver="mu", beta_loss="kullback-leibler",
                        max_iter=opts.sklearn_max_iter, tol=opts.sklearn_tol, random_state=seed)
            W, H = _normalise_sklearn(model.fit_transform(X), model.components_)
            F = np.ones((K, len(L)))
        elif method == "plain":
            W, H = plainNMF(X, n_fit, volume_regularization=vr).fit(**_torch_kwargs(opts, opts.n_epochs))[0]
            F = np.ones((K, len(L)))
        elif method == "bnmf":
            W, H, F = batchNMF(X, B, n_fit, volume_regularization=vr).fit(**_torch_kwargs(opts, opts.n_epochs))[0]
        elif method == "sfnmf":
            m = sigmoidfilterNMF(X, B, n_fit, volume_regularization=vr, lengths=L)
            W, H, F = m.fit(**_torch_kwargs(opts, opts.n_epochs))[0]
        elif method == "oracle":
            W, H, F = batchNMF(X, B, n_fit, F=sim["F"][:K]).fit(**_torch_kwargs(opts, opts.n_epochs))[0]
        else:
            raise ValueError(method)
    return dict(W=W, H=H, F=np.asarray(F), loss=kl_loss(X, W, H, F[batch]))


def predict_chunk(method, sim, X, H, opts, seed, F_true_test) -> tuple[np.ndarray, np.ndarray]:
    """Infer the weights (and one filter) for one test chunk with the signatures H fixed."""
    n, M = X.shape
    ones = np.ones((n, 1))
    torch.manual_seed(seed)
    kwargs = _torch_kwargs(opts, opts.predict_n_epochs)
    kwargs["restarts"] = opts.predict_restarts
    with _quiet(opts.verbose), warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if method == "nmf":
            W, _, _ = non_negative_factorization(X, H=H, update_H=False, n_components=H.shape[0], init="random",
                                                 solver="mu", beta_loss="kullback-leibler",
                                                 max_iter=opts.sklearn_max_iter, tol=opts.sklearn_tol,
                                                 random_state=seed)
            return rownorm(W + 1e-300), np.ones(M)
        if method == "plain":
            return plainNMF(X, H.shape[0], H).fit(**kwargs)[0], np.ones(M)
        if method == "bnmf":
            W, F = batchNMF(X, ones, H.shape[0], H).fit(**kwargs)[0]
        elif method == "sfnmf":
            W, F = sigmoidfilterNMF(X, ones, H.shape[0], H, lengths=sim["L"]).fit(**kwargs)[0]
        elif method == "oracle":
            W, F = batchNMF(X, ones, H.shape[0], H, F=F_true_test[None, :]).fit(**kwargs)[0]
        else:
            raise ValueError(method)
    return W, np.asarray(F)[0]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def to_view(W, H, f):
    """Express weights and signatures as seen through filter f (e.g. the reference batch)."""
    retained = H @ f
    return rownorm(W * retained), rownorm(H * f)


def cosine(A, B):
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    B = B / np.linalg.norm(B, axis=1, keepdims=True)
    return A @ B.T


def jsd(P, Q):
    """Jensen-Shannon divergence (natural log) between matching rows of P and Q."""
    M = 0.5 * (P + Q)
    return 0.5 * np.sum(xlogy(P, P) - xlogy(P, M), axis=-1) + 0.5 * np.sum(xlogy(Q, Q) - xlogy(Q, M), axis=-1)


def match_signatures(H_true, H_est):
    """perm[c] = estimated signature matched to true signature c (None if not matched)."""
    rows, cols = linear_sum_assignment(-cosine(H_true, H_est))
    perm = [None] * H_true.shape[0]
    for r, c in zip(rows, cols):
        perm[r] = c
    return perm


def w_metrics(W_true, W_est, tumor_idx) -> dict:
    err = W_est - W_true
    sst = np.sum((W_true - W_true.mean(axis=0)) ** 2)
    return dict(w_mae=np.abs(err).mean(), w_rmse=np.sqrt((err**2).mean()),
                w_r2=1 - np.sum(err**2) / sst if sst > 0 else np.nan,
                tumor_w_mae=np.abs(err[:, tumor_idx]).mean())


def auc(scores, is_case) -> float:
    if is_case.all() or not is_case.any():
        return np.nan
    return roc_auc_score(is_case, scores)


def filter_log_rmse(r_est, r_true, weights) -> float:
    """Weighted RMSE of log filter ratios after removing the (unidentified) overall level."""
    w = weights / weights.sum()
    d = np.log(np.clip(r_est, 1e-300, None)) - np.log(np.clip(r_true, 1e-300, None))
    d = d - np.sum(w * d)
    return float(np.sqrt(np.sum(w * d**2)))


def corrected_jsd(X, F_rows_est, f0_est, P_true_ref) -> float:
    corrected = rownorm(X / F_rows_est * f0_est)
    return float(np.mean(jsd(corrected, P_true_ref)))


def evaluate_train(fit, sim, n_fit) -> dict:
    tr, H_true, tumor = sim["train"], sim["H"], sim["tumor_idx"]
    f0 = fit["F"][0]
    W_v, H_v = to_view(fit["W"], fit["H"], f0)
    out = dict(kl_loss=fit["loss"], n_samples_eval=len(tr["X"]))
    perm = match_signatures(H_true, H_v)
    if n_fit == H_true.shape[0]:
        W_m, H_m = W_v[:, perm], H_v[perm]
        out.update(w_metrics(tr["W"], W_m, tumor))
        out.update(h_cos=np.mean(np.diag(cosine(H_true, H_m))), h_jsd=np.mean(jsd(H_true, H_m)),
                   h_rmse=np.sqrt(np.mean((H_true - H_m) ** 2)))
        tumor_col = perm[tumor]
    else:
        tumor_col = int(np.argmax(cosine(H_true[[tumor]], H_v)[0]))
    out["tumor_auc"] = auc(W_v[:, tumor_col], tr["is_case"])
    K = sim["n_batches"]
    if K > 1:
        errs = []
        for k in range(1, K):
            idx = tr["batch"] == k
            errs.append(filter_log_rmse(fit["F"][k] / f0, sim["F"][k] / sim["F"][0], tr["P"][idx].mean(axis=0)))
        out["filter_log_rmse"] = float(np.mean(errs))
    P_true_ref = rownorm(tr["W"] @ (H_true * sim["F"][0]))
    out["corrected_jsd"] = corrected_jsd(tr["X"], fit["F"][tr["batch"]], f0, P_true_ref)
    out["_perm"], out["_tumor_col"] = perm, tumor_col
    return out


def spread(fits, best, tie_tol, H_true) -> dict:
    """How different are starts whose loss ties with the best start?"""
    tied = [f for f in fits if (f["loss"] - best["loss"]) <= tie_tol * abs(best["loss"])]
    W_b, H_b = to_view(best["W"], best["H"], best["F"][0])
    w_spread, h_cos = 0.0, 1.0
    for f in tied:
        if f is best:
            continue
        W_f, H_f = to_view(f["W"], f["H"], f["F"][0])
        perm = match_signatures(H_b, H_f)
        if None in perm:
            continue
        w_spread = max(w_spread, float(np.abs(W_f[:, perm] - W_b).mean()))
        h_cos = min(h_cos, float(np.min(np.diag(cosine(H_b, H_f[perm])))))
    return dict(n_tied_starts=len(tied), spread_w_mae=w_spread, spread_h_cos=h_cos)


def evaluate_test(method, best, train_eval, sim, opts, n_fit, chunk_size) -> dict:
    te, H_true, tumor, K = sim["test"], sim["H"], sim["tumor_idx"], sim["n_batches"]
    n = len(te["X"])
    n_chunks = n // chunk_size
    if opts.max_test_chunks > 0:
        n_chunks = min(n_chunks, opts.max_test_chunks)
    if n_chunks == 0:
        return {}
    f0 = best["F"][0]
    W_est, F_est, idx_all, losses, frmse = [], [], [], [], []
    for j in range(n_chunks):
        idx = np.arange(j * chunk_size, (j + 1) * chunk_size)
        X = te["X"][idx]
        W, F = predict_chunk(method, sim, X, best["H"], opts, seed=10_000 * opts.replicate + j, F_true_test=sim["F"][K])
        W_est.append(W)
        F_est.append(np.tile(F, (len(idx), 1)))
        idx_all.append(idx)
        losses.append(kl_loss(X, W, best["H"], np.tile(F, (len(idx), 1))) * len(idx))
        frmse.append(filter_log_rmse(F / f0, sim["F"][K] / sim["F"][0], te["P"][idx].mean(axis=0)))
    idx = np.concatenate(idx_all)
    W_est, F_est = np.vstack(W_est), np.vstack(F_est)
    W_v, _ = to_view(W_est, best["H"], f0)
    out = dict(kl_loss=float(np.sum(losses) / len(idx)), n_samples_eval=len(idx), filter_log_rmse=float(np.mean(frmse)))
    perm, tumor_col = train_eval["_perm"], train_eval["_tumor_col"]
    if n_fit == H_true.shape[0]:
        out.update(w_metrics(te["W"][idx], W_v[:, perm], tumor))
    out["tumor_auc"] = auc(W_v[:, tumor_col], te["is_case"][idx])
    P_true_ref = rownorm(te["W"][idx] @ (H_true * sim["F"][0]))
    out["corrected_jsd"] = corrected_jsd(te["X"][idx], F_est, f0, P_true_ref)
    return out


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

SCENARIO_COLUMNS = ("replicate", "n_batches", "n_per_batch", "n_test", "case_fraction", "n_signatures",
                    "signatures", "tumor_periodicity", "alpha", "tumor_fraction", "filter_family", "filter_strength",
                    "steepness", "sample_filter_sd", "depth", "dm_concentration", "n_signatures_fit", "n_starts",
                    "volume_regularization")
METRIC_COLUMNS = ("method", "set", "chunk_size", "n_samples_eval", "kl_loss", "w_mae", "w_rmse", "w_r2", "tumor_w_mae",
                  "tumor_auc", "h_cos", "h_jsd", "h_rmse", "filter_log_rmse", "corrected_jsd", "n_tied_starts",
                  "spread_w_mae", "spread_h_cos", "runtime_s")


def run(opts) -> pd.DataFrame:
    methods = [m.strip() for m in opts.methods.split(",") if m.strip()]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        raise ValueError(f"unknown methods {sorted(unknown)}")
    if opts.data:
        sim, scenario = load_dataset(opts.data)
        opts.replicate = scenario["replicate"]  # seeds of the fits
    else:
        sim, scenario = simulate(opts), scenario_values(opts)
    C_true = sim["H"].shape[0]
    n_fit = opts.n_signatures_fit or C_true
    chunk_sizes = [int(v) for v in opts.test_chunk_sizes.split(",") if v.strip()]
    scenario = {c: scenario.get(c, getattr(opts, c, None)) for c in SCENARIO_COLUMNS}
    scenario.update(n_signatures_fit=n_fit, n_starts=opts.n_starts, volume_regularization=opts.volume_regularization)
    scenario["n_signatures"] = C_true
    rows = []
    for method in methods:
        t0 = time.time()
        fits = [fit_train(method, sim, opts, n_fit, seed=1000 * opts.replicate + s) for s in range(opts.n_starts)]
        best = min(fits, key=lambda f: f["loss"])
        train_eval = evaluate_train(best, sim, n_fit)
        row = dict(scenario, method=method, set="train", chunk_size=np.nan,
                   **{k: v for k, v in train_eval.items() if not k.startswith("_")},
                   **spread(fits, best, opts.tie_tol, sim["H"]), runtime_s=time.time() - t0)
        rows.append(row)
        print(f"{method:7s} train: w_mae={row.get('w_mae', np.nan):.4f} tumor_auc={row['tumor_auc']:.3f} "
              f"kl={row['kl_loss']:.3e} ({row['runtime_s']:.0f} s)", flush=True)
        if sim["test"] is None:
            continue
        for size in chunk_sizes:
            t0 = time.time()
            test_eval = evaluate_test(method, best, train_eval, sim, opts, n_fit, size)
            if not test_eval:
                continue
            rows.append(dict(scenario, method=method, set="test", chunk_size=size, **test_eval,
                             runtime_s=time.time() - t0))
            print(f"{method:7s} test (chunks of {size}): w_mae={test_eval.get('w_mae', np.nan):.4f} "
                  f"tumor_auc={test_eval['tumor_auc']:.3f}", flush=True)
    df = pd.DataFrame(rows).reindex(columns=[*SCENARIO_COLUMNS, *METRIC_COLUMNS])
    Path(opts.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(opts.out, sep="\t", index=False)
    return df


DATASET_KEYS = ("W", "is_case", "P", "X", "batch")


def scenario_values(opts) -> dict:
    """The simulation options that define a dataset (written to the .npz and to the results)."""
    parser = argparse.ArgumentParser(add_help=False)
    add_scenario_args(parser)
    return {a.dest: getattr(opts, a.dest) for a in parser._actions}


def save_dataset(sim, opts, path) -> None:
    """Write everything `run --data` needs to one .npz file."""
    arrays = {"L": sim["L"], "H": sim["H"], "F": sim["F"], "n_batches": sim["n_batches"], "tumor_idx": sim["tumor_idx"],
              "scenario": json.dumps(scenario_values(opts))}
    for part in ("train", "test"):
        if sim[part] is not None:
            arrays.update({f"{part}_{k}": sim[part][k] for k in DATASET_KEYS})
    np.savez_compressed(path, **arrays)


def load_dataset(path) -> tuple[dict, dict]:
    with np.load(path) as f:
        sim = {"L": f["L"], "H": f["H"], "F": f["F"], "n_batches": int(f["n_batches"]),
               "tumor_idx": int(f["tumor_idx"])}
        for part in ("train", "test"):
            sim[part] = {k: f[f"{part}_{k}"] for k in DATASET_KEYS} if f"{part}_X" in f else None
        scenario = json.loads(str(f["scenario"]))
    return sim, scenario


def write_dataset(opts) -> None:
    """Write the simulated data to an .npz file and/or as TSV files."""
    if not opts.npz and not opts.out_dir:
        raise SystemExit("simulate: give --npz and/or --out_dir")
    sim = simulate(opts)
    if opts.npz:
        save_dataset(sim, opts, opts.npz)
    if opts.out_dir:
        write_tsv_files(sim, opts)


def write_tsv_files(sim, opts) -> None:
    """Write the simulated data as batch-nmf input files plus the ground truth."""
    out = Path(opts.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cols = [f"length_{int(x)}" for x in sim["L"]]
    sig_names = [f"Sig{c + 1}" for c in range(sim["H"].shape[0])]
    K = sim["n_batches"]

    def save(df, name, index_name):
        df.index.name = index_name
        df.to_csv(out / name, sep="\t")

    for name, part, batch_names in (("train", sim["train"], [f"batch{k}" for k in range(K)]),
                                    ("test", sim["test"], [f"batch{k}" for k in range(K + 1)])):
        if part is None:
            continue
        samples = [f"{name}_{i + 1}" for i in range(len(part["X"]))]
        data = part["counts"] if part["counts"] is not None else part["X"]
        save(pd.DataFrame(data, index=samples, columns=cols), f"{name}_input.tsv", "sample_id")
        pd.DataFrame({"sample_id": samples, "batch": [batch_names[b] for b in part["batch"]]}).to_csv(
            out / f"{name}_batches.tsv", sep="\t", index=False, header=False)
        truth = pd.DataFrame(part["W"], index=samples, columns=sig_names)
        truth["is_case"] = part["is_case"].astype(int)
        save(truth, f"{name}_true_weights.tsv", "sample_id")
    save(pd.DataFrame(sim["H"], index=sig_names, columns=cols), "true_signatures.tsv", "signature")
    save(pd.DataFrame(sim["F"], index=[f"batch{k}" for k in range(K)] + ["test"], columns=cols),
         "true_filters.tsv", "batch_id")
    print(f"wrote simulated data to {out}/ (train_input.tsv and train_batches.tsv can be given to batch-nmf fit; "
          f"use --pseudo_count {opts.pseudo_count if sim['train']['counts'] is not None else 0})")


def main(args: list[str] | None = None) -> int:
    opts = get_parser().parse_args(args)
    if opts.command == "simulate":
        write_dataset(opts)
    else:
        run(opts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
