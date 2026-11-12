"""Tests for utils/simulate.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from batch_nmf import cli

SCRIPT = Path(__file__).parent.parent / "utils" / "simulate.py"


@pytest.fixture(scope="module")
def sim():
    spec = importlib.util.spec_from_file_location("simulate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SMALL = ["--min_length", "100", "--max_length", "160", "--n_signatures", "3", "--n_batches", "2",
         "--n_per_batch", "8", "--n_test", "4"]


def _opts(sim, command, *extra):
    return sim.get_parser().parse_args([command, *SMALL, *extra])


def test_view_is_invariant_to_moving_a_common_filter(sim):
    """Moving a filter g shared by all batches from the filters into H (and adjusting W)
    leaves the model unchanged, and the batch view must not change either."""
    rng = np.random.default_rng(0)
    W = sim.rownorm(rng.random((6, 3)))
    H = sim.rownorm(rng.random((3, 20)))
    F = rng.uniform(0.2, 1.0, (2, 20))
    g = rng.uniform(0.3, 1.0, 20)
    H2 = sim.rownorm(H * g)
    W2 = sim.rownorm(W * (H @ g))
    F2 = F / g
    for k in range(2):
        assert np.allclose(sim.rownorm((W @ H) * F[k]), sim.rownorm((W2 @ H2) * F2[k]))
        Wv, Hv = sim.to_view(W, H, F[k])
        Wv2, Hv2 = sim.to_view(W2, H2, F2[k])
        assert np.allclose(Wv, Wv2)
        assert np.allclose(Hv, Hv2)


def test_filter_error_ignores_overall_level(sim):
    r = np.linspace(0.2, 1.0, 30)
    w = np.ones(30)
    assert sim.filter_log_rmse(3.0 * r, r, w) == pytest.approx(0.0, abs=1e-12)
    assert sim.filter_log_rmse(r**2, r, w) > 0.1


@pytest.mark.parametrize("family", ["sigmoid", "short_depletion", "smooth", "periodic", "noisy"])
def test_simulate(sim, family):
    data = sim.simulate(_opts(sim, "simulate", "--out_dir", "unused", "--filter_family", family,
                              "--sample_filter_sd", "0.1"))
    tr, te = data["train"], data["test"]
    assert data["H"].shape == (3, 61)
    assert np.allclose(data["F"][0], 1.0)  # reference batch is unfiltered
    assert data["F"].shape == (3, 61)  # two training batches + the test batch
    assert np.all(data["F"] > 0) and np.allclose(data["F"].max(axis=1), 1.0)
    assert np.allclose(tr["X"].sum(axis=1), 1.0) and np.allclose(tr["W"].sum(axis=1), 1.0)
    assert tr["X"].shape == (16, 61) and te["X"].shape == (4, 61)
    assert tr["is_case"].sum() == 8
    assert list(tr["batch"]) == [0] * 8 + [1] * 8


def test_same_replicate_same_weights_across_scenarios(sim):
    a = sim.simulate(_opts(sim, "simulate", "--out_dir", "x", "--depth", "1e5"))
    b = sim.simulate(_opts(sim, "simulate", "--out_dir", "x", "--depth", "inf", "--filter_family", "smooth"))
    assert np.allclose(a["train"]["W"], b["train"]["W"])
    assert np.allclose(b["train"]["X"], b["train"]["P"])  # noiseless


def test_run_smoke(sim, tmp_path):
    out = tmp_path / "res.tsv"
    sim.run(_opts(sim, "run", "--n_starts", "2", "--n_epochs", "200", "--predict_n_epochs", "100",
                  "--sklearn_max_iter", "200", "--test_chunk_sizes", "4,1", "--out", str(out)))
    res = pd.read_csv(out, sep="\t")
    assert set(res["method"]) == {"nmf", "plain", "bnmf", "sfnmf", "oracle"}
    assert len(res) == 5 * 3  # train + two chunk sizes per method
    train = res[res["set"] == "train"].set_index("method")
    assert train.loc["oracle", "filter_log_rmse"] == pytest.approx(0.0, abs=1e-9)
    assert train["w_mae"].notna().all() and (train["n_tied_starts"] >= 1).all()
    test = res[res["set"] == "test"]
    assert (test["n_samples_eval"] == 4).all()  # every chunk size is evaluated on all test samples


def test_misspecified_number_of_signatures(sim, tmp_path):
    out = tmp_path / "res.tsv"
    sim.run(_opts(sim, "run", "--methods", "bnmf", "--n_signatures_fit", "2", "--n_starts", "1", "--n_epochs", "100",
                  "--predict_n_epochs", "50", "--test_chunk_sizes", "4", "--out", str(out)))
    res = pd.read_csv(out, sep="\t")
    assert res["w_mae"].isna().all() and res["kl_loss"].notna().all()


def test_simulated_files_work_with_cli(sim, tmp_path):
    sim.main(["simulate", *SMALL, "--out_dir", str(tmp_path)])
    for name in ("train_input.tsv", "train_batches.tsv", "train_true_weights.tsv", "test_input.tsv",
                 "true_signatures.tsv", "true_filters.tsv"):
        assert (tmp_path / name).exists()
    assert cli.main([
        "fit", "--input_matrix", str(tmp_path / "train_input.tsv"), "--batch_file", str(tmp_path / "train_batches.tsv"),
        "--n_signatures", "3", "--signature_file", str(tmp_path / "sig.tsv"), "--exposure_file", str(tmp_path / "exp.tsv"),
        "--n_restarts", "0", "--n_epochs", "5",
    ]) == 0


def test_dataset_round_trip(sim, tmp_path):
    """simulate --npz followed by run --data gives the same results as run on its own."""
    npz = tmp_path / "data.npz"
    sim.main(["simulate", *SMALL, "--replicate", "3", "--npz", str(npz)])
    data, scenario = sim.load_dataset(npz)
    direct = sim.simulate(_opts(sim, "simulate", "--replicate", "3"))
    for part in ("train", "test"):
        for key in sim.DATASET_KEYS:
            assert np.array_equal(data[part][key], direct[part][key])
    assert np.array_equal(data["F"], direct["F"]) and scenario["replicate"] == 3

    fit = ["--methods", "bnmf", "--n_starts", "1", "--n_epochs", "50", "--predict_n_epochs", "20",
           "--test_chunk_sizes", "4"]
    sim.main(["run", "--data", str(npz), *fit, "--out", str(tmp_path / "a.tsv")])
    sim.main(["run", *SMALL, "--replicate", "3", *fit, "--out", str(tmp_path / "b.tsv")])
    a = pd.read_csv(tmp_path / "a.tsv", sep="\t")
    b = pd.read_csv(tmp_path / "b.tsv", sep="\t")
    assert list(a.columns) == list(b.columns)
    assert a["replicate"].tolist() == [3, 3] and a["n_batches"].tolist() == [2, 2]
    assert np.allclose(a["kl_loss"], b["kl_loss"], rtol=1e-6)


def test_test_batch_does_not_depend_on_training_design(sim):
    a = sim.simulate(_opts(sim, "simulate", "--out_dir", "x"))
    b = sim.simulate(_opts(sim, "simulate", "--out_dir", "x", "--n_batches", "3", "--n_per_batch", "5"))
    assert np.allclose(a["test"]["X"], b["test"]["X"])
    assert np.allclose(a["F"][-1], b["F"][-1])
    assert np.allclose(a["F"][1], b["F"][1])  # first filtered training batch is also unchanged


def test_w_tvd(sim):
    W_true = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]])
    W_est = np.array([[0.8, 0.2, 0.0], [0.5, 0.3, 0.2]])
    m = sim.w_metrics(W_true, W_est, tumor_idx=1)
    assert m["w_tvd"] == pytest.approx(0.2)  # 20% of each sample's weight is misassigned
    assert m["w_mae"] == pytest.approx(0.8 / 6)


def test_corrected_jsd_ignores_lengths_without_signal(sim):
    """With the true filter as the estimate, the corrected profile should match the truth, even
    though the observed data have a pseudo-count floor where the filter removes everything."""
    L = np.arange(50, 151, dtype=float)
    P = sim.rownorm(np.exp(-0.5 * ((L - 120) / 20) ** 2)[None, :])
    F = np.vstack([np.ones_like(L), sim.hill_increasing(L, 100, 30)])  # reference + strong short depletion
    X = sim.rownorm(1e6 * sim.rownorm(P * F[1]) + 1)  # counts + pseudo-count, normalised
    batch = np.array([1])
    unmasked = sim.corrected_jsd(X, F[batch], F[0], P, np.ones_like(X, dtype=bool))
    masked = sim.corrected_jsd(X, F[batch], F[0], P, sim.retained_lengths(F, batch, 0.01))
    assert unmasked > 0.1
    assert masked < 1e-3
    assert sim.retained_lengths(F, np.array([0]), 0.01).all()  # nothing is dropped in the reference batch


def test_evaluate_reproduces_run(sim, tmp_path):
    npz = tmp_path / "data.npz"
    sim.main(["simulate", *SMALL, "--npz", str(npz)])
    fit = ["--methods", "bnmf,sfnmf,nmf", "--n_starts", "3", "--n_epochs", "100", "--predict_n_epochs", "30",
           "--sklearn_max_iter", "100", "--test_chunk_sizes", "4,2"]
    sim.main(["run", "--data", str(npz), *fit, "--out", str(tmp_path / "run.tsv"), "--fits", str(tmp_path / "fits.npz")])
    sim.main(["evaluate", "--data", str(npz), "--fits", str(tmp_path / "fits.npz"), "--out", str(tmp_path / "ev.tsv")])
    a = pd.read_csv(tmp_path / "run.tsv", sep="\t")
    b = pd.read_csv(tmp_path / "ev.tsv", sep="\t")
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 3 * 3  # train + two chunk sizes for each method

    # a looser tie tolerance only changes the spread columns
    sim.main(["evaluate", "--data", str(npz), "--fits", str(tmp_path / "fits.npz"), "--tie_tol", "1",
              "--out", str(tmp_path / "ev2.tsv")])
    c = pd.read_csv(tmp_path / "ev2.tsv", sep="\t")
    changed = [col for col in a.columns if not a[col].equals(c[col])]
    assert set(changed) <= {"tie_tol", "n_tied_starts", "spread_w_tvd", "spread_w_mae", "spread_h_cos"}
    assert (c.loc[c.set == "train", "n_tied_starts"] == 3).all()
