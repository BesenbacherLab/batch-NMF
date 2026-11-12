"""Tests for the `cli` module."""

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from batch_nmf import cli
from batch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF


#def test_main():
#    """Basic CLI test."""
#    assert cli.main([]) == 0


def test_show_help(capsys):
    """
    Show help.

    Arguments:
        capsys: Pytest fixture to capture output.
    """
    with pytest.raises(SystemExit):
        cli.main(["-h"])
    captured = capsys.readouterr()
    assert "batch-nmf" in captured.out


def test_version_is_the_installed_version(capsys):
    """`--version` reports the version in pyproject.toml (via the installed package metadata)."""
    from importlib.metadata import version

    with pytest.raises(SystemExit):
        cli.main(["--version"])
    assert capsys.readouterr().out.strip() == f"batch-nmf {version('batch-nmf')}"


LENGTH_COLS = [f"length_{i}" for i in range(100, 112)]


def _write_tsv(path, matrix, index, columns, index_name):
    df = pd.DataFrame(matrix, index=index, columns=columns)
    df.index.name = index_name
    df.to_csv(path, sep="\t")
    return str(path)


@pytest.fixture
def input_files(tmp_path):
    """Write a small count matrix (6 samples x 12 lengths) and its true signatures."""
    W = np.array([[0.9, 0.1], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.1, 0.9], [0.6, 0.4]])
    H = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]], dtype=float)
    H = H / H.sum(1, keepdims=True)
    counts = np.round(10000 * W @ H)
    samples = [f"sample_{i}" for i in range(1, 7)]
    matrix = _write_tsv(tmp_path / "input.tsv", counts, samples, LENGTH_COLS, "sample_id")
    return tmp_path, matrix, H


def _predict_scipy(tmp_path, matrix, signature_file, name):
    exposure_file = str(tmp_path / f"{name}_exposures.tsv")
    assert cli.main([
        "predict", "--method", "scipy", "--input_matrix", matrix,
        "--signature_file", signature_file, "--exposure_file", exposure_file,
    ]) == 0
    return pd.read_csv(exposure_file, sep="\t", index_col=0)


def test_predict_reorders_signature_columns(input_files):
    tmp_path, matrix, H = input_files
    sig = _write_tsv(tmp_path / "sig.tsv", H, ["Sig1", "Sig2"], LENGTH_COLS, "signature")
    perm = np.random.default_rng(1).permutation(len(LENGTH_COLS))
    sig_perm = _write_tsv(tmp_path / "sig_perm.tsv", H[:, perm], ["Sig1", "Sig2"],
                          [LENGTH_COLS[i] for i in perm], "signature")
    expected = _predict_scipy(tmp_path, matrix, sig, "ordered")
    result = _predict_scipy(tmp_path, matrix, sig_perm, "permuted")
    assert_allclose(result.to_numpy(), expected.to_numpy())


@pytest.mark.parametrize(
    "sig_cols",
    [
        [f"length_{i}" for i in range(101, 113)],  # same number of columns, shifted lengths
        LENGTH_COLS[:-1],  # one column too few
    ],
)
def test_predict_rejects_mismatched_signature_columns(input_files, sig_cols):
    tmp_path, matrix, H = input_files
    sig = _write_tsv(tmp_path / "sig_bad.tsv", H[:, : len(sig_cols)], ["Sig1", "Sig2"], sig_cols, "signature")
    with pytest.raises(ValueError, match="do not match"):
        _predict_scipy(tmp_path, matrix, sig, "bad")


@pytest.fixture
def recorded_lengths(monkeypatch):
    """Replace sigmoidfilterNMF in the CLI with a subclass that records the fragment lengths."""
    recorded = []

    class RecordingSFNMF(sigmoidfilterNMF):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            recorded.append(self.bp.numpy().copy())

    monkeypatch.setattr(cli, "sigmoidfilterNMF", RecordingSFNMF)
    return recorded


def _fit_sfnmf(tmp_path, matrix, *extra):
    return cli.main([
        "fit", "--method", "sfNMF", "--all_separate_batch", "--input_matrix", matrix,
        "--signature_file", str(tmp_path / "sig_out.tsv"), "--exposure_file", str(tmp_path / "exp_out.tsv"),
        "--n_restarts", "0", "--n_epochs", "2", *extra,
    ])


def test_sfnmf_lengths_from_column_names(input_files, recorded_lengths):
    tmp_path, matrix, _ = input_files
    assert _fit_sfnmf(tmp_path, matrix) == 0
    assert_array_equal(recorded_lengths[-1], np.arange(100, 112))


def test_sfnmf_min_length_option(input_files, recorded_lengths):
    tmp_path, matrix, _ = input_files
    assert _fit_sfnmf(tmp_path, matrix, "--min_length", "80") == 0
    assert_array_equal(recorded_lengths[-1], np.arange(80, 92))


def test_sfnmf_unparsable_column_names(tmp_path):
    counts = np.full((3, 4), 100.0)
    matrix = _write_tsv(tmp_path / "input.tsv", counts, ["s1", "s2", "s3"], ["a", "b", "c", "d"], "sample_id")
    with pytest.raises(ValueError, match="min_length"):
        _fit_sfnmf(tmp_path, matrix)


def test_epoch_options_reach_the_model(input_files):
    tmp_path, matrix, _ = input_files
    logfile = tmp_path / "log.tsv"
    assert cli.main([
        "fit", "--method", "bNMF", "--all_separate_batch", "--input_matrix", matrix,
        "--signature_file", str(tmp_path / "sig_out.tsv"), "--exposure_file", str(tmp_path / "exp_out.tsv"),
        "--n_restarts", "2", "--n_epochs_restart", "7", "--n_epochs", "5", "--tol", "0",
        "--logfile", str(logfile),
    ]) == 0
    log = pd.read_csv(logfile, sep="\t")
    assert log.groupby("run")["epoch"].max().to_dict() == {"final": 4, "initial_0": 6, "initial_1": 6}


def _fit_outputs(tmp_path, matrix, method, seed, name):
    """Run a short fit and return the output files as arrays."""
    files = {k: str(tmp_path / f"{name}_{k}.tsv") for k in ("sig", "exp", "filt")}
    batch = [] if method == "scipy" else ["--all_separate_batch", "--filter_file", files["filt"]]
    assert cli.main([
        "fit", "--method", method, *batch, "--input_matrix", matrix,
        "--signature_file", files["sig"], "--exposure_file", files["exp"],
        "--n_restarts", "2", "--n_epochs_restart", "30", "--n_epochs", "30", "--tol", "0", "--seed", str(seed),
    ]) == 0
    keys = ("sig", "exp") if method == "scipy" else ("sig", "exp", "filt")
    return [pd.read_csv(files[k], sep="\t", index_col=0).to_numpy() for k in keys]


@pytest.mark.parametrize("method", ["bNMF", "sfNMF", "scipy"])
def test_seed_makes_fit_repeatable(input_files, method):
    tmp_path, matrix, _ = input_files
    first = _fit_outputs(tmp_path, matrix, method, 1, "a")
    again = _fit_outputs(tmp_path, matrix, method, 1, "b")
    other = _fit_outputs(tmp_path, matrix, method, 2, "c")
    for x, y in zip(first, again):
        assert_array_equal(x, y)
    assert not all(np.array_equal(x, z) for x, z in zip(first, other))


def test_seed_makes_predict_repeatable(input_files):
    tmp_path, matrix, H = input_files
    sig = _write_tsv(tmp_path / "sig.tsv", H, ["Sig1", "Sig2"], LENGTH_COLS, "signature")

    def predict(name):
        exposure_file = str(tmp_path / f"{name}_exp.tsv")
        assert cli.main([
            "predict", "--method", "bNMF", "--all_separate_batch", "--input_matrix", matrix,
            "--signature_file", sig, "--exposure_file", exposure_file, "--filter_file", str(tmp_path / f"{name}_filt.tsv"),
            "--n_restarts", "1", "--n_epochs_restart", "30", "--n_epochs", "30", "--tol", "0", "--seed", "3",
        ]) == 0
        return pd.read_csv(exposure_file, sep="\t", index_col=0).to_numpy()

    assert_array_equal(predict("a"), predict("b"))
