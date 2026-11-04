"""Smoke test for scripts/convergence_benchmark.py."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).parent.parent / "scripts" / "convergence_benchmark.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("convergence_benchmark", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convergence_benchmark_runs(tmp_path, capsys):
    rng = np.random.default_rng(0)
    counts = pd.DataFrame(
        rng.integers(50, 500, size=(6, 10)),
        index=[f"s{i}" for i in range(6)],
        columns=[f"length_{i}" for i in range(100, 110)],
    )
    counts.index.name = "sample_id"
    counts.to_csv(tmp_path / "input.tsv", sep="\t")
    (tmp_path / "batches.tsv").write_text("".join(f"s{i}\tb{i % 2}\n" for i in range(6)), encoding="utf-8")
    out = tmp_path / "trajectory.tsv"

    for method in ("bNMF", "sfNMF"):
        assert _load_script().main([
            "fit", "--method", method, "--input_matrix", str(tmp_path / "input.tsv"),
            "--batch_file", str(tmp_path / "batches.tsv"),
            "--n_starts", "2", "--n_epochs", "300", "--check_every", "100", "--out", str(out),
        ]) == 0
        text = capsys.readouterr().out
        assert "Convergence check" in text
        assert "Restarts" in text
        traj = pd.read_csv(out, sep="\t")
        assert list(traj["epoch"].unique()) == [0, 100, 200, 300]
        assert sorted(traj["seed"].unique()) == [1, 2]
