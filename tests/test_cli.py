"""Tests for the `cli` module."""

import pytest

from batch_nmf import cli


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
    assert "batch_nmf" in captured.out
