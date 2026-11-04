"""Tests for the `io_utils` module."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from batch_nmf.io_utils import align_columns, get_fragment_lengths


def test_fragment_lengths_from_column_names():
    cols = ["length_100", "length_101", "length_102"]
    assert_array_equal(get_fragment_lengths(cols), [100, 101, 102])


def test_fragment_lengths_from_plain_numbers():
    assert_array_equal(get_fragment_lengths(["30", "31", "33"]), [30, 31, 33])


def test_fragment_lengths_min_length_overrides_names():
    cols = ["length_100", "length_101", "length_102"]
    assert_array_equal(get_fragment_lengths(cols, min_length=80), [80, 81, 82])


def test_fragment_lengths_unparsable_names():
    with pytest.raises(ValueError, match="min_length"):
        get_fragment_lengths(["short", "long"])


def test_fragment_lengths_duplicates():
    with pytest.raises(ValueError, match="not unique"):
        get_fragment_lengths(["a_100", "b_100"])


def test_align_columns_identical():
    m = np.arange(6.0).reshape(2, 3)
    assert align_columns(m, ["a", "b", "c"], ["a", "b", "c"], "sig.tsv") is m


def test_align_columns_reorders():
    m = np.array([[1.0, 2.0, 3.0]])
    out = align_columns(m, ["c", "a", "b"], ["a", "b", "c"], "sig.tsv")
    assert_array_equal(out, [[2.0, 3.0, 1.0]])


@pytest.mark.parametrize(
    "sig_cols",
    [
        ["a", "b", "x"],  # different names
        ["a", "b"],  # too few
        ["a", "b", "c", "d"],  # too many
    ],
)
def test_align_columns_mismatch(sig_cols):
    m = np.ones((1, len(sig_cols)))
    with pytest.raises(ValueError, match="do not match"):
        align_columns(m, sig_cols, ["a", "b", "c"], "sig.tsv")
