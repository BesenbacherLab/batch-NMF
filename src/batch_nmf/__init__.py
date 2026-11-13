"""
batch-nmf: NMF with batch effects for cfDNA fragment length profiles.

The models (`batchNMF`, `sigmoidfilterNMF`, `plainNMF`) and the input helpers can be imported
directly from `batch_nmf`.
"""

from importlib.metadata import PackageNotFoundError, version

from batch_nmf.io_utils import get_fragment_lengths, read_batch_file, read_matrix
from batch_nmf.nmf.batch import batchNMF
from batch_nmf.nmf.plain import plainNMF
from batch_nmf.nmf.scipy_nmf import fit_scipy_nmf
from batch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF

__all__ = [
    "batchNMF",
    "fit_scipy_nmf",
    "get_fragment_lengths",
    "plainNMF",
    "read_batch_file",
    "read_matrix",
    "sigmoidfilterNMF",
]

# The version is set in pyproject.toml (change it with `uv version`); read it from the installed package.
try:
    __version__ = version("batch-nmf")
except PackageNotFoundError:  # source tree that has not been installed
    __version__ = "unknown"
