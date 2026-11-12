"""
batch-nmf package.

Train NMF (and extensions of NMF) using pytorch
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = []

# The version is set in pyproject.toml (change it with `uv version`); read it from the installed package.
try:
    __version__ = version("batch-nmf")
except PackageNotFoundError:  # source tree that has not been installed
    __version__ = "unknown"
