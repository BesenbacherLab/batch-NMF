---
title: Python usage
---

# Python usage

The models can also be used from Python. They take the input matrix as a numpy array with one row per sample, normalised so that each row sums to 1. This example fits bNMF with two signatures to the example data in the repository:

```python
import torch
from batch_nmf import batchNMF, read_batch_file, read_matrix

counts, samples, columns = read_matrix("test_data/length_data.tsv")
X = counts + 1                        # pseudo-count, as in the command line tool
X = X / X.sum(axis=1, keepdims=True)  # each row sums to 1
B, batches = read_batch_file("test_data/batch_file.tsv", samples)

torch.manual_seed(1)                  # makes the fit repeatable
model = batchNMF(X, B, n_signatures=2)
(W, H, F), loss, loss_without_penalty, penalty = model.fit()
```

`W` holds the weights of the signatures in each sample (each row sums to 1), `H` the signatures and `F` the filter of each batch. The first batch in `batches` is the reference batch, and its filter is 1.

`fit` takes the same options as the command line tool, with the same defaults, for example `model.fit(restarts=10, n_epochs=500000)`. See the [API reference](reference/api.md) for all options.

## Weights of known signatures in new samples

To fit only the weights and a filter for each batch, as `batch-nmf predict` does, give the signatures as `H`:

```python
new_model = batchNMF(X_new, B_new, n_signatures=2, H=H)
(W_new, F_new), *_ = new_model.fit()
```

## Other models

sfNMF also needs the fragment length of each column. `get_fragment_lengths` reads them from column names such as `length_50`:

```python
from batch_nmf import get_fragment_lengths, sigmoidfilterNMF

model = sigmoidfilterNMF(X, B, n_signatures=2, lengths=get_fragment_lengths(columns))
(W, H, F), *_ = model.fit()
```

Standard NMF without batch effects is available as `plainNMF(X, n_signatures=2)`, fitted with Adam like bNMF and sfNMF, and as `fit_scipy_nmf(X, 2)`, fitted with scikit-learn.

## CPU and GPU

The models run on a GPU automatically if PyTorch finds one. On CPUs, PyTorch may start one thread per core. Set the environment variable `OMP_NUM_THREADS` to the number of cores it may use, for example in a cluster job.
