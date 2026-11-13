---
title: API reference
hide:
- navigation
---

# API reference

The models and input helpers below can all be imported from `batch_nmf`, for example
`from batch_nmf import batchNMF`. See [Python usage](../python.md) for an example of how they
fit together.

## Models

::: batch_nmf.batchNMF
    options:
      inherited_members: [fit, get_W, get_H]
      members: [fit, get_W, get_H, get_F, get_parameters]

::: batch_nmf.sigmoidfilterNMF
    options:
      inherited_members: [fit, get_W, get_H]
      members: [fit, get_W, get_H, get_F, get_parameters]

::: batch_nmf.plainNMF
    options:
      inherited_members: [fit, get_W, get_H]
      members: [fit, get_W, get_H, get_parameters]

::: batch_nmf.fit_scipy_nmf

## Fitting defaults

The defaults of `fit`, which are also the defaults of the command line tool.

::: batch_nmf.utils
    options:
      show_root_heading: false
      show_root_toc_entry: false
      members: [DEFAULT_LOSS, DEFAULT_RESTARTS, DEFAULT_LR, DEFAULT_LR_FACTOR, DEFAULT_MIN_LR, DEFAULT_EPOCHS, DEFAULT_RESTART_EPOCHS, DEFAULT_TOL, DEFAULT_CHECK_EVERY]
      summary: false
      show_source: false
      backlinks: false

## Reading input

::: batch_nmf.read_matrix

::: batch_nmf.read_batch_file

::: batch_nmf.get_fragment_lengths
