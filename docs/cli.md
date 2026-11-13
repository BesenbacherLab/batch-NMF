---
title: Command line
---

# Command line

`batch-nmf` has two commands:

- `fit` learns a set of signatures and their weights in each sample from an input matrix. bNMF and sfNMF also learn a filter for each batch.
- `predict` infers the weights of known signatures in new samples. bNMF and sfNMF also learn a filter for each batch of the new samples.

For example, to fit bNMF with two signatures to the example data in the repository, and then infer the weights of those signatures in new samples:

```bash
batch-nmf fit --input_matrix test_data/length_data.tsv --batch_file test_data/batch_file.tsv \
    --n_signatures 2 --signature_file signatures.tsv --exposure_file exposures.tsv --filter_file filters.tsv
batch-nmf predict --input_matrix new_samples.tsv --batch_file new_batches.tsv \
    --signature_file signatures.tsv --exposure_file new_exposures.tsv --filter_file new_filters.tsv
```

`--method` chooses the model: `bNMF` (the default), `sfNMF`, or standard NMF without batch effects, `adam` or `scipy`.

## Input files

- **Input matrix** (`--input_matrix`, `.tsv` or `.csv`): one row per sample and one column per fragment length. The first row holds the column names and the first column the sample names. For sfNMF, the column names should end with the fragment length (for example `length_50`), or give `--min_length`. The values are usually counts: a pseudo-count (`--pseudo_count`, default 1) is added and each row is scaled to sum to 1. If the rows are already normalised, use `--pseudo_count 0`.
- **Batch file** (`--batch_file`, bNMF and sfNMF): two columns without a header, the sample name and its batch. For `fit` with bNMF, the first batch in the file is the reference batch.

## Output files

All are tab-separated, with names in the first row and column.

- **Signatures** (`--signature_file`): one row per signature and one column per fragment length. This file is the input to `predict`.
- **Weights** (`--exposure_file`): one row per sample and one column per signature. Each row sums to 1.
- **Filters** (`--filter_file`, bNMF and sfNMF): one row per batch and one column per fragment length.
- **Training log** (`--logfile`, optional): the loss and learning rate every 5000 epochs and at the end of each run.

## Options

```python exec="yes" result="text"
import os

os.environ["COLUMNS"] = "80"  # line width of the help text

from batch_nmf.cli import get_parser

print(get_parser().format_help())
```
