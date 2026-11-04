import re

import pandas as pd
import numpy as np

def print_tsv(A, filename, row_names, col_names, index_name):
    if not filename is None:
        df = pd.DataFrame(A, columns = col_names, index=row_names)
        if not row_names is None:
            df.index.name = index_name
        df.to_csv(filename, sep = "\t")

def read_batch_file(fname, sample_ids):
    batch2num = {}
    n_batches = 0
    sample2batchnum = {}
    batch_ids = []
    with open(fname, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # skip blank lines (e.g. trailing newline at end of file)
            sample_id, batch_id = line.split()
            if sample_id not in sample_ids:
                continue
            if batch_id not in batch2num:
                batch2num[batch_id] = n_batches
                n_batches += 1
                batch_ids.append(batch_id)
            sample2batchnum[sample_id] = batch2num[batch_id]
    missing = [s for s in sample_ids if s not in sample2batchnum]
    assert not missing, f"Samples missing from batch file {fname}: {missing[:10]}"
    A = np.zeros((len(sample_ids), n_batches))
    for i, sample_id in enumerate(sample_ids):
        batchnum = sample2batchnum[sample_id]
        A[i,batchnum] = 1
    return A, batch_ids


def read_matrix(fname):
    row_names = None
    col_names = None
    if fname.endswith('.tsv'):
        d = pd.read_csv(fname, sep='\t', index_col=0)
        matrix = d.to_numpy(dtype=np.double)
        row_names = d.index
        col_names = d.columns
    elif fname.endswith('.csv'):
        d = pd.read_csv(fname, index_col=0)
        matrix = d.to_numpy(dtype=np.double)
        row_names = d.index
        col_names = d.columns
    else:
        raise Exception('Unknown input_matrix ending. The file name should end on ".tsv" or ".csv".')
    return matrix, row_names, col_names


def get_fragment_lengths(col_names, min_length=None):
    """Return the fragment length of each column of the input matrix.

    Arguments:
        col_names: Column names of the input matrix.
        min_length: If given, the columns are taken to be the consecutive lengths
            min_length, min_length+1, ... Otherwise each length is read from the
            integer at the end of the column name (e.g. "length_100" -> 100).

    Returns:
        A numpy array with one length per column.
    """
    n_cols = len(col_names)
    if min_length is not None:
        return np.arange(min_length, min_length + n_cols)
    lengths = []
    for name in col_names:
        match = re.search(r"(\d+)$", str(name).strip())
        if match is None:
            raise ValueError(
                f"Cannot read a fragment length from column name {name!r}. "
                "Use column names that end with the length (e.g. 'length_100') or set --min_length."
            )
        lengths.append(int(match.group(1)))
    if len(set(lengths)) != n_cols:
        raise ValueError(
            "The fragment lengths read from the column names are not unique. "
            "Check the column names or set --min_length."
        )
    return np.array(lengths)


def align_columns(matrix, matrix_cols, target_cols, fname):
    """Reorder the columns of `matrix` so they match `target_cols`.

    Used in `predict` to make sure the signatures are matched to the input matrix
    column by column (i.e. fragment length by fragment length).

    Raises:
        ValueError: If the two sets of column names are not the same.
    """
    matrix_cols = [str(c) for c in matrix_cols]
    target_cols = [str(c) for c in target_cols]
    if matrix_cols == target_cols:
        return matrix
    missing = [c for c in target_cols if c not in set(matrix_cols)]
    extra = [c for c in matrix_cols if c not in set(target_cols)]
    if missing or extra or len(matrix_cols) != len(target_cols):
        raise ValueError(
            f"The columns of {fname} ({len(matrix_cols)} columns) do not match the columns of the input matrix "
            f"({len(target_cols)} columns). Missing from {fname}: {missing[:5]}{' ...' if len(missing) > 5 else ''}. "
            f"Not in the input matrix: {extra[:5]}{' ...' if len(extra) > 5 else ''}."
        )
    print(f"Reordering the columns of {fname} to match the input matrix.")
    index = {c: i for i, c in enumerate(matrix_cols)}
    return matrix[:, [index[c] for c in target_cols]]
