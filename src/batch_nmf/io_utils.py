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
    f = open(fname)
    sample2batchnum = {}
    batch_ids = []
    for line in f:
        sample_id, batch_id = line.split()
        if sample_id not in sample_ids:
            continue
        if batch_id not in batch2num:
            batch2num[batch_id] = n_batches
            n_batches += 1
            batch_ids.append(batch_id)
        sample2batchnum[sample_id] = batch2num[batch_id]
    assert len(sample2batchnum) == len(sample_ids)
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
