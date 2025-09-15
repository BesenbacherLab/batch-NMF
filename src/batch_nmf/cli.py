"""Module that contains the command line application."""

import argparse
from batch_nmf import __version__
from batch_nmf.io_utils import print_tsv, read_matrix, read_batch_file
from batch_nmf.nmf.plain import plainNMF
from batch_nmf.nmf.batch import batchNMF
from batch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF
from batch_nmf.nmf.scipy_nmf import fit_scipy_nmf
from sklearn.decomposition import non_negative_factorization
import numpy as np

def get_parser():
    """
    Return the CLI argument parser.

    Returns:
        An argparse parser.
    """
    parser = argparse.ArgumentParser(
        prog="batch-nmf",
        description='''
        pytorch implementation of NMF extentions that incorporate batch effects including Batch NMF(bNMF) and Sigmoid Filter NMF (sfNMF).
        ''')

    parser.add_argument('command', type=str, choices= ['fit', 'predict'])

    parser.add_argument(
        '-V', '--version', action='version', version=f'%(prog)s {__version__}')
 
    parser.add_argument('--input_matrix', type=str, required=True,
                        help='Input matrix file that should be factorized. '
                        'Format can be either tsv or csv depending on the file suffix. ' \
                        'Rows are samples, columns are fragment lengths. '
                        'The first row is a header with column names '
                        'and the first column a header with sample names.'
                        )
        
    parser.add_argument('--n_signatures', type=int, default = 2,
                        help='Number of signatures to be estimated. Default is 2.')
    parser.add_argument('--signature_file', type = str, required=True,
                        help='TSV file with signatures. Output file for fit command, input file for predict command. ')
    parser.add_argument('--exposure_file', type = str, required=True,
                        help='Output TSV file with sample exposures.')
    parser.add_argument('--batch_file', type = str, required=False,
                        help='TSV file with sample to batch mapping. '
                        'The first column is sample id, the second column is batch id.' \
                        'Required for batch NMF and sigmoid filter NMF. ')
    parser.add_argument('--filter_file', type = str, required=False,
                        help='Output TSV file with batch filter. '
                        'This file is only created when batch NMF or sigmoid filter NMF is used. ')
    parser.add_argument('--pseudo_count', type=float, default = 1.0,
                        help='Pseudo count to be added to the input matrix before normalizing it so rows sum to 1.' \
                        'Default is 1.0. If the matrix is already normalized, set this to 0.0.')
    
    parser.add_argument('--method', type=str, default = 'bNMF',
                        help='which NMF version should be used',
                        choices=['bNMF', 'sfNMF', 'scipy'])
    parser.add_argument('--n_epochs', type=int, default = 50000)
    parser.add_argument('--learning_rate', type=float, default = 0.5)
    parser.add_argument('--n_restarts', type=int, default = 4)  
    parser.add_argument('--filter_regularization', type=float, default=-1)
    parser.add_argument('--scaler_regularization', type=float, default=-1)
    parser.add_argument('--volume_regularization', type=float, default=-1)
    parser.add_argument('--all_separate_batch', action='store_true',
                        help = 'Treat all samples as seperate batches.')
    parser.add_argument('--logfile', type=argparse.FileType('w'))
    parser.add_argument('--loss', type=str, default='kullback-leibler',
                        choices = ['frobenius', 'kullback-leibler'],
                        help='Loss function to be used. Default is kullback-leibler divergence.')
    parser.add_argument('--weighted_loss', action='store_true',
                        help = 'Each batch contributete equally to loss')
    parser.add_argument('--vrgrid', action='store_true',
                            help = '(EXPERIMENTAL) Use grid search to set best volume regularization parameter')

    return parser



def main(args = None):
    """
    Run the main program.

    This function is executed when you type `pytorch-nmf` or `python -m pytorch_nmf`.

    Arguments:
        args: Arguments passed from the command line.

    Returns:
        An exit code.
    """
    parser = get_parser()
    #print(args)
    #if args is None or args == []:
    #    print("Use --help to see available options.")
    #    parser.print_help()
    #    return 0

    opts = parser.parse_args(args=args)
    print(opts)
    if opts.command is None:
        parser.print_help()
        return 0

    input_matrix, sample_names, col_names = read_matrix(opts.input_matrix)

    if input_matrix.sum(1, keepdims=True).max() < 1.5:
        assert opts.pseudo_count < 1e-4, "Input matrix seems to be already normalized. Set --pseudo_count to 0 to avoid adding pseudo count."
    assert opts.pseudo_count >= 0
    input_matrix += opts.pseudo_count
    
    assert input_matrix.min() >= 0, "Input matrix contains negative values. Please check your input data."
    assert np.isfinite(input_matrix).all(), "Input matrix contains non-finite values. Please check your input data."
    
    # Normalize input matrix so that each row sums to 1.
    input_matrix = input_matrix/input_matrix.sum(1, keepdims=True)  

    if opts.command == 'fit':
        sig_names = [f'Sig{i+1}' for i in range(opts.n_signatures)]

        if opts.batch_file is None and not opts.all_separate_batch:
            assert opts.method == 'scipy', "batch_file is required unless --method=scipy or --all_separate_batch option is set"
            print("Fitting plain NMF using scipy...")
            loss, W_est, H_est = fit_scipy_nmf(input_matrix, opts.n_signatures, opts.n_restarts, beta_loss=opts.loss)
        else:
            assert opts.method != 'scipy', "--batch_file or --all_separate_batch option should not be set when method is scipy"
            if opts.all_separate_batch:
                assert opts.batch_file is None, "batch_file should not be supplied when --all_separate_batch option is set"
                batch_matrix = np.identity(len(sample_names))
                batch_names = sample_names
            else:
                batch_matrix, batch_names = read_batch_file(opts.batch_file, sample_names)

            if opts.method == 'bNMF':
                print("Fitting batch NMF...")

                m = batchNMF(
                    input_matrix, 
                    batch_matrix, 
                    opts.n_signatures,
                    filter_regularization = opts.filter_regularization, 
                    scaler_regularization = opts.scaler_regularization,
                    volume_regularization = opts.volume_regularization,
                    )   
                res, loss, loss_np, penalty = m.fit(
                    opts.n_epochs, 
                    logfile = opts.logfile, 
                    lr = opts.learning_rate,
                    restarts = opts.n_restarts,
                    vrgrid = opts.vrgrid,
                    loss = opts.loss,
                    weighted_loss = opts.weighted_loss,
                    )
                W_est, H_est, F_est = res
                print_tsv(F_est, opts.filter_file, batch_names, col_names, "batch_id")
                    
            elif opts.method =='sfNMF':
                print("Fitting sigmoid filter NMF...")

                m = sigmoidfilterNMF(
                    input_matrix, 
                    batch_matrix, 
                    opts.n_signatures,
                    filter_regularization = opts.filter_regularization, 
                    scaler_regularization = opts.scaler_regularization,
                    volume_regularization = opts.volume_regularization,
                    )   
                res, loss, loss_np, penalty = m.fit(
                    opts.n_epochs, 
                    logfile = opts.logfile, 
                    lr = opts.learning_rate,
                    restarts = opts.n_restarts,
                    vrgrid = opts.vrgrid,
                    loss = opts.loss,
                    weighted_loss = opts.weighted_loss,
                    )
                W_est, H_est, F_est = res
                print_tsv(F_est, opts.filter_file, batch_names, col_names, "batch_id")


        W_est = W_est/W_est.sum(1, keepdims=True)
        print_tsv(W_est, opts.exposure_file, sample_names, sig_names, "sample_id")
        print_tsv(H_est, opts.signature_file, sig_names, col_names, "signature")

    elif opts.command == 'predict':
        signature_matrix, sig_names, sig_col_names = read_matrix(opts.signature_file)
        
        if sig_names is None:
            sig_names = [f'Sig{i+1}' for i in range(signature_matrix.shape[0])]
        
        n_signatures = len(sig_names)

        if opts.batch_file is None and not opts.all_separate_batch:
            assert opts.method == 'scipy', "batch_file is required unless --method=scipy or --all_separate_batch option is set"
            W_est, H_est, n_iter = non_negative_factorization(input_matrix, H=signature_matrix, update_H=False, n_components=n_signatures, tol=1e-7, max_iter=1000000, init="random", beta_loss=opts.loss, solver='mu')
            loss = -1
        else:
            assert opts.method != 'scipy', "--batch_file or --all_separate_batch option should not be set when method is scipy"
            if opts.all_separate_batch:
                assert opts.batch_file is None, "batch_file should not be supplied when --all_separate_batch option is set"

                batch_matrix = np.identity(len(sample_names))
                batch_names = sample_names
            else:
                batch_matrix, batch_names = read_batch_file(opts.batch_file, sample_names)
                
            if opts.method == 'bNMF':
                m = batchNMF(
                    input_matrix, 
                    batch_matrix, 
                    n_signatures, 
                    signature_matrix,
                    filter_regularization = opts.filter_regularization, 
                    scaler_regularization = opts.scaler_regularization,
                    volume_regularization = opts.volume_regularization,
                )
                
                res, loss, loss_np, penalty = m.fit(
                    opts.n_epochs, 
                    logfile = opts.logfile, 
                    lr = opts.learning_rate,
                    restarts = opts.n_restarts,
                    loss = opts.loss,
                    weighted_loss = opts.weighted_loss,
                )
                    
                W_est, F_est = res
                print_tsv(F_est, opts.filter_file, batch_names, col_names, "batch_id")

            elif opts.method == 'sfNMF':
                m = sigmoidfilterNMF(
                    input_matrix, 
                    batch_matrix, 
                    n_signatures,
                    signature_matrix,
                    filter_regularization = opts.filter_regularization, 
                    scaler_regularization = opts.scaler_regularization,
                    volume_regularization = opts.volume_regularization,
                    )   
                res, loss, loss_np, penalty = m.fit(
                    opts.n_epochs, 
                    logfile = opts.logfile, 
                    lr = opts.learning_rate,
                    restarts = opts.n_restarts,
                    vrgrid = opts.vrgrid,
                    loss = opts.loss,
                    weighted_loss = opts.weighted_loss,
                    )
                W_est, F_est = res
                print_tsv(F_est, opts.filter_file, batch_names, col_names, "batch_id")

        W_est = W_est/W_est.sum(1, keepdims=True)
        print_tsv(W_est, opts.exposure_file, sample_names, sig_names, "sample_id")

    print(f'loss={loss}')
    return 0
