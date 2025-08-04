"""Module that contains the command line application."""

import argparse
from pytorch_nmf import __version__
from pytorch_nmf.io_utils import print_tsv, read_matrix, read_batch_file
from pytorch_nmf.nmf.plain import plainNMF
from pytorch_nmf.nmf.batch import batchNMF
from pytorch_nmf.nmf.sigmoid_filter import sigmoidfilterNMF
from pytorch_nmf.nmf.scipy_nmf import fit_scipy_nmf
from sklearn.decomposition import non_negative_factorization

#from pytorch_nmf.nmf import filter, shared_filter, plain, shared_monotonic_filter
import numpy as np

def get_parser():
    """
    Return the CLI argument parser.

    Returns:
        An argparse parser.
    """
    parser = argparse.ArgumentParser(
        prog="pytorch-nmf",
        description='''
        Train NMF (and extensions of NMF) using pytorch
        ''')

    parser.add_argument('command', type=str, choices= ['fit', 'predict'], required=True)

    parser.add_argument(
        '-V', '--version', action='version', version=f'%(prog)s {__version__}')
 

    parser.add_argument('--input_matrix', type=str, required=True)
        
    parser.add_argument('--n_signatures', type=int, default = 2)
    parser.add_argument('--signature_file', type = str, required=True)
    parser.add_argument('--exposure_file', type = str, required=True)
    parser.add_argument('--batch_file', type = str, required=False)
    parser.add_argument('--filter_file', type = str, required=False)
    parser.add_argument('--filter_weight_file', type = str, required=False)
    parser.add_argument('--batch_filter_file', type = str, required=False)
    parser.add_argument('--raw_filter_file', type = str, required=False)
    parser.add_argument('--pseudo_count', type=float, default = 1.0)


    parser.add_argument('--method', type=str, default = 'bNMF',
        help='which NMF version should be used',
        choices=['bNMF', 'sfNMF', 'scipy'])
    parser.add_argument('--n_epochs', type=int, default = 50000)
    parser.add_argument('--learning_rate', type=float, default = 0.5)
    parser.add_argument('--n_restarts', type=int, default = 4)
    parser.add_argument('--filter_regularization', type=float, default=-1)
    parser.add_argument('--scaler_regularization', type=float, default=-1)
    parser.add_argument('--volume_regularization', type=float, default=-1)
    parser.add_argument('--no_normalization', action='store_true',
                        help = 'Do not make input rows sum to one.')
    parser.add_argument('--all_separate_batch', action='store_true',
                        help = 'Treat all samples as seperate batch')
    parser.add_argument('--logfile', type=argparse.FileType('w'))
    parser.add_argument('--loss', type=str, default='kullback-leibler',
        choices = ['frobenius', 'kullback-leibler'])
    parser.add_argument('--weighted_loss', action='store_true',
                        help = 'Each batch contributete equally to loss')
    parser.add_argument('--vrgrid', action='store_true',
                            help = 'use grid search to set best volume regularization parameter')

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
    opts = parser.parse_args(args=args)
    #print(opts) 
    
    if opts.command is None:
        parser.print_help()
        return 0

    input_matrix, sample_names, col_names = read_matrix(opts.input_matrix)

    assert opts.pseudo_count >= 0
    input_matrix += opts.pseudo_count

    if not opts.no_normalization:
        input_matrix = input_matrix/input_matrix.sum(1, keepdims=True)
    
    if opts.command == 'fit':
        sig_names = [f'Sig{i+1}' for i in range(opts.n_signatures)]

        if opts.batch_file is None and not opts.all_separate_batch:
            assert opts.method == 'scipy', "batch_file is required unless --method=scipy or --all_separate_batch option is set"
            print("fitting plain NMF using scipy")
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
                print("fitting batch NMF")

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
                if not opts.raw_filter_file is None:
                    rawF = m.F.cpu().detach().numpy()
                    if rawF.shape[0] == len(batch_names):
                        print_tsv(rawF, opts.raw_filter_file, batch_names, col_names, "batch_id")
                    else:
                        print_tsv(rawF, opts.raw_filter_file, batch_names[1:], col_names, "batch_id")
                    
            elif opts.method =='sfNMF':
                print("fitting sigmoid filter NMF")

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
                
            if opts.input_filter_file is None:

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
