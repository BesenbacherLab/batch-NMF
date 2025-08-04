from sklearn.decomposition import NMF

def fit_scipy_nmf(input_matrix, n_components, n_restarts=10, beta_loss = 'frobenius'):
    minModel = None
    minErr = 1e100
    for i in range(n_restarts):
        model = NMF(n_components=n_components, max_iter=1000000, tol=1e-9, init="random", beta_loss= beta_loss, solver='mu')
        tmp_W = model.fit_transform(input_matrix)
        if model.reconstruction_err_ < minErr:
            minModel = model
            minErr = model.reconstruction_err_
            W_est = tmp_W
            H_est = minModel.components_
    return minErr, W_est, H_est

