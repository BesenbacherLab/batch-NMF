import numpy as np
from sklearn.decomposition import NMF

def fit_scipy_nmf(input_matrix: np.ndarray, n_components: int, n_restarts: int = 10, beta_loss: str = 'frobenius',
                  random_state: int | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    """Standard NMF fitted with scikit-learn: the best of `n_restarts` random starts.

    Uses `sklearn.decomposition.NMF` with multiplicative updates. This is `--method scipy` in
    the command line tool.

    Args:
        input_matrix: Input matrix with one row per sample.
        n_components: Number of signatures.
        n_restarts: Number of random starts; the fit with the lowest reconstruction error is kept.
        beta_loss: `'frobenius'` or `'kullback-leibler'`.
        random_state: Seed for the random starts (start i uses `random_state + i`), or None.

    Returns:
        A tuple `(error, W, H)`: the reconstruction error and the weights and signatures. As
            in scikit-learn, the scale of each signature is not fixed: the rows of `H` need not
            sum to 1.
    """
    minModel = None
    minErr = 1e100
    for i in range(n_restarts):
        seed = None if random_state is None else random_state + i
        model = NMF(n_components=n_components, max_iter=1000000, tol=1e-9, init="random", beta_loss= beta_loss, solver='mu',
                    random_state=seed)
        tmp_W = model.fit_transform(input_matrix)
        if model.reconstruction_err_ < minErr:
            minModel = model
            minErr = model.reconstruction_err_
            W_est = tmp_W
            H_est = minModel.components_
    return minErr, W_est, H_est

