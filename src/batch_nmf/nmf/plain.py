import numpy as np

from batch_nmf.nmf.base import baseNMF

class plainNMF(baseNMF):
    """Standard NMF without batch effects, fitted with the Adam optimizer.

    Models the normalised fragment length profiles `X` (one row per sample) as `X ≈ W H`, with
    the same parametrisation and fitting as bNMF and sfNMF. This is `--method adam` in the
    command line tool.

    Args:
        target: Input matrix with one row per sample, rows summing to 1.
        n_signatures: Number of signatures.
        H: Fixed signatures (signatures × lengths). If given, only the weights are fitted.
        volume_regularization: If positive, the strength of a penalty on the volume of the
            signatures, `log det(H Hᵀ + I)`.
    """
    def __init__(self, target: np.ndarray, n_signatures: int = 2, H: np.ndarray | None = None,
                 volume_regularization: float = -1):
        super().__init__(target, n_signatures, H, volume_regularization=volume_regularization)
        self.name = "plain"
        self.rand_init()

    def forward(self):
        return self.get_W() @ self.get_H()

    def get_parameters(self):
        """The fitted parameters as numpy arrays: `(W, H)`, or `W` when `H` was given."""
        if self.predict:
            return self.get_W().cpu().detach().numpy()
        else:
            return (self.get_W().cpu().detach().numpy(), self.get_H().cpu().detach().numpy())

