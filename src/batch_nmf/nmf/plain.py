from batch_nmf.nmf.base import baseNMF

class plainNMF(baseNMF):
    """Standard NMF (no batch effects) with softmax-normalised W and H, fitted with Adam.

    This is `--method adam` in the command line tool.
    """
    def __init__(self, target, n_signatures=2, H=None, volume_regularization=-1):
        super().__init__(target, n_signatures, H, volume_regularization=volume_regularization)
        self.name = "plain"
        self.rand_init()

    def forward(self):
        return self.get_W() @ self.get_H()

    def get_parameters(self):
        if self.predict:
            return self.get_W().cpu().detach().numpy()
        else:
            return (self.get_W().cpu().detach().numpy(), self.get_H().cpu().detach().numpy())

