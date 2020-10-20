import torch.nn as nn


class Diffeq1DWave(nn.Module):
    """
    A wrapper for differential equations. Currently this class supports only DEs of order 1.

    TODO: Support for higher order DEs.
    """

    def __init__(self, func, order=1):
        super().__init__()
        # TODO(Finbar) the name model is overused maybe change this to func, like in the
        # TODO parent process
        self.func = func
        self.nfe = 0.  # number of function evaluations.
        self.order = order
        self._intloss = None
        self._sensitivity = None
        self._dxds = None

    def forward(self, s, x, q, p, train_wgrads=False):
        self.nfe += 1

        if self.order > 1:
            None  # TODO
        else:
            if train_wgrads:
                x = self.func.forward_wgrads(x, q, p)
            else:
                x = self.func(x, q, p)
        self._dxds = x
        return x
