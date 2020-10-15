import torch.nn as nn


class DiffEq(nn.Module):
    """
    A wrapper for differential equations. Currently this class supports only DEs of order 1.

    TODO: Support for higher order DEs.
    """

    def __init__(self, model, order=1):
        super().__init__()
        self.model = model
        self.nfe = 0.  # number of function evaluations.
        self.order = order
        self._intloss = None
        self._sensitivity = None
        self._dxds = None

    def forward(self, s, x):
        self.nfe += 1

        if self.order > 1:
            None  # TODO
        else:
            x = self.model(x)
        self._dxds = x
        return x
