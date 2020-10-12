import torch.nn as nn


class DiffEq_1DWave(nn.Module):
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

    def forward(self, s, x, q, p):
        self.nfe += 1

        # This was needed for bugfix with datatype
        for _, module in self.model.named_modules():
            if hasattr(module, 's'):
                module.s = s

        if self.order > 1:
            None  # TODO
        else:
            x = self.model(x, q, p)
        self._dxds = x
        return x
