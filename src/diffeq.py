import torch
import torch.nn as nn


class DiffEq(nn.Module):
    """
    A wrapper for differential equations. Currently this class supports only DEs of order 1.

    TODO: Support for higher order DEs.
    """

    def __init__(self, model, order=1):
        super().__init__()
        self.model = model
        self.nfe = 0.
        self.order = order
        self._intloss = None
        self._sensitivity = None
        self._dxds = None

    def forward(self, s, x):
        self.nfe += 1
        for _, module in self.model.named_modules():
            if hasattr(module, 's'):
                module.s = s

        # if-else to handle autograd training with integral loss propagated in x[:, 0]
        if (self._intloss is not None) and (self._sensitivity == 'autograd'):
            x_dyn = x[:, 1:]
            dlds = self._intloss(x_dyn)
            if len(dlds.shape) == 1:
                dlds = dlds[:, None]
            if self.order > 1:
                None  # TODO
            else:
                x_dyn = self.model(x_dyn)
            self._dxds = x_dyn
            return torch.cat([dlds, x_dyn], 1).to(x_dyn)

        # regular forward
        else:
            if self.order > 1:
                None  # TODO
            else:
                x = self.model(x)
            self._dxds = x
            return x
