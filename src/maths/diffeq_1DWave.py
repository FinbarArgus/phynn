import torch.nn as nn
import copy


class Diffeq1DWave(nn.Module):
    """
    A wrapper for differential equations.

    """

    def __init__(self, func, train_wHmodel=False, order=1):
        super().__init__()
        self.func = func
        if train_wHmodel:
            self.funcH = copy.deepcopy(func)
        self.nfe = 0.  # number of function evaluations.
        self.order = order
        self._intloss = None
        self._sensitivity = None
        self._dxds = None

    def forward(self, s, x, q, p, H_cons_net=False, train_wgrads=False):
        self.nfe += 1

        if self.order > 1:
            None  # TODO
        else:
            if train_wgrads:
                x = self.func.forward_wgrads(x, q, p)
            else:
                if H_cons_net:
                    x = self.funcH(x, q, p)
                else:
                    x = self.func(x, q, p)

        self._dxds = x
        return x

    # TODO use this function
    def reset_func(self, func):
        # This function resets both NN funcs to the same NN,
        self.func = func
        self.funcH = func
