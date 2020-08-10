import torch
import torch.nn as nn
import torchdiffeq
import pytorch_lightning as pl
from .diffeq import DiffEq
from .adjoint.adjoint import Adjoint


class DENNet(pl.LightningModule):
    """
    A class to handle lower-level parameters of a differential equation neural network.
    This class inherits from PyTorch-Lightning
    """

    def __init__(self, func: nn.Module, order=1, sensitivity='autograd', s_span=torch.linspace(0, 1, 2), solver='rk4',
                 atol=1e-4, rtol=1e-4, intloss=None):

        super().__init__()

        self.de_function = DiffEq(func, order)
        self.order = order
        self.sensitivity = sensitivity
        self.s_span = s_span
        self.solver = solver
        self.nfe = self.de_function.nfe
        self.rtol = rtol
        self.atol = atol
        self.intloss = intloss
        self.u = None
        self.controlled = False  # data-control

        if sensitivity == 'adjoint': self.adjoint = Adjoint(self.intloss);

    def _prep_odeint(self, x: torch.Tensor):
        self.s_span = self.s_span.to(x)

        excess_dims = 0
        if self.intloss is not None and self.sensitivity == 'autograd':
            excess_dims += 1

        for name, module in self.de_function.named_modules():
            if hasattr(module, 'trace_estimator'):
                if module.noise_dist is not None:
                    module.noise = module.noise_dist.sample((x.shape[0],))
                excess_dims += 1

        # TODO: merge the named_modules loop for perf
        for name, module in self.de_function.named_modules():
            if hasattr(module, 'u'):
                self.controlled = True
                module.u = x[:, excess_dims:].detach()

        return x

    def forward(self, x: torch.Tensor):
        x = self._prep_odeint(x)
        switcher = {
            'autograd': self._autograd,
            'adjoint': self._adjoint,
        }
        odeint = switcher.get(self.sensitivity)
        out = odeint(x)
        return out

    def trajectory(self, x: torch.Tensor, s_span: torch.Tensor):
        x = self._prep_odeint(x)
        sol = torchdiffeq.odeint(self.de_function, x, s_span,
                                 rtol=self.rtol, atol=self.atol, method=self.solver)
        return sol

    def backward_trajectory(self, x: torch.Tensor, s_span: torch.Tensor):
        """ # TODO """
        raise NotImplementedError

    def reset(self):
        self.nfe, self.de_function.nfe = 0, 0

    def _autograd(self, x):
        self.de_function.intloss, self.de_function.sensitivity = self.intloss, self.sensitivity

        if self.intloss is None:
            return torchdiffeq.odeint(self.de_function, x, self.s_span, rtol=self.rtol,
                                      atol=self.atol, method=self.solver)[-1]
        else:
            return torchdiffeq.odeint(self.de_function, x, self.s_span,
                                      rtol=self.rtol, atol=self.atol, method=self.solver)[-1]

    def _adjoint(self, x):
        return self.adjoint(self.de_function, x, self.s_span, rtol=self.rtol, atol=self.atol, method=self.solver)

    @property
    def nfe(self):
        return self.de_function.nfe

    @nfe.setter
    def nfe(self, val):
        self.de_function.nfe = val
