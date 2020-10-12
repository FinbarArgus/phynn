import torch
import math
from torch import nn
import torch.nn.functional as F


class Sine(nn.Module):
    """
    A wrapper for PyTorch sine function.
    """
    def __init__(self, w0=1.):
        super().__init__()
        self.w0 = w0

    @staticmethod
    def forward(x):
        return torch.sin(x)


class Siren(nn.Module):
    """
    An implementation of the Sine activation function known as Siren.
    """
    def __init__(self, dim_in, dim_out, w0=30., c=6., is_first=False):
        """

        :param dim_in: input dimension.
        :param dim_out: output dimension.
        :param w0: initial weight.
        :param c: parameter to distribute the weights uniformly so that after sine activation the input is
        arcsine-distributed.
        :param is_first: boolean to check if it's the first layer.
        """
        super().__init__()
        self.dim_in = dim_in
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out)
        self.initialization(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)
        self.activation = Sine(w0)

    def initialization(self, weight, bias, c, w0):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out
