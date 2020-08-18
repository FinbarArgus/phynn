import torch
import torch.nn as nn


class HNN(nn.Module):
    def __init__(self, hamiltonian: nn.Module, dim=1):
        super().__init__()
        self.H = hamiltonian
        self.n = dim

    def forward(self, x):
        # TODO (Finbar) do I need to include a kwarg for grads=True so i can set it to false when evaluating time step?
        # TODO (Mahyar) I don't think that's needed as we're using `detach()` when evaluating. Let me know if disagree.

        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            grad_h = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0]

        return torch.cat([grad_h[:, self.n:], -grad_h[:, :self.n]], 1).to(x)


class HNNSeparable(nn.Module):
    def __init__(self, hamiltonian: nn.Module, dim=1):
        super().__init__()
        self.H = hamiltonian
        self.n = dim

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)

            grad_hq = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0][:, 0].unsqueeze(
                1)
            grad_hp = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0][:, 1].unsqueeze(
                1)
        return torch.cat([grad_hp, -grad_hq], 1).to(x)

    # TODO (Finbar) Forward_q and forward_p functions
