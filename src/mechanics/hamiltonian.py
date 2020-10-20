import torch
import torch.nn as nn


class HNNMassSpring(nn.Module):
    def __init__(self, hamiltonian: nn.Module, dim=1):
        super().__init__()
        self.H = hamiltonian
        self.n = dim

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            grad_h = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0]

        return torch.cat([grad_h[:, self.n:], -grad_h[:, :self.n]], 1).to(x)


class HNNMassSpringSeparable(nn.Module):
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


class HNN1DWaveSeparable(nn.Module):
    def __init__(self, hamiltonian: nn.Module, dim=1):
        super().__init__()
        self.H = hamiltonian
        self.n = dim

    def forward(self, x, q, p, detach=False):
        """This function calculates y_hat = (q_dot_hat, p_dot_hat) = (dH_dp_dx, -dH_dq_dx)
        """
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            q = q.requires_grad_(True)
            p = p.requires_grad_(True)

            dH_dq = torch.autograd.grad(self.H(torch.cat([x, q, p], dim=1)).sum(), q, allow_unused=False,
                                        create_graph=True)[0]
            dH_dp = torch.autograd.grad(self.H(torch.cat([x, q, p], dim=1)).sum(), p, allow_unused=False,
                                        create_graph=True)[0]

            dH_dq_dx = torch.Tensor(len(x), len(x[0])).to(x)
            dH_dp_dx = torch.Tensor(len(x), len(x[0])).to(x)
            # loop through dH_dq and dH_dp entries and get the derivative of each one wrt to the corresponding x_idx
            # we cant differentiate the whole dH_dq or dH_dq because we only want the derivative of the n'th dH_dq
            # wrt the n'th x coordinate
            for x_idx in range(len(dH_dq[0])):
                dH_dq_dx[:, x_idx] = torch.autograd.grad(dH_dq[:, x_idx].sum(), x,
                                                         allow_unused=False, create_graph=True)[0][:, x_idx]
                dH_dp_dx[:, x_idx] = torch.autograd.grad(dH_dp[:, x_idx].sum(), x,
                                                         allow_unused=False, create_graph=True)[0][:, x_idx]

            if detach:
                dH_dq_dx = dH_dq_dx.detach()
                dH_dp_dx = dH_dp_dx.detach()

        return -dH_dp_dx, -dH_dq_dx

    def forward_wgrads(self, x, q, p, detach=False):
        """This function calculates y_hat = (q_dot_hat, p_dot_hat) = (-dH_dp_dx, -dH_dq_dx)
        and y_hat_dot = (dq_dx_dot_hat, dp_dx_dot_hat) = (-dH_dp_dx_dx, -dH_dq_dx_dx)
        """
        grads = self.forward(x, q, p, detach=False)
        dH_dp_dx = -grads[0]
        dH_dq_dx = -grads[1]

        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)

            dH_dq_dx_dx = torch.Tensor(len(x), len(x[0])).to(x)
            dH_dp_dx_dx = torch.Tensor(len(x), len(x[0])).to(x)

            for x_idx in range(len(dH_dq_dx[0])):
                dH_dq_dx_dx[:, x_idx] = torch.autograd.grad(dH_dq_dx[:, x_idx].sum(), x,
                                      allow_unused=False, create_graph=True)[0][:, x_idx]
                dH_dp_dx_dx[:, x_idx] = torch.autograd.grad(dH_dp_dx[:, x_idx].sum(), x,
                                      allow_unused=False, create_graph=True)[0][:, x_idx]

            if detach:
                dH_dq_dx = dH_dq_dx.detach()
                dH_dp_dx = dH_dp_dx.detach()
                dH_dq_dx_dx = dH_dq_dx.detach()
                dH_dp_dx_dx = dH_dp_dx.detach()


        return -dH_dp_dx, -dH_dq_dx, -dH_dp_dx_dx, -dH_dq_dx_dx
