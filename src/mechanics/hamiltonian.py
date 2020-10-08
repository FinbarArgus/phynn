import torch
import torch.nn as nn
import numpy as np


class HNNMassSpring(nn.Module):
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
    def __init__(self, hamiltonian: nn.Module, num_points, dim=1):
        super().__init__()
        self.H = hamiltonian
        self.n = dim
        self.num_points = num_points

    def forward(self, x, q, p, detach=False):
        #This function calculates y_hat = (q_dot_hat, p_dot_hat) = (dH_dp_dx, -dH_dq_dx)
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            q = q.requires_grad_(True)
            p = p.requires_grad_(True)

            dH_dq = torch.autograd.grad(self.H(torch.cat([x, q, p])).sum(), q, allow_unused=False,
                                        create_graph=True)[0].unsqueeze(1)
            dH_dp = torch.autograd.grad(self.H(torch.cat([x, q, p])).sum(), p, allow_unused=False,
                                        create_graph=True)[0].unsqueeze(1)

            dH_dq_dx_list = []
            dH_dp_dx_list = []

            # loop through dH_dq and dH_dp entries and get the derivative of each one wrt to the corresponding x_idx
            for x_idx, (dH_dq_entry, dH_dp_entry) in enumerate(zip(dH_dq, dH_dp)):
                dH_dq_dx_entry = torch.autograd.grad(dH_dq_entry, x, allow_unused=False, create_graph=True)[0][x_idx]
                dH_dp_dx_entry = torch.autograd.grad(dH_dp_entry, x, allow_unused=False, create_graph=True)[0][x_idx]
                dH_dq_dx_list.append(dH_dq_dx_entry)
                dH_dp_dx_list.append(dH_dp_dx_entry)

            if detach:
                dH_dq_dx = torch.stack(dH_dq_dx_list).detach()
                dH_dp_dx = torch.stack(dH_dp_dx_list).detach()
            else:
                dH_dq_dx = torch.stack(dH_dq_dx_list)
                dH_dp_dx = torch.stack(dH_dp_dx_list)

        return dH_dp_dx, -dH_dq_dx

    def forward_wgrads(self, x, q, p, dq_dx, dp_dx):
        #This function calculates y_hat = (q_dot_hat, p_dot_hat) = (dH_dp_dx, -dH_dq_dx)
        grads = self.forward(x, q, p, detach=False)
        dH_dp_dx = grads[0]
        dH_dq_dx = -grads[1]

        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            q = q.requires_grad_(True)
            p = p.requires_grad_(True)
            dq_dx = dq_dx.requires_grad_(True)
            dp_dx = dp_dx.requires_grad_(True)

            dH_dq_dx_dx_list = []
            dH_dp_dx_dx_list = []

            for x_idx, (dH_dq_dx_entry, dH_dp_dx_entry) in enumerate(zip(dH_dq_dx, dH_dp_dx)):
                dH_dq_dx_entry = torch.autograd.grad(dH_dq_dx_entry, x, allow_unused=False, create_graph=True)[0][x_idx]
                dH_dp_dx_entry = torch.autograd.grad(dH_dp_dx_entry, x, allow_unused=False, create_graph=True)[0][x_idx]
                dH_dq_dx_dx_list.append(dH_dq_dx_entry)
                dH_dp_dx_dx_list.append(dH_dp_dx_entry)

            dH_dq_dx_dx = torch.stack(dH_dq_dx_dx_list)
            dH_dp_dx_dx = torch.stack(dH_dp_dx_dx_list)

        return dH_dp_dx.detach(), -dH_dq_dx.detach(), dH_dp_dx_dx.detach(), -dH_dq_dx_dx.detach()








