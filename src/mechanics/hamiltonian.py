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

    def forward(self, x, aa, bb, cc, dd):
        #This function calculates y_hat = (q_dot_hat, p_dot_hat) = (dH_dp_dx, -dH_dq_dx)
        # x is the vector of all inputs
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            aa = aa.requires_grad_(True)
            xCoords_start = 0
            q_start = self.num_points
            p_start = 2*self.num_points

            # xCoords = torch.Tensor.narrow(x, 0, xCoords_start, self.num_points)
            # q = torch.Tensor.narrow(x, 0, q_start, self.num_points)
            # p = torch.Tensor.narrow(x, 0, p_start, self.num_points)

            dH_dq = torch.autograd.grad(self.H(torch.cat([x, aa, bb, cc, dd])).sum(), x, allow_unused=False, create_graph=True)[0].unsqueeze(1)
            dH_dqa = torch.autograd.grad(self.H(torch.cat([x, aa, bb, cc, dd])).sum(), aa, allow_unused=False, create_graph=True)[0].unsqueeze(1)

            #dH_dq = torch.Tensor.narrow(dH_dinp, 0, q_start, self.num_points)
            #dH_dp = torch.Tensor.narrow(dH_dinp, 0, p_start, self.num_points)

            #dH_dq_dinp = torch.autograd.grad(dH_dq, x, allow_unused=False, create_graph=True)[0].unsqueeze(1)
            #dH_dp_dinp = torch.autograd.grad(dH_dp, x, allow_unused=False, create_graph=True)[0].unsqueeze(1)

            #dH_dq_dx = dH_dq_dinp[xCoords_idx]
            #dH_dp_dx = dH_dp_dinp[xCoords_idx]

            # dH_dq_dx_list = []
            # dH_dp_dx_list = []
            #
            # # This doesn't work
            # for dH_dq_entry, dH_dp_entry, x_entry in zip(dH_dq, dH_dp, x[xCoords_idx]):
            #     dH_dq_dx_entry = torch.autograd.grad(dH_dq_entry, x_entry, allow_unused=False, create_graph=True)[0]
            #     dH_dp_dx_entry = torch.autograd.grad(dH_dp_entry, x_entry, allow_unused=False, create_graph=True)[0]
            #     dH_dq_dx_list.append(dH_dq_dx_entry)
            #     dH_dp_dx_list.append(dH_dp_dx_entry)
            #
            # # This works but is not efficient
            # for x_idx, (dH_dq_entry, dH_dp_entry) in enumerate(zip(dH_dq, dH_dp)):
            #     dH_dq_dx_entry = torch.autograd.grad(dH_dq_entry, x, allow_unused=False, create_graph=True)[0][x_idx]
            #     dH_dp_dx_entry = torch.autograd.grad(dH_dp_entry, x, allow_unused=False, create_graph=True)[0][x_idx]
            #     dH_dq_dx_list.append(dH_dq_dx_entry)
            #     dH_dp_dx_list.append(dH_dp_dx_entry)
            #
            # dH_dq_dx = torch.stack(dH_dq_dx_list)
            # dH_dp_dx = torch.stack(dH_dp_dx_list)

        # return torch.cat([dH_dp_dx, -dH_dq_dx], 0).to(x)
        return torch.cat([dH_dq], 0).to(x)

    # TODO (Finbar) Forward_q and forward_p functions
