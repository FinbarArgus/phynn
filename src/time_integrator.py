import torch
import torch.nn as nn


class TimeIntegrator(nn.Module):
    """
    A class for integrating in time with a HNN.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def euler_step(self, x0, dt):
        """
        Explicit forward Euler time integration method.

        :param x0:
        :param dt:
        :return:
        """
        grads = self.model.forward(x0)
        x1 = x0 + dt * grads
        return x1

    def sv_step(self, x0, dt):
        """
        Stormer-Verlet (SV) (aka leapfrog) time integration scheme.

        :param x0:
        :param dt:
        :return:
        """
        grads0 = self.model.forward(x0)
        p_temp = x0[:, 1] + dt / 2 * grads0[:, 1]
        x_temp = torch.cat([x0[:, 0].unsqueeze(1), p_temp.unsqueeze(1)], 1).detach()
        grads_temp = self.model.forward(x_temp).detach()
        q1 = x0[:, 0] + dt * grads_temp[:, 0]
        x_temp2 = torch.cat([q1.unsqueeze(1), p_temp.unsqueeze(1)], 1).detach()
        grads_temp2 = self.model.forward(x_temp2).detach()
        p1 = p_temp + dt / 2 * grads_temp2[:, 1]
        return torch.cat([q1.unsqueeze(1).detach(), p1.unsqueeze(1).detach()], 1)

    def integrate(self, x_init, t_span, method='Euler'):
        x_path = torch.zeros([x_init.shape[0], x_init.shape[1], t_span.shape[0]]).to(x_init)
        x_path[:, :, 0] = x_init
        for count, t in enumerate(t_span):
            if count == 0:
                continue
            dt = t - t_span[count - 1]
            if method == 'Euler':
                x_path[:, :, count] = self.euler_step(x_path[:, :, count - 1], dt)
            elif method == 'SV':
                x_path[:, :, count] = self.sv_step(x_path[:, :, count - 1], dt)

        return x_path
