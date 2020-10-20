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

    def sv_step(self, x, q_0, p_0, dt):
        """
        Stormer-Verlet (SV) (aka leapfrog) time integration scheme.

        :param x:
        :param q_0:
        :param p_0:
        :param dt:
        :return:
        """
        q_dot_0, p_dot_0 = self.model.forward(x, q_0, p_0, detach=True)
        x = x.detach()
        q_0 = q_0.detach()
        p_0 = p_0.detach()

        p_temp = p_0 + dt / 2 * p_dot_0

        q_dot_temp, p_dot_temp = self.model.forward(x, q_0, p_temp, detach=True)
        x = x.detach()
        q_0 = q_0.detach()
        p_temp = p_temp.detach()

        q_1 = q_0 + dt * q_dot_temp

        q_dot_temp2, p_dot_temp2 = self.model.forward(x, q_1, p_temp, detach=True)
        # x = x.detach()
        q_1 = q_1.detach()
        p_temp = p_temp.detach()

        p_1 = p_temp + dt / 2 * p_dot_temp2

        return q_1, p_1

    def sv_step_wgrads(self, x, q_0, p_0, dq_dx_0, dp_dx_0, dt):
        """
        Stormer-Verlet (SV) (aka leapfrog) time integration scheme.

        :param x:
        :param q_0:
        :param p_0:
        :param dq_dx_0:
        :param dp_dx_0:
        :param dt:
        :return:
        """
        q_dot_0, p_dot_0, dq_dot_dx_0, dp_dot_dx_0 = self.model.forward_wgrads(x, q_0, p_0)
        x = x.detach()
        q_0 = q_0.detach()
        p_0 = p_0.detach()
        dq_dx_0 = dq_dx_0.detach()
        dp_dx_0 = dp_dx_0.detach()

        p_temp = p_0 + dt / 2 * p_dot_0
        dp_dx_temp = dp_dx_0 + dt / 2 * dp_dot_dx_0
        q_dot_temp, p_dot_temp, dq_dot_dx_temp, dp_dot_dx_temp = self.model.forward_wgrads(x, q_0, p_temp)
        x = x.detach()
        q_0 = q_0.detach()
        p_temp = p_temp.detach()
        dq_dx_0 = dq_dx_0.detach()
        dp_dx_temp = dp_dx_temp.detach()

        q_1 = q_0 + dt * q_dot_temp
        dq_dx_1 = dq_dx_0 + dt * dq_dot_dx_temp
        q_dot_temp2, p_dot_temp2, dq_dot_dx_temp2, dp_dot_dx_temp2 = self.model.forward_wgrads(x, q_1, p_temp)
        # x = x.detach()
        q_1 = q_1.detach()
        p_temp = p_temp.detach()
        dq_dx_1 = dq_dx_1.detach()
        dp_dx_temp = dp_dx_temp.detach()

        p_1 = p_temp + dt / 2 * p_dot_temp2
        dp_dx_1 = dp_dx_temp + dt / 2 * dp_dot_dx_temp2

        return q_1, p_1, dq_dx_1, dp_dx_1

    def integrate(self, x_0, q_0, p_0, t_span, method='SV'):
        q_path = torch.zeros([1, x_0[0].shape[0], t_span.shape[0]]).to(x_0)
        p_path = torch.zeros([1, x_0[0].shape[0], t_span.shape[0]]).to(x_0)
        H_path = torch.zeros([t_span.shape[0]]).to(x_0)
        q_path[:, :, 0] = q_0
        p_path[:, :, 0] = p_0
        H_path[0] = self.model.H(torch.cat([x_0[0, :], q_0[0, :], p_0[0, :]]))
        print('integrating for trajectory')
        for count, t in enumerate(t_span):
            print(count)
            if count == 0:
                continue
            dt = t - t_span[count - 1]
            if method == 'Euler':
                q_path[:, :, count], p_path[:, :, count] = self.euler_step(x_0, q_path[:, :, count - 1],
                                                                           p_path[:, :, count - 1], dt)
            elif method == 'SV':
                q_path[:, :, count], p_path[:, :, count] = self.sv_step(x_0, q_path[:, :, count - 1],
                                                                        p_path[:, :, count - 1], dt)
            H_path[count] = self.model.H(torch.cat([x_0[0, :], q_path[0, :, count],
                                                          p_path[0, :, count]]))

        return q_path, p_path, H_path
