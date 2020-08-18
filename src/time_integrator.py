import torch
import torch.nn as nn


class Time_integrator(nn.Module):
    """
    A class for integrating in time with a HNN

    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def euler_step(self, x0, dt):
        grads = self.model.forward(x0)
        x1 = x0 + dt*grads
        return x1

    def SV_step(self, x0, dt):
        grads0 = self.model.forward(x0)
        pTemp = x0[:, 1] + dt/2*grads0[:, 1]
        xTemp = torch.cat([x0[:, 0].unsqueeze(1), pTemp.unsqueeze(1)], 1)
        gradsTemp = self.model.forward(xTemp)
        q1 = x0[:, 0] + dt*gradsTemp[:, 0]
        xTemp2 = torch.cat([q1.unsqueeze(1), pTemp.unsqueeze(1)], 1)
        gradsTemp2 = self.model.forward(xTemp2)
        p1 = pTemp + dt/2*gradsTemp2[:, 1]
        return torch.cat([q1.unsqueeze(1), p1.unsqueeze(1)], 1)

    def integrate(self, xInit, tSpan, method='Euler'):
        xPath = torch.zeros([xInit.shape[0], xInit.shape[1], tSpan.shape[0]]).to(xInit)
        xPath[:, :, 0] = xInit
        for count, t in enumerate(tSpan):
            if count == 0:
                continue
            dt = t - tSpan[count-1]
            if method == 'Euler':
                xPath[:, :, count] = self.euler_step(xPath[:, :, count - 1], dt)
            elif method == 'SV':
                xPath[:, :, count] = self.SV_step(xPath[:, :, count - 1], dt)


        return xPath
