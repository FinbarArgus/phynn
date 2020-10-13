import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from src.maths.dennet import DENNet
from src.mechanics.hamiltonian import HNN1DWaveSeparable
from src.time_integrator_1DWave import TimeIntegrator

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Learner(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.c = 0

    def forward(self, x):
        return self.model.de_function(0, x)

    def backward(self, use_amp, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)
        return

    @staticmethod
    def loss(y, y_hat):
        return ((y - y_hat) ** 2).sum()

    @staticmethod
    def calculate_f(x, dq_dx, dp_dx):
        # this function calculates f = (q_dot, p_dot) = (dp_dx, -dq_dx)
        return dp_dx, -dq_dx

    def training_step(self, batch, batch_idx):
        x = batch[0][0]
        q = batch[1][0]
        p = batch[2][0]
        dq_dx = batch[3][0]
        dp_dx = batch[4][0]
        # Calculate y_hat = (q_dot_hat, p_dot_hat) from the gradient of the HNN
        q_dot_hat, p_dot_hat = self.model.de_function(0, x, q, p)
        y_hat = torch.cat([q_dot_hat, p_dot_hat])
        # Calculate the y = (q_dot, p_dot) from the governing equations
        q_dot, p_dot = self.calculate_f(x, dq_dx, dp_dx)
        # set the boundary q_dot and p_dot to zero
        # TODO create an apply_boundary_conditions function
        p_dot[0] = 0
        p_dot[-1] = 0

        y = torch.cat([q_dot, p_dot])
        loss = self.loss(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)

    @staticmethod
    def train_dataloader():
        return trainloader


def separable_hnn(num_points, input_h_s=None, input_model=None):
    """
    Separable Hamiltonian network.

    :return:
    """
    if input_h_s:
        h_s = input_h_s
        model = input_model
    else:
        h_s = HNN1DWaveSeparable(nn.Sequential(
            nn.Linear(3*num_points, 30),
            nn.Tanh(),
            nn.Linear(30, 30),
            nn.Tanh(),
            nn.Linear(30, 1)), num_points).to(device)
        model = DENNet(h_s, case='wave').to(device)

    learn_sep = Learner(model)
    logger = TensorBoardLogger('separable_logs')
    trainer_sep = pl.Trainer(min_epochs=300, max_epochs=300, logger=logger)
    trainer_sep.fit(learn_sep)

    return h_s, model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training conditions
    num_train_samples = 1
    num_train_xCoords = 20
    num_tSteps_training = 1
    num_batches = 30
    # Training initial conditions [q, p, dq/dx, dp/dx, x]
    x_coord = torch.zeros(num_batches, num_train_xCoords).to(device)
    x_coord[:, 1:-1] = torch.rand(num_batches, num_train_xCoords-2).sort()[0]
    x_coord[:, -1] = 1.0
    q = torch.zeros(num_batches, num_train_xCoords).to(device)
    p = torch.zeros(num_batches, num_train_xCoords).to(device)
    dq_dx = torch.zeros(num_batches, num_train_xCoords).to(device)
    dp_dx = torch.zeros(num_batches, num_train_xCoords).to(device)
    for batch_idx in range(num_batches):
        amplitude_q = 2*np.random.rand(1)[0]-1
        # amplitude_p = 2*np.random.rand(1)[0]-1
        q[batch_idx, :] = amplitude_q*np.pi * torch.cos(np.pi * x_coord[batch_idx, :]).to(device)
        # p[batch_idx, :] = amplitude_p*np.pi * torch.sin(np.pi * x_coord).to(device)
        dq_dx[batch_idx, :] = -amplitude_q*np.pi ** 2 * torch.sin(np.pi * x_coord[batch_idx, :]).to(device)
        # dq_dx[batch_idx, :] = amplitude_p*np.pi ** 2 * torch.cos(np.pi * x_coord).to(device)

    # X_sv = torch.cat([
    #     x_coord,
    #     np.pi*torch.cos(np.pi*x_coord),
    #     torch.zeros(num_train_xCoords),
    #     -np.pi**2*torch.sin(np.pi*x_coord),
    #     torch.zeros(num_train_xCoords)
    # ]).to(device)
    # X_euler = X_sv
    # Training time step
    dt_train = 0.01

    # Testing conditions
    # temporarily test with the same as one of the training init conditions
    x_coord_test = x_coord[0, :]
    #x_coord_test = torch.rand(num_train_xCoords).to(device)
    q_test = np.pi * torch.cos(np.pi * x_coord_test).to(device)
    p_test = torch.zeros(num_train_xCoords).to(device)
    dq_dx_test = -np.pi ** 2 * torch.sin(np.pi * x_coord_test).to(device)
    dp_dx_test = torch.zeros(num_train_xCoords).to(device)
    # Testing time span
    t_span_test = torch.linspace(0, 4, 400).to(device)

    # Wrap in for loop and change inputs to the stepped forward p's and q's
    for tStep in range(num_tSteps_training):

        train = data.TensorDataset(x_coord, q, p, dq_dx, dp_dx)
        trainloader = data.DataLoader(train, batch_size=1, shuffle=True)

        # hamiltonian, basic_model = basic_hnn()
        if tStep == 0:
            separable, separable_model = separable_hnn(num_train_xCoords)
        else:
            separable, separable_model = separable_hnn(num_train_xCoords,
                                                       input_h_s=separable, input_model=separable_model)

        # set up time integrator that uses our HNN
        # time_integrator_euler = TimeIntegrator(hamiltonian).to(device)
        # time_integrator_sv = TimeIntegrator(separable).to(device)

        # Evaluate the HNN trajectory for 1 step and then reset the initial condition for more training
        # # # # q, p = time_integrator_sv.sv_step(x_coord, q, p, dt_train)
        # q, p, dq_dx, dp_dx = time_integrator_sv.sv_step_wgrads(x_coord, q, p, dq_dx, dp_dx, dt_train)
        # q = q.detach()
        # p = p.detach()
        # dq_dx = dq_dx.detach()
        # dp_dx = dp_dx.detach()

    print('model finished training')

    # calculate trajectory with odeint
    # traj = model.trajectory(xInit, s_span).detach().cpu()

    # set up time integrator that uses our HNN
    # time_integrator_euler = TimeIntegrator(hamiltonian).to(device)
    # calculate trajectory with an euler step
    # traj_HNN_Euler = time_integrator_euler.integrate(X_test, t_span_test, method='Euler').detach().cpu()

    # set up time integrator that uses our separable HNN
    time_integrator_sv = TimeIntegrator(separable).to(device)
    # calculate trajectory
    q_traj, p_traj = time_integrator_sv.integrate(x_coord_test, q_test, p_test,
                                                  t_span_test, method='SV')
    x_coord_test = x_coord_test.detach().cpu()
    q_traj = q_traj.detach().cpu()
    p_traj = p_traj.detach().cpu()

    fig = plt.figure(figsize=(8, 8))

    ax = plt.axes(xlim=(0, 1), ylim=(-2, 2))
    line, = ax.plot([], [], lw=3)

    def init():
        line.set_data([], [])
        return line,


    def animate(i):
        line.set_data(x_coord_test, q_traj[:, i])
        return line,


    anim = FuncAnimation(fig, animate, frames=len(t_span_test), init_func=init, blit=True)
    plt.show()
    #anim.save('test_anim.gif', writer='ffmpeg')



