import torch
import torch.nn as nn
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from maths.dennet import DENNet
from mechanics.hamiltonian import HNN, HNNSeparable
from src.time_integrator import TimeIntegrator

import matplotlib.pyplot as plt


class Learner(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.c = 0

    def forward(self, x):
        return self.model.de_function(0, x)

    @staticmethod
    def loss(y, y_hat):
        return ((y - y_hat) ** 2).sum()

    @staticmethod
    def calculate_f(x):
        q_dot = x[:, 1].unsqueeze(1)
        p_dot = -x[:, 0].unsqueeze(1)
        f = torch.cat([q_dot, p_dot], 1).to(x)
        return f

    def training_step(self, batch, batch_idx):
        x = batch[0]
        y_hat = self.model.de_function(0, x)
        # will need to calculate a y here from the equation residual
        y = self.calculate_f(x)
        loss = self.loss(y_hat, y)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

    @staticmethod
    def train_dataloader():
        return trainloader


def basic_hnn():
    """
    Simple Hamiltonian network.

    :return:
    """
    h = HNN(nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 1))).to(device)

    model = DENNet(h).to(device)
    learn = Learner(model)
    logger = TensorBoardLogger('HNN_logs')
    trainer = pl.Trainer(gpus=1, min_epochs=500, max_epochs=1000, logger=logger)
    trainer.fit(learn)

    return h, model


def separable_hnn():
    """
    Separable Hamiltonian network.

    :return:
    """
    h_s = HNNSeparable(nn.Sequential(
        nn.Linear(2, 100),
        nn.Tanh(),
        nn.Linear(100, 1))).to(device)

    model = DENNet(h_s).to(device)
    learn_sep = Learner(model)
    logger = TensorBoardLogger('separable_logs')
    trainer_sep = pl.Trainer(min_epochs=500, max_epochs=1000, logger=logger)
    trainer_sep.fit(learn_sep)

    return h_s, model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_train_data = 1000

    X = torch.cat([
        (3 * torch.rand(num_train_data) - 1.5).unsqueeze(1),
        (3 * torch.rand(num_train_data) - 1.5).unsqueeze(1)
    ], 1).to(device)

    # TODO Wrap in for loop and change inputs to the stepped forward p's and q's

    train = data.TensorDataset(X)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    hamiltonian, basic_model = basic_hnn()
    separable, separable_model = separable_hnn()

    q_init = torch.linspace(0.2, 1.5, 3)
    p_init = torch.zeros(q_init.shape)
    x_init = torch.cat([q_init.unsqueeze(1), p_init.unsqueeze(1)], 1).to(device)

    # Evaluate the HNN trajectories for 1s
    t_span = torch.linspace(0, 20, 400).to(device)
    # calculate trajectory with odeint
    # traj = model.trajectory(xInit, s_span).detach().cpu()

    # set up time integrator that uses our HNN
    time_integrator = TimeIntegrator(hamiltonian).to(device)
    # calculate trajectory with an euler step
    traj_HNN_Euler = time_integrator.integrate(x_init, t_span, method='Euler').detach().cpu()

    # set up time integrator that uses our separable HNN
    time_integrator = TimeIntegrator(separable).to(device)
    # calculate trajectory
    traj_HNN_SV = time_integrator.integrate(x_init, t_span, method='SV').detach().cpu()

    n_grid = 50
    x = torch.linspace(-2, 2, n_grid)
    Q, P = torch.meshgrid(x, x)
    H, U, V = torch.zeros(Q.shape), torch.zeros(Q.shape), torch.zeros(Q.shape)
    for i in range(n_grid):
        for j in range(n_grid):
            x = torch.cat([Q[i, j].reshape(1, 1), P[i, j].reshape(1, 1)], 1).to(device)
            H[i, j] = basic_model.de_function.model.H(x).detach().cpu()
            O = basic_model.de_function(0, x).detach().cpu()
            U[i, j], V[i, j] = O[0, 0], O[0, 1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.contourf(Q, P, H, 100, cmap='seismic')
    # ax.streamplot(Q.T.numpy(), P.T.numpy(), U.T.numpy(), V.T.numpy(), color='black')
    # ax.plot(traj[:, 0, 0], traj[:, 0, 1], color='k')
    for count in range(len(traj_HNN_Euler[:, 0, 0]) - 1):
        ax.plot(traj_HNN_Euler[count, 0, :], traj_HNN_Euler[count, 1, :], color='y')
        ax.plot(traj_HNN_SV[count, 0, :], traj_HNN_SV[count, 1, :], color='g')
    # plot last index with a label for legend
    count = count + 1
    ax.plot(traj_HNN_Euler[count, 0, :], traj_HNN_Euler[count, 1, :], color='y', label='Euler')
    ax.plot(traj_HNN_SV[count, 0, :], traj_HNN_SV[count, 1, :], color='g', label='Stormer-Verlet')

    ax.legend()
    ax.set_xlim([Q.min(), Q.max()])
    ax.set_ylim([P.min(), P.max()])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p$")
    plt.show()
