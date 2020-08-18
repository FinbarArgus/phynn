from src.dennet import DENNet
from src.time_integrator import Time_integrator

import torch
import torch.nn as nn
import torch.utils.data as data

import pytorch_lightning as pl

import numpy as np
import matplotlib.pyplot as plt


class HNN(nn.Module):
    def __init__(self, hamiltonian: nn.Module, dim=1):
        super().__init__()
        self.H = hamiltonian
        self.n = dim

    def forward(self, x):
        # TODO(Finbar) do I need to include a kwarg for grads=True so i can set it to false when evaluating time step
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            grad_h = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0]
        return torch.cat([grad_h[:, self.n:], -grad_h[:, :self.n]], 1).to(x)


class HNN_seperable(nn.Module):
    def __init__(self, hamiltonian: nn.Module, dim=1):
        super().__init__()
        self.H = hamiltonian
        self.n = dim

    def forward(self, x):
        # TODO(Finbar) do I need to include a kwarg for grads=True so i can set it to false when evaluating time step
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            # TODO(Fix below)
            grad_hq = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0][:,0].unsqueeze(1)
            grad_hp = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0][:,1].unsqueeze(1)
        return torch.cat([grad_hp, -grad_hq], 1).to(x)
    #TODO(Finbar) Forward_q and forward_p functions


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

    def calculate_f(self, x):
        q_dot = x[:, 1].unsqueeze(1)
        p_dot = -x[:, 0].unsqueeze(1)
        f = torch.cat([x[:, 1].unsqueeze(1), -x[:, 0].unsqueeze(1)], 1).to(x)
        return f

    def training_step(self, batch, batch_idx):
        x, yNotUsed = batch
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


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # time
    # t = torch.linspace(0, 1, 100).reshape(-1, 1)
    numTrainData = 1000

    X = torch.cat([
        (3*torch.rand(numTrainData) - 1.5).unsqueeze(1),
        (3*torch.rand(numTrainData) - 1.5).unsqueeze(1)
    ], 1).to(device)

    # TODO(Finbar) get rid of this, it is only used as input to TensorDataSet
    y = torch.cat([
        2*torch.rand(numTrainData).unsqueeze(1),
        2*torch.rand(numTrainData).unsqueeze(1)
    ], 1).to(device)
    # Wrap in for loop and change inputs to the stepped forward p's and q's

    # TODO(Finbar) figure out how to call this without y
    train = data.TensorDataset(X, y)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    HamFunc = HNN(nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 1))).to(device)

    HamFunc_sep = HNN_seperable(nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 1))).to(device)

    model = DENNet(HamFunc).to(device)
    model_sep = DENNet(HamFunc_sep).to(device)

    # setup train and train HNN
    learn = Learner(model)
    trainer = pl.Trainer(min_epochs=1000, max_epochs=1000)
    #this also takes in trainloader
    trainer.fit(learn)

    # train seperable Hamiltonian
    learn_sep = Learner(model_sep)
    trainer_sep = pl.Trainer(min_epochs=1000, max_epochs=1000)
    #this also takes in trainloader
    trainer_sep.fit(learn_sep)


    # grads = HamFunc.forward()

    # Wrap to here
    qInit = torch.linspace(0.2, 1.5, 3)
    pInit = torch.zeros(qInit.shape)
    xInit = torch.cat([qInit.unsqueeze(1), pInit.unsqueeze(1)],1).to(device)
    # xInit2 = torch.randn(4, 2).to(device)

    # Evaluate the HNN trajectories for 1s
    s_span = torch.linspace(0, 20, 400).to(device)
    # calculate trajectory with odeint
    # traj = model.trajectory(xInit, s_span).detach().cpu()

    # set up time integrator that uses our HNN
    timeIntegrator = Time_integrator(HamFunc).to(device)
    # calculate trajectory with an euler step
    traj_HNN_Euler = timeIntegrator.integrate(xInit, s_span, method='Euler').detach().cpu()

    # set up time integrator that uses our seperable HNN
    timeIntegrator = Time_integrator(HamFunc_sep).to(device)
    # calculate trajectory
    traj_HNN_SV = timeIntegrator.integrate(xInit, s_span, method='SV').detach().cpu()

    n_grid = 50
    x = torch.linspace(-2, 2, n_grid)
    Q, P = torch.meshgrid(x, x)
    H, U, V = torch.zeros(Q.shape), torch.zeros(Q.shape), torch.zeros(Q.shape)
    for i in range(n_grid):
        for j in range(n_grid):
            x = torch.cat([Q[i, j].reshape(1, 1), P[i, j].reshape(1, 1)], 1).to(device)
            H[i, j] = model.de_function.model.H(x).detach().cpu()
            O = model.de_function(0, x).detach().cpu()
            U[i, j], V[i, j] = O[0, 0], O[0, 1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.contourf(Q, P, H, 100, cmap='seismic')
    # ax.streamplot(Q.T.numpy(), P.T.numpy(), U.T.numpy(), V.T.numpy(), color='black')
    # ax.plot(traj[:, 0, 0], traj[:, 0, 1], color='k')
    for count in range(len(traj_HNN_Euler[:, 0, 0])-1):
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
