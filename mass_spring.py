from src.dennet import DENNet
from src.diffeq import DiffEq

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
        with torch.set_grad_enabled(True):
            x = x.requires_grad_(True)
            grad_h = torch.autograd.grad(self.H(x).sum(), x, allow_unused=False, create_graph=True)[0]
        return torch.cat([grad_h[:, self.n:], -grad_h[:, :self.n]], 1).to(x)


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

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model.de_function(0, x)
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
    t = torch.linspace(0, 1, 100).reshape(-1, 1)

    X = torch.cat([
        torch.sin(2 * np.pi * t),
        torch.cos(2 * np.pi * t)
    ], 1).to(device)

    y = torch.cat([
        torch.cos(2 * np.pi * t),
        -torch.sin(2 * np.pi * t)
    ], 1).to(device)

    train = data.TensorDataset(X, y)
    trainloader = data.DataLoader(train, batch_size=len(X), shuffle=False)

    HamFunc = HNN(nn.Sequential(
        nn.Linear(2, 128),
        nn.Tanh(),
        nn.Linear(128, 1))).to(device)

    model = DENNet(HamFunc).to(device)

    learn = Learner(model)
    trainer = pl.Trainer(min_epochs=200, max_epochs=200)
    trainer.fit(learn)

    X_t = torch.randn(1000, 2).to(device)
    # Evaluate the HNN trajectories for 1s
    s_span = torch.linspace(0, 1, 100)
    # This currently isn't used, why?
    traj = model.trajectory(X_t, s_span).detach().cpu()

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
    ax.streamplot(Q.T.numpy(), P.T.numpy(), U.T.numpy(), V.T.numpy(), color='black')
    ax.set_xlim([Q.min(), Q.max()])
    ax.set_ylim([P.min(), P.max()])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p$")
    # ax.set_title("Learned Hamiltonian & Vector Field")
    plt.show()
