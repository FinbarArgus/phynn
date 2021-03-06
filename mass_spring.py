import torch
import torch.nn as nn
import torch.utils.data as data

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger


from src.maths.dennet import DENNet
from src.mechanics.hamiltonian import HNNMassSpring, HNNMassSpringSeparable
from src.time_integrator import TimeIntegrator
from src.utils.activation import Siren

import matplotlib.pyplot as plt


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


class SirenNet(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, w0=30., w0_initial=30.):
        super().__init__()
        layers = list()
        for ind in range(num_layers):
            is_first = ind == 0
            layer_w0 = w0_initial if is_first else w0
            layer_dim_in = dim_in if is_first else dim_hidden

            layers.append(Siren(
                dim_in=layer_dim_in,
                dim_out=dim_hidden,
                w0=layer_w0,
                is_first=is_first
            ))

        self.net = nn.Sequential(*layers)
        self.last_layer = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        x = self.net(x)
        return self.last_layer(x)


def basic_hnn():
    """
    Simple Hamiltonian network.

    :return:
    """
    h = HNNMassSpring(nn.Sequential(
        nn.Linear(2, 50),
        nn.Tanh(),
        nn.Linear(50, 1))).to(device)

    model = DENNet(h, case='springmass').to(device)
    learn = Learner(model)
    logger = TensorBoardLogger('HNN_logs')
    trainer = pl.Trainer(gpus=1, min_epochs=50, max_epochs=300, logger=logger)
    trainer.fit(learn)

    return h, model


def separable_hnn(input_h_s=None, input_model=None):
    """
    Separable Hamiltonian network.

    :return:
    """
    if input_h_s:
        h_s = input_h_s
        model = input_model
    else:
        h_s = HNNMassSpringSeparable(nn.Sequential(
            nn.Linear(2, 100),
            nn.Tanh(),
            nn.Linear(100, 1))).to(device)
        model = DENNet(h_s, case='springmass').to(device)

    learn_sep = Learner(model)
    logger = TensorBoardLogger('separable_logs')
    trainer_sep = pl.Trainer(min_epochs=50, max_epochs=100, logger=logger)
    trainer_sep.fit(learn_sep)

    return h_s, model


def separable_hnn_siren(input_h_s=None, input_model=None):
    """
    Separable Hamiltonian network using Siren network.

    :return:
    """
    if input_h_s:
        h_s = input_h_s
        model = input_model
    else:
        network = SirenNet(
            dim_in=2,
            dim_hidden=100,
            dim_out=1,
            num_layers=5,
            w0_initial=30.
        )

        h_s = HNNMassSpringSeparable(network).to(device)
        model = DENNet(h_s).to(device)

    learn_sep = Learner(model)
    logger = TensorBoardLogger('separable_logs_siren')
    trainer_sep = pl.Trainer(min_epochs=50, max_epochs=100, logger=logger, gpus=1)
    trainer_sep.fit(learn_sep)

    return h_s, model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training conditions
    num_train_data = 100
    num_tSteps_training = 5

    X_sv = torch.cat([  # Training initial conditions
        (3 * torch.rand(num_train_data) - 1.5).unsqueeze(1),
        (3 * torch.rand(num_train_data) - 1.5).unsqueeze(1)
    ], 1).to(device)

    dt_train = 0.05  # Training time step

    # Testing conditions
    q_init = torch.linspace(0.2, 1.5, 3)
    p_init = torch.zeros(q_init.shape)
    x_init = torch.cat([q_init.unsqueeze(1), p_init.unsqueeze(1)], 1).to(device)

    t_span_test = torch.linspace(0, 20, 400).to(device)  # Testing time span

    # Wrap in for loop and change inputs to the stepped forward p's and q's
    for tStep in range(num_tSteps_training):

        train = data.TensorDataset(X_sv)
        trainloader = data.DataLoader(train, batch_size=len(X_sv), shuffle=False)

        # hamiltonian, basic_model = basic_hnn()
        if tStep == 0:
            separable, separable_model = separable_hnn_siren()
        else:
            separable, separable_model = separable_hnn_siren(input_h_s=separable, input_model=separable_model)

        time_integrator_sv = TimeIntegrator(separable).to(device)

        X_sv = time_integrator_sv.sv_step(X_sv, dt_train).detach()

    # set up time integrator that uses our separable HNN
    time_integrator_sv = TimeIntegrator(separable).to(device)
    # calculate trajectory
    traj_HNN_sv = time_integrator_sv.integrate(x_init, t_span_test, method='SV').detach().cpu()

    n_grid = 50
    x = torch.linspace(-2, 2, n_grid)
    Q, P = torch.meshgrid(x, x)
    H, U, V = torch.zeros(Q.shape), torch.zeros(Q.shape), torch.zeros(Q.shape)
    for i in range(n_grid):
        for j in range(n_grid):
            x = torch.cat([Q[i, j].reshape(1, 1), P[i, j].reshape(1, 1)], 1).to(device)
            H[i, j] = separable_model.de_function.model.H(x).detach().cpu()
            O = separable_model.de_function(0, x).detach().cpu()
            U[i, j], V[i, j] = O[0, 0], O[0, 1]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.contourf(Q, P, H, 100, cmap='seismic')
    # ax.streamplot(Q.T.numpy(), P.T.numpy(), U.T.numpy(), V.T.numpy(), color='black')
    # ax.plot(traj[:, 0, 0], traj[:, 0, 1], color='k')
    for count in range(len(traj_HNN_sv[:, 0, 0]) - 1):
        # ax.plot(traj_HNN_Euler[count, 0, :], traj_HNN_Euler[count, 1, :], color='y')
        ax.plot(traj_HNN_sv[count, 0, :], traj_HNN_sv[count, 1, :], color='g')
    # plot last index with a label for legend
    count = count + 1
    # ax.plot(traj_HNN_Euler[count, 0, :], traj_HNN_Euler[count, 1, :], color='y', label='Euler')
    ax.plot(traj_HNN_sv[count, 0, :], traj_HNN_sv[count, 1, :], color='g', label='Stormer-Verlet')

    ax.legend()
    ax.set_xlim([Q.min(), Q.max()])
    ax.set_ylim([P.min(), P.max()])
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$p$")
    plt.show()
