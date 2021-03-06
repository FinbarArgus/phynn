import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

import time
import os

from src.maths.dennet import DENNet
from src.mechanics.hamiltonian import HNN1DWaveSeparable
from src.time_integrator_1DWave import TimeIntegrator

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Learner(pl.LightningModule):
    def __init__(self, model: nn.Module, num_boundary=1, save_path='temp_save_path',
                 epoch_save=100):
        super().__init__()
        self.model = model
        self.c = 0
        self.num_boundary = num_boundary
        self.clockTime = time.process_time()
        self.save_path = save_path
        self.epoch_save = epoch_save

    def forward(self, x):
        return self.model.de_function(0, x)

    def backward(self, use_amp, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)
        return

    @staticmethod
    def loss(y, y_hat, batch_size):
        return ((y - y_hat) ** 2).sum()/batch_size

    @staticmethod
    def calculate_f(x, dq_dx, dp_dx):
        # this function calculates f = (q_dot, p_dot) = (dp_dx, -dq_dx)
        return -dp_dx, -dq_dx

    @staticmethod
    def add_gaussian_noise(input, mean, stddev):
        noise = input.data.new(input.size()).normal_(mean, stddev)
        return input + noise

    def training_step(self, batch, batch_idx):
        batch_size = len(batch[0])

        # get input data and add noise
        x = batch[0]
        q = self.add_gaussian_noise(batch[1], 0.0, 0.05)
        p = self.add_gaussian_noise(batch[2], 0.0, 0.05)
        dq_dx = self.add_gaussian_noise(batch[3], 0.0, 0.05)
        dp_dx =  self.add_gaussian_noise(batch[4], 0.0, 0.05)


        # Calculate y_hat = (q_dot_hat, p_dot_hat) from the gradient of the HNN
        q_dot_hat, p_dot_hat = self.model.de_function(0, x, q, p)
        y_hat = torch.cat([q_dot_hat, p_dot_hat], dim=1)
        # Calculate the y = (q_dot, p_dot) from the governing equations
        q_dot, p_dot = self.calculate_f(x, dq_dx, dp_dx)
        # The below two lines currently aren't needed because p_dot is equal to -dq_dx which is equal to 0
        # at the first and last num_boundary points because x is set to 0 and 1 on the boundaries
        p_dot[:, 0:self.num_boundary] = 0
        p_dot[:, -1*self.num_boundary:] = 0
        y = torch.cat([q_dot, p_dot], dim=1)
        # set the boundary q_dot and p_dot to zero
        # TODO create an apply_boundary_conditions function
        # TODO make sure I am setting the correct numbers to zero here

        loss = self.loss(y_hat, y, batch_size)
        logs = {'train_loss': loss}

        #if epoch is a multiple of specified number then save the model
        if self.current_epoch != 0 and self.current_epoch%self.epoch_save == 0 and batch_idx == 0:
            print(' Saving model for epoch number {}, in {}'.format(self.current_epoch,
                                                                    self.save_path))
            torch.save({
                'epoch': self.current_epoch,
                'model_state_dict': self.model.state_dict(),
                'loss': loss,
                }, self.save_path)
            # 'optimizer_state_dict': optimizer.state_dict(),

        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.0001)

    @staticmethod
    def train_dataloader():
        return trainloader


def separable_hnn(num_points, input_h_s=None, input_model=None,
                  save_path='temp_save_path', train=True, epoch_save=100):
    """
    Separable Hamiltonian network.

    :return:
    """
    if input_h_s:
        h_s = input_h_s
        model = input_model
    else:
        h_s = HNN1DWaveSeparable(nn.Sequential(
            nn.Linear(3*num_points, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 1))).to(device)
        model = DENNet(h_s, case='1DWave').to(device)

    if train:
        learn_sep = Learner(model, num_boundary=num_boundary, save_path=save_path,
                            epoch_save=epoch_save)
        logger = TensorBoardLogger('separable_logs')
        trainer_sep = pl.Trainer(min_epochs=701, max_epochs=701, logger=logger, gpus=1)
        trainer_sep.fit(learn_sep)

    return h_s, model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using cuda device: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))

    #save path
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_name = 'model_6Layers_20Neurons.pt'
    save_path = os.path.join(save_dir, model_name)
    # save every epoch_save epochs
    epoch_save = 50

    # Training conditions
    num_train_samples = 1
    num_train_xCoords = 60
    num_tSteps_training = 1
    num_boundary = 5
    num_datasets = 8192
    batch_size = 1024
    # Training initial conditions [q, p, dq/dx, dp/dx, x]
    x_coord = torch.zeros(num_datasets, num_train_xCoords).to(device)
    x_coord[:, num_boundary:-1*num_boundary] = torch.rand(num_datasets, num_train_xCoords-num_boundary*2).sort()[0]
    x_coord[:, -1*num_boundary:] = 1.0
    q = torch.zeros(num_datasets, num_train_xCoords).to(device)
    p = torch.zeros(num_datasets, num_train_xCoords).to(device)
    dq_dx = torch.zeros(num_datasets, num_train_xCoords).to(device)
    dp_dx = torch.zeros(num_datasets, num_train_xCoords).to(device)
    for batch_idx in range(num_datasets):
        amplitude_q = 2*np.random.rand(1)[0]-1
        amplitude_p = 2*np.random.rand(1)[0]-1
        q[batch_idx, :] = amplitude_q*np.pi * torch.cos(np.pi * x_coord[batch_idx, :]).to(device)
        p[batch_idx, :] = amplitude_p*np.pi * torch.sin(np.pi * x_coord[batch_idx, :]).to(device)
        dq_dx[batch_idx, :] = -amplitude_q*np.pi ** 2 * torch.sin(np.pi * x_coord[batch_idx, :]).to(device)
        dp_dx[batch_idx, :] = amplitude_p*np.pi ** 2 * torch.cos(np.pi * x_coord[batch_idx, :]).to(device)

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
    x_coord_test = x_coord[0:1, :]
    amplitude_q_test = 0.5
    #x_coord_test = torch.rand(num_train_xCoords).to(device)
    q_test = amplitude_q_test * np.pi * torch.cos(np.pi * x_coord_test).to(device)
    p_test = torch.zeros(num_train_xCoords).to(device)
    dq_dx_test = -amplitude_q_test*np.pi ** 2 * torch.sin(np.pi * x_coord_test).to(device)
    dp_dx_test = torch.zeros(num_train_xCoords).to(device)
    # Testing time span
    t_span_test = torch.linspace(0, 0.3, 60).to(device)

    # bool to determine if model is loaded
    load_model = False

    # Wrap in for loop and change inputs to the stepped forward p's and q's
    for tStep in range(num_tSteps_training):

        train = data.TensorDataset(x_coord, q, p, dq_dx, dp_dx)
        trainloader = data.DataLoader(train, batch_size=batch_size, shuffle=True)

        # hamiltonian, basic_model = basic_hnn()
        # load model if required
        if tStep == 0 and load_model:
            separable, separable_model = separable_hnn(num_train_xCoords, train=False)
            # optimizer = TheOptimizerClass(*args, **kwargs)

            checkpoint = torch.load(save_path)
            separable_model.load_state_dict(checkpoint['model_state_dict'])
            separable = separable_model.de_function.model
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print('loaded model at epoch {} from {}'.format(epoch, save_path))

        if tStep == 0 and not load_model:
            separable, separable_model = separable_hnn(num_train_xCoords, save_path=save_path,
                                                       epoch_save=epoch_save)
        else:
            separable, separable_model = separable_hnn(num_train_xCoords,
                                                       input_h_s=separable, input_model=separable_model,
                                                       save_path=save_path, epoch_save=epoch_save)

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
    line2, = ax.plot([], [], lw=3, color='r')

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        return line,


    def animate(i):
        line.set_data(x_coord_test[0], q_traj[0, :, i])
        line2.set_data(x_coord_test[0], p_traj[0, :, i])
        return line, line2,


    anim = FuncAnimation(fig, animate, frames=len(t_span_test), init_func=init, blit=True)
    plt.show()
    # anim.save('test_anim.gif', writer='ffmpeg')



