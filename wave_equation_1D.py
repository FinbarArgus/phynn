import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from matplotlib.animation import PillowWriter

import time
import os

from src.maths.dennet import DENNet
from src.mechanics.hamiltonian import HNN1DWaveSeparable
from src.time_integrator_1DWave import TimeIntegrator

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)

class Learner(pl.LightningModule):
    def __init__(self, model: nn.Module, num_boundary=1, save_path='temp_save_path',
                 epoch_save=100, dt=0.01):
        super().__init__()
        self.model = model
        self.c = 0
        self.eps = 1e-9
        self.num_boundary = num_boundary
        self.clockTime = time.process_time()
        self.save_path = save_path
        self.epoch_save = epoch_save
        self.dt=dt

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

        # Note: training with gradients slows down training to a crawl
        train_wgrads = True
        train_wH = True

        # get input data and add noise
        x = batch[0]
        q = self.add_gaussian_noise(batch[1], 0.0, 0.05)
        p = self.add_gaussian_noise(batch[2], 0.0, 0.05)
        dq_dx = self.add_gaussian_noise(batch[3], 0.0, 0.05)
        dp_dx =  self.add_gaussian_noise(batch[4], 0.0, 0.05)


        # Calculate y_hat = (q_dot_hat, p_dot_hat) from the gradient of the HNN
        q_dot_hat, p_dot_hat, _ = self.model.de_function(0, x, q, p)

        y_hat = torch.cat([q_dot_hat, p_dot_hat], dim=1)
        # Calculate the y = (q_dot, p_dot) from the governing equations
        q_dot, p_dot = self.calculate_f(x, dq_dx, dp_dx)
        # The below two lines currently aren't needed because p_dot is equal to -dq_dx which is equal to 0
        # at the first and last num_boundary points because x is set to 0 and 1 on the boundaries
        p_dot[:, 0:self.num_boundary] = 0
        p_dot[:, -1*self.num_boundary:] = 0
        y = torch.cat([q_dot, p_dot], dim=1)

        loss_main = self.loss(y_hat, y, batch_size)
        # add the spatial gradients to the loss function to make them smooth
        loss_grads = 0
        loss_H_const = 0
        if train_wgrads:
            grad_scale = 0.005/batch_size
            q_dot_diff_approx = torch.abs((q_dot_hat[:, self.num_boundary + 1:-self.num_boundary] -
                                 q_dot_hat[:, self.num_boundary:-self.num_boundary-1]) / \
                                (x[:, self.num_boundary + 1:-self.num_boundary] -
                                 x[:, self.num_boundary:-self.num_boundary-1] + self.eps))
            p_dot_diff_approx = torch.abs((p_dot_hat[:, self.num_boundary + 1:-self.num_boundary] -
                                           p_dot_hat[:, self.num_boundary:-self.num_boundary-1]) / \
                                          (x[:, self.num_boundary + 1:-self.num_boundary] -
                                           x[:, self.num_boundary:-self.num_boundary-1] + self.eps))
            loss_grads += grad_scale*(q_dot_diff_approx.sum() + p_dot_diff_approx.sum())
        if train_wH:
            H_train_scale = 0.03/(self.dt*batch_size)
            time_integrator = TimeIntegrator(self.model.de_function.func).to(device)
            # TODO should i include noise here?
            q_new, p_new, H_old = time_integrator.sv_step(x, q, p, self.dt)
            H_new = self.model.de_function.func.H(torch.cat([x, q_new, p_new], dim=1))
            loss_H_const += H_train_scale*abs(H_new - H_old).sum()
        loss = loss_main + loss_grads + loss_H_const
        logs = {'train_loss': loss, 'loss_main': loss_main,
                'loss_grads': loss_grads, 'loss_H':loss_H_const}

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
        return torch.optim.Adam(self.model.parameters(), lr=0.0007, weight_decay=0.01)

    @staticmethod
    def train_dataloader():
        return trainloader


def separable_hnn(num_points, input_h_s=None, input_model=None,
                  save_path='temp_save_path', train=True, epoch_save=100, dt=0.01):
    """
    Separable Hamiltonian network.

    :return:
    """
    if input_h_s:
        h_s = input_h_s
        model = input_model
    else:
        h_s = HNN1DWaveSeparable(nn.Sequential(
            nn.Linear(3*num_points, 40),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(40, 40),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(40, 30),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(30, 20),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(20, 10),
            nn.Dropout(0.2),
            nn.Tanh(),
            nn.Linear(10, 1),
            nn.Dropout(0.2),
            nn.Softplus())).to(device)
        for m in h_s.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

        model = DENNet(h_s, case='1DWave').to(device)


    if train:
        learn_sep = Learner(model, num_boundary=num_boundary, save_path=save_path,
                            epoch_save=epoch_save, dt=dt)
        logger = TensorBoardLogger('separable_logs')
        trainer_sep = pl.Trainer(min_epochs=2001, max_epochs=2001, logger=logger, gpus=1)
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
    num_train_xCoords = 20
    num_tSteps_training = 1
    num_boundary = 2
    num_datasets = 65536
    batch_size = 8192
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
    dt_train = 0.001

    # Testing conditions
    # temporarily test with the same as one of the training init conditions
    # x_coord_test = x_coord[0:1, :]
    x_coord_test = torch.zeros(1, num_train_xCoords).to(device)
    x_coord_test[:1, num_boundary-1:-num_boundary+1] = torch.linspace(0, 1.0, num_train_xCoords-num_boundary)
    x_coord_test[:1, -num_boundary:] = 1.0
    amplitude_q_test = 0.5
    #x_coord_test = torch.rand(num_train_xCoords).to(device)
    q_test = amplitude_q_test * np.pi * torch.cos(np.pi * x_coord_test).to(device)
    p_test = torch.zeros(1, num_train_xCoords).to(device)
    dq_dx_test = -amplitude_q_test*np.pi ** 2 * torch.sin(np.pi * x_coord_test).to(device)
    dp_dx_test = torch.zeros(1, num_train_xCoords).to(device)
    # Testing time span
    t_span_test = torch.linspace(0, 8.0, 1600).to(device)

    # bool to determine if model is loaded
    load_model = True
    do_train = False

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
            separable = separable_model.de_function.func
            # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            loss = checkpoint['loss']

            print('loaded model at epoch {} from {}'.format(epoch, save_path))

        if tStep == 0 and not load_model:
            if do_train:
                separable, separable_model = separable_hnn(num_train_xCoords, save_path=save_path,
                                                           epoch_save=epoch_save, dt=dt_train)
            else:
                print('currently not loading a model and not training, WHAT are you doing?')
        elif do_train:
            separable, separable_model = separable_hnn(num_train_xCoords,
                                                       input_h_s=separable, input_model=separable_model,
                                                       save_path=save_path, epoch_save=epoch_save, dt=dt_train)

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
    q_traj, p_traj, H_traj = time_integrator_sv.integrate(x_coord_test, q_test, p_test,
                                                  t_span_test, method='SV')
    x_coord_test = x_coord_test.detach().cpu()
    t_span_test = t_span_test.detach().cpu()
    q_traj = q_traj.detach().cpu()
    p_traj = p_traj.detach().cpu()
    H_traj = H_traj.detach().cpu()

    fig, (ax, ax2, ax3) = plt.subplots(3, 1, gridspec_kw={'height_ratios': [1, 1, 2]}, figsize=(5, 8))

    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax2.set_xlim(0, t_span_test[-1])
    ax2.set_ylim(-10, 10)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('Hamiltonian [J]')
    ax2.set_xlabel('q')
    ax2.set_ylabel('p [m/s]')
    line, = ax.plot([], [], label='q', lw=3)
    line2, = ax.plot([], [], label='p', lw=3, color='r')
    line3, = ax2.plot([], [], lw=3, color='k')
    line4, = ax3.plot([], [], lw=1, color='g')
    ax.legend()
    fig.tight_layout(pad=1.0)

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        return line,


    def animate(i):
        line.set_data(x_coord_test[0], q_traj[0, :, i])
        line2.set_data(x_coord_test[0], p_traj[0, :, i])
        line3.set_data(t_span_test[0:i], H_traj[0:i])
        line4.set_data(q_traj[0, 5, 0:i], p_traj[0, 5, 0:i])
        return line, line2, line3


    anim = FuncAnimation(fig, animate, frames=len(t_span_test), init_func=init, blit=True)
    #plt.show()
    writer = LoopingPillowWriter(fps=20)
    # TODO change name for saved animation
    anim.save('tanh_1d_wave_V6.gif', writer=writer)



