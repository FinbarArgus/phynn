import torch
import torch.nn as nn
import torch.utils.data as data
import numpy as np

import pytorch_lightning as pl
from pytorch_lightning.trainer import training_loop
from pytorch_lightning.loggers import TensorBoardLogger
from matplotlib.animation import PillowWriter

import time
import os

from src.maths.dennet import DENNet
from src.mechanics.hamiltonian import HNN1DWaveSeparable
from src.time_integrator_1DWave import TimeIntegrator
from src.dense_net import SimpleEncoder
from src.dense_net import DenseNet

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class LoopingPillowWriter(PillowWriter):
    def finish(self):
        self._frames[0].save(
            self._outfile, save_all=True, append_images=self._frames[1:],
            duration=int(1000 / self.fps), loop=0)


class Learner(pl.LightningModule):
    def __init__(self, model: nn.Module, train_wHmodel=True, num_boundary=1,
                 save_path='temp_save_path', epoch_save=100, dt=0.01):
        super().__init__()
        self.model = model
        self.c = 0
        self.eps = 1e-6
        self.train_wHmodel= train_wHmodel
        self.num_boundary = num_boundary
        self.clockTime = time.process_time()
        self.save_path = save_path
        self.epoch_save = epoch_save
        self.dt=dt
        self.weight_bias_loss = 0
        self.second_opt = None
        self.scheduler = None

    def forward(self, x):
        return self.model.de_function(0, x)

    def backward(self, use_amp, loss, optimizer, optimizer_idx):
        loss.backward(retain_graph=True)
        return

    @staticmethod
    def loss(y, y_hat):
        return ((y - y_hat) ** 2).mean()

    @staticmethod
    def calculate_f(x, dq_dx, dp_dx):
        # this function calculates f = (q_dot, p_dot) = (dp_dx, -dq_dx)
        return -dp_dx, -dq_dx

    @staticmethod
    def add_gaussian_noise(input, mean, stddev):
        noise = input.data.new(input.size()).normal_(mean, stddev)
        return input + noise

    def calculate_weight_loss(self):
        """
        :return: returns a list of lists of related params from all tasks specific networks
        """
        weight_bias_loss = 0.
        for param, paramH in zip(self.model.de_function.func.parameters(),
                                 self.model.de_function.funcH.parameters()):
            weight_bias_loss += ((paramH - param)**2).sum()

        return weight_bias_loss

    def on_batch_end(self):
        """Here we calculate the L2 difference between weights and biases of the
        two networks. Then update them accordingly.
        This is soft parameter sharing
        """
        # TODO temporarily don't do this
        return
#        self.weight_bias_loss = self.calculate_weight_loss()

        # backward prop to get gradients on weights for the parameter sharing
#        self.weight_bias_loss.backward(retain_graph=True)
        # step the weights forward according to the gradients from the above back prop
#        self.second_opt.step()
#        self.second_opt.zero_grad()


    def training_step(self, batch, batch_idx):
        batch_size = len(batch[0])

        # Note: training with gradients slows down training to a crawl
        train_wgradApprox = False
        train_wH = False
        train_wH_exact = True

        if train_wH and self.train_wHmodel:
            print('cant have train_wH and train_wHmodel set to True, quitting')
            quit()
        if train_wH and train_wH_exact:
            print('cant have train_wH and train_wH_exact set to True, quitting')
            quit()

        # get input data and add noise
        x = batch[0]
        q = self.add_gaussian_noise(batch[1], 0.0, 0.01)
        p = self.add_gaussian_noise(batch[2], 0.0, 0.01)
        dq_dx = self.add_gaussian_noise(batch[3], 0.0, 0.01)
        dp_dx =  self.add_gaussian_noise(batch[4], 0.0, 0.01)
        if batch_size > 4:
            H_exact = self.add_gaussian_noise(batch[5], 0.0, 0.01)


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

        loss_main = self.loss(y_hat, y)
        # add the spatial gradients to the loss function to make them smooth
        loss_grads = torch.tensor(0.).to(x)
        loss_H_const = torch.tensor(0.).to(x)
        # loss_nonZero_H = torch.tensor(0.).to(x)
        grad_scale = 0.000005
        H_train_scale = 1.0  # 0.1/self.dt
        H_train_scale2 = 1.0

        if train_wgradApprox:
            q_dot_diff_approx = ((q_dot_hat[:, self.num_boundary + 1:-self.num_boundary] -
                                 q_dot_hat[:, self.num_boundary:-self.num_boundary-1]) / \
                                (x[:, self.num_boundary + 1:-self.num_boundary] -
                                 x[:, self.num_boundary:-self.num_boundary-1] + self.eps))**2
            p_dot_diff_approx = ((p_dot_hat[:, self.num_boundary + 1:-self.num_boundary] -
                                           p_dot_hat[:, self.num_boundary:-self.num_boundary-1]) / \
                                          (x[:, self.num_boundary + 1:-self.num_boundary] -
                                           x[:, self.num_boundary:-self.num_boundary-1] + self.eps))**2
            loss_grads += grad_scale*(q_dot_diff_approx.mean() + p_dot_diff_approx.mean())
        if train_wH:
            time_integrator = TimeIntegrator(self.model.de_function.func).to(device)
            # TODO should i include noise here?
            q_new, p_new, H_old = time_integrator.sv_step(x, q, p, self.dt)
            H_new = self.model.de_function.func.H(torch.cat([x, q_new, p_new], dim=1))
            loss_H_const += H_train_scale*((H_new - H_old)**2).mean()

        if self.train_wHmodel:
            time_integrator = TimeIntegrator(self.model.de_function.funcH).to(device)
            # TODO should i include noise here?
            q_new, p_new, H_old = time_integrator.sv_step(x, q, p, self.dt)
            H_new = self.model.de_function.funcH.H(torch.cat([x, q_new, p_new], dim=1))

            loss_H_const += H_train_scale*((H_new[:, 0] - H_exact)**2).mean()
            if train_wH_exact:
                loss_H_const += H_train_scale2 * ((H_exact - H_old[:, 0])**2).mean()
        elif train_wH_exact:
            H_old = self.model.de_function.funcH.H(torch.cat([x, q, p], dim=1))
            loss_H_const += H_train_scale2*((H_exact - H_old[:, 0])**2).mean()


        loss = loss_H_const # loss_main + loss_grads + loss_H_const # + loss_nonZero_H
        logs = {'train_loss': loss, 'loss_main': loss_main,
                'loss_grads': loss_grads, 'loss_H':loss_H_const,
                'loss_weights': self.weight_bias_loss, 'H learning rate': self.second_opt.param_groups[0]['lr']}

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
        adaptive_lr = False
        # set up lr scheduler on first epoch
        if self.current_epoch == 0 and batch_idx == 0:
            lmda_lr = lambda epoch: 2.0
            self.scheduler = torch.optim.lr_scheduler.MultiplicativeLR(self.second_opt, lr_lambda=lmda_lr)
        # If epoch is a multiple of 300, increase the learning rate for H conservation
        if adaptive_lr and self.current_epoch != 0 and \
                self.current_epoch % 100 == 0 and batch_idx == 0 and self.second_opt.param_groups[0]['lr'] < 0.1:
            self.scheduler.step()

        return {'loss': loss, 'log': logs}

    def configure_optimizers(self):
        self.second_opt = torch.optim.SGD(self.model.parameters(), lr=0.00001) #lr=0.005)
        return torch.optim.Adam(self.model.parameters(), lr=0.0005, weight_decay=0.01)

    @staticmethod
    def train_dataloader():
        return trainloader


def separable_hnn(num_points, train_wHmodel=True, input_h_s=None, input_model=None,
                  save_path='temp_save_path', train=True, epoch_save=100, dt=0.01):
    """
    Separable Hamiltonian network.

    :return:
    """
    if input_h_s:
        h_s = input_h_s
        model = input_model
    else:
        # architecture = SimpleEncoder([3*num_points, 60, 60, 40, 20, 10, 1],
        #                              drop_rate=0.2, batch_norm=False).to(device)
        architecture = DenseNet(3*num_points, 1, [8], [40], drop_rate=0.0, batch_norm=True).to(device)
        h_s = HNN1DWaveSeparable(architecture)

        # initialise weights
        # if train:
        for m in h_s.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        model = DENNet(h_s, case='1DWave').to(device)


    if train:
        learn_sep = Learner(model, train_wHmodel=train_wHmodel, num_boundary=num_boundary,
                            save_path=save_path, epoch_save=epoch_save, dt=dt)
        logger = TensorBoardLogger('separable_logs')
        trainer_sep = pl.Trainer(min_epochs=5, max_epochs=5, logger=logger, gpus=1)
        # step the weights forward according to the gradients from the above back prop
        trainer_sep.fit(learn_sep)

    return h_s, model


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using cuda device: {}'.format(torch.cuda.get_device_name(torch.cuda.current_device())))

    # bool to determine if model is loaded
    load_model = True
    do_train = False
    # whether we want to train with a second Hamiltonian model
    train_wHmodel = False

    #save path
    save_dir = 'saved_models'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    model_name = 'model_denseNet.pt'
    save_path = os.path.join(save_dir, model_name)
    # save every epoch_save epochs
    epoch_save = 50

    # Training conditions
    num_train_samples = 1
    num_train_xCoords = 20
    num_tSteps_training = 1
    num_boundary = 2
    num_datasets = 2**17
    batch_size = 2048
    # Training initial conditions [q, p, dq/dx, dp/dx, x]
    x_coord = torch.zeros(num_datasets, num_train_xCoords).to(device)
    q = torch.zeros(num_datasets, num_train_xCoords).to(device)
    p = torch.zeros(num_datasets, num_train_xCoords).to(device)
    dq_dx = torch.zeros(num_datasets, num_train_xCoords).to(device)
    dp_dx = torch.zeros(num_datasets, num_train_xCoords).to(device)
    H_exact = torch.zeros(num_datasets).to(device)
    H_normalise = 0.1

    if do_train:
        x_coord[:, num_boundary:-1 * num_boundary] = \
            torch.rand(num_datasets, num_train_xCoords - num_boundary * 2).sort()[0]
        x_coord[:, -1 * num_boundary:] = 1.0
        for batch_idx in range(num_datasets):
            amplitude_q = 2*np.random.rand(1)[0]-1
            amplitude_p = 2*np.random.rand(1)[0]-1
            q[batch_idx, :] = amplitude_q*np.pi * torch.cos(np.pi * x_coord[batch_idx, :]).to(device)
            p[batch_idx, :] = amplitude_p*np.pi * torch.sin(np.pi * x_coord[batch_idx, :]).to(device)
            dq_dx[batch_idx, :] = -amplitude_q*np.pi ** 2 * torch.sin(np.pi * x_coord[batch_idx, :]).to(device)
            dp_dx[batch_idx, :] = amplitude_p*np.pi ** 2 * torch.cos(np.pi * x_coord[batch_idx, :]).to(device)
            # This is the exact hamiltonian calculated by hand
            # In future will have to calculate a general hamiltonian for different frequencies
            H_exact[batch_idx] = H_normalise*0.5*(amplitude_q**2*np.pi**2 + amplitude_p**2*np.pi**2)


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
    num_test = 10
    # temporarily test with the same as one of the training init conditions
    # x_coord_test = x_coord[0:1, :]
    x_coord_test = torch.zeros(num_test, num_train_xCoords).to(device)
    x_coord_test[:, num_boundary:-1*num_boundary] = torch.rand(num_test, num_train_xCoords - num_boundary*2).sort()[0]
    # overwrite first entry to be uniform
    x_coord_test[:1, num_boundary:-1*num_boundary] = torch.linspace(0, 1.0, num_train_xCoords-num_boundary*2)
    # set right boundary to 1
    x_coord_test[:, -1*num_boundary:] = 1.0
    q_test = torch.zeros(num_test, num_train_xCoords).to(device)
    p_test = torch.zeros(num_test, num_train_xCoords).to(device)
    H_exact_test = torch.zeros(num_test)

    # first entry is uniform with 0.5 amplitude
    amplitude_q_test = 0.5
    amplitude_p_test = 0.5
    for test_idx in range(num_test):
        q_test[test_idx, :] = amplitude_q_test*np.pi * torch.cos(np.pi * x_coord_test[test_idx, :]).to(device)
        p_test[test_idx, :] = amplitude_p_test*np.pi * torch.sin(np.pi * x_coord_test[test_idx, :]).to(device)
        # This is the exact hamiltonian calculated by hand
        # In future will have to calculate a general hamiltonian for different frequencies
        H_exact_test[test_idx] = H_normalise*0.5*(amplitude_q_test**2*np.pi**2 + amplitude_p_test**2*np.pi**2)

        amplitude_q_test = 2*np.random.rand(1)[0]-1
        amplitude_p_test = 2*np.random.rand(1)[0]-1

    t_span_test = torch.linspace(0, 0.3, 60).to(device)


    # Wrap in for loop and change inputs to the stepped forward p's and q's
    for tStep in range(num_tSteps_training):

        if do_train:
            train = data.TensorDataset(x_coord, q, p, dq_dx, dp_dx, H_exact)
            trainloader = data.DataLoader(train, batch_size=batch_size, shuffle=True)

        # hamiltonian, basic_model = basic_hnn()
        # load model if required
        if tStep == 0:
            if load_model:
                separable, separable_model = separable_hnn(num_train_xCoords, train_wHmodel=train_wHmodel,
                                                           train=False)
                # optimizer = TheOptimizerClass(*args, **kwargs)

                checkpoint = torch.load(save_path)
                separable_model.load_state_dict(checkpoint['model_state_dict'])
                separable_model.eval()
                separable = separable_model.de_function.func
                separable_H = separable_model.de_function.funcH
                # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']

                print('loaded model at epoch {} from {}'.format(epoch, save_path))

                if do_train:
                    _, separable_model = separable_hnn(num_train_xCoords, train_wHmodel=train_wHmodel,
                                                               input_h_s=separable, input_model=separable_model,
                                                               save_path=save_path, epoch_save=epoch_save, dt=dt_train)
                    separable_model.eval()
                    separable = separable_model.de_function.func
                    separable_H = separable_model.de_function.funcH

            else:
                if do_train:
                    _, separable_model = separable_hnn(num_train_xCoords, train_wHmodel=train_wHmodel,
                                                               save_path=save_path, epoch_save=epoch_save, dt=dt_train)
                    separable_model.eval()
                    # TODO do i need to make sure separable is eval()
                    separable = separable_model.de_function.func
                    separable_H = separable_model.de_function.funcH
                else:
                    print('currently not loading a model and not training, WHAT are you doing?')
        else:
            print('currently not working for multiple time steps')
            exit()


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

    print('Hamiltonian comparison')
    fig, ax = plt.subplots(1, 1)
    # ax.set_ylim(0, 5)
    ax.set_xlabel('test number')
    ax.set_ylabel('Hamiltonian [J]')

    #H_test = torch.zeros(num_test).to(device)
    #H_test = separable_model.de_function.func.H(torch.cat([x_coord_test, q_test, p_test], dim=1))
    #H_test = H_test.detach().cpu()
    H_test_H = torch.zeros(num_test).to(device)
    H_test_H = separable_H.H(torch.cat([x_coord_test, q_test, p_test], dim=1))
    H_test_H = H_test_H.detach().cpu() # / H_normalise
    # H_exact_test = H_exact_test.detach().cpu() # / H_normalise
    #plt.plot(H_test, marker='x', linestyle='', color='r', label='H_test_an')
    plt.plot(H_test_H, marker='x', linestyle='', color='b', label='H_test')
    plt.plot(H_exact_test, marker='x', linestyle='', color='r', label='H_test_exact')
    ax.legend()
    plt.savefig('H_comparison.png')

    # set up time integrator that uses our HNN
    # time_integrator_euler = TimeIntegrator(hamiltonian).to(device)
    # calculate trajectory with an euler step
    # traj_HNN_Euler = time_integrator_euler.integrate(X_test, t_span_test, method='Euler').detach().cpu()
    plot_wHmodel = True

    if not plot_wHmodel:
        # set up time integrator that uses our separable HNN
        time_integrator_sv = TimeIntegrator(separable).to(device)
        # calculate trajectory
        q_traj, p_traj, H_traj = time_integrator_sv.integrate(x_coord_test[:1], q_test[:1], p_test[:1],
                                                      t_span_test, method='SV')
        x_coord_test = x_coord_test.detach().cpu()
        t_span_test = t_span_test.detach().cpu()
        q_traj = q_traj.detach().cpu()
        p_traj = p_traj.detach().cpu()
        H_traj = H_traj.detach().cpu()

    else:
        # set up time integrator that uses our separable HNN for the Hamiltonian conservation NN
        time_integrator_sv = TimeIntegrator(separable_H).to(device)
        # calculate trajectory
        q_traj, p_traj, H_traj = time_integrator_sv.integrate(x_coord_test[:1], q_test[:1], p_test[:1],
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
    ax2.set_ylim(0, 1)
    ax3.set_xlim(-2, 2)
    ax3.set_ylim(-2, 2)
    ax.set_xlabel('x [m]')
    ax2.set_xlabel('t [s]')
    ax2.set_ylabel('Hamiltonian [J]')
    ax3.set_xlabel('q')
    ax3.set_ylabel('p [m/s]')
    line, = ax.plot([], [], label='q', lw=3)
    line2, = ax.plot([], [], label='p', lw=3, color='r')
    line3, = ax2.plot([], [], lw=1, color='g', label='Test')
    line4, = ax2.plot([], [], lw=1, color='k', label='Exact')
    line5, = ax3.plot([], [], lw=1, color='g')
    ax.legend()
    ax2.legend()
    fig.tight_layout(pad=1.0)

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])
        return line, line2, line3, line4, line5


    def animate(i):
        line.set_data(x_coord_test[0], q_traj[0, :, i])
        line2.set_data(x_coord_test[0], p_traj[0, :, i])
        line3.set_data(t_span_test[0:i], H_traj[0:i])
        line4.set_data(t_span_test, H_exact_test[0]*np.ones(len(t_span_test)))
        line5.set_data(q_traj[0, 5, 0:i], p_traj[0, 5, 0:i])
        return line, line2, line3, line4, line5


    anim = FuncAnimation(fig, animate, frames=len(t_span_test), init_func=init, blit=True)
    #plt.show()
    writer = LoopingPillowWriter(fps=20)
    # TODO change name for saved animation
    anim.save('tanh_1d_wave_V6.gif', writer=writer)



