import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from E1_1 import SolveLQR
from collections import namedtuple
from typing import Tuple


class FFN(nn.Module):

    def __init__(self, sizes, activation = nn.ReLU, output_activation=nn.Identity, batch_norm=False):
        super().__init__()

        layers = [nn.BatchNorm1d(sizes[0]), ] if batch_norm else []
        for j in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[j], sizes[j + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j + 1], affine=True))
            if j < (len(sizes) - 2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)
        self.double()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, t, x):
        t_x = torch.cat([t, x], 1)
        return self.net(t_x)


def train_policy(max_updates,
                 layer_size,
                 learning_rate=0.001,
                 milestones=(10000,),
                 gamma=0.1,
                 batch_size=1000,
                 loss_fn=nn.MSELoss(),
                 **kwags
                 ):

    Net = FFN(layer_size)

    optimizer = torch.optim.Adam(Net.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma)

    H = kwags.get('H')
    M = kwags.get('M')
    C = kwags.get('C')
    D = kwags.get('D')
    R = kwags.get('R')
    Sigma = kwags.get('Sigma')
    T_grid = kwags.get('T_grid')
    LQR1 = SolveLQR(H, M, C, D, R, Sigma, T_grid)

    running_loss = 0
    episdoe = []
    loss_eps = []

    for it in range(max_updates):
        optimizer.zero_grad()

        input_domain = (torch.rand(batch_size, 1, 2) - 0.5)*6
        t = torch.rand(batch_size).double()
        target_functional = LQR1.get_controller(t, input_domain).squeeze().double()

        t_net = t.unsqueeze(1)
        input_domain_net = input_domain.squeeze()
        u_of_tx = Net(t_net, input_domain_net)
        loss = loss_fn(u_of_tx, target_functional)

        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        if (it + 1) % (max_updates // 20) == 0:  # print every 5%
            mean_loss = running_loss / (max_updates // 20)
            episdoe.append(it)
            loss_eps.append(mean_loss)
            print('[%5d] Training loss: %.9f' % (it + 1, mean_loss))

            running_loss = 0.0

    if kwags['visualize']:
        plt.plot(episdoe, loss_eps)
        plt.xlabel("Iterations", fontsize = 18)
        plt.ylabel("Averaged Loss", fontsize = 18)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.title('Supervised learning of Markov control by FFN', fontsize = 20)
        plt.tight_layout(pad = 0.3)

        plt.show()

kwags = {
    'H': np.identity(2),
    'M': np.identity(2),
    'C': 0.8*np.identity(2),
    'D': 0.1*np.identity(2),
    'R': np.identity(2),
    'Sigma': np.diag([0.05, 0.05]),
    'T_grid': np.linspace(0,1,1000),
    'visualize': True,
}

train_policy(max_updates = 120,
             layer_size = [3, 100, 100, 2],
             activation = 'Tanh',
             learning_rate = 0.001,
             milestones = (10000,),
             gamma=0.1,
             batch_size = 1000,
             loss_fn = nn.MSELoss(reduction = 'mean'),
             **kwags)