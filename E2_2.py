import torch
import torch.nn as nn
import numpy as np
from E1_1 import SolveLQR

from collections import namedtuple
from typing import Tuple


class FFN(nn.Module):

    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False):
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




max_updates = 1000
Net = FFN([3, 100, 2])

optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)

loss_fn = nn.MSELoss()

H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.1*np.identity(2)
D = 0.1*np.identity(2)
T = 1
Sigma = np.diag([0.05, 0.05])
LQR1 = SolveLQR(H, M, Sigma, C, D, R, T)

batch_size = 1000
running_loss = 0
for it in range(max_updates):
    optimizer.zero_grad()

    input_domain = (torch.rand(batch_size, 1, 2) - 0.5)*6
    t = torch.from_numpy(np.linspace(0, 1, batch_size))
    target_functional = LQR1.get_controller(t, input_domain).double()

    t_net = t.unsqueeze(1)
    input_domain_net = input_domain.squeeze()
    u_of_tx = Net(t_net, input_domain_net)
    loss = loss_fn(u_of_tx, target_functional)

    loss.backward()
    optimizer.step()
    scheduler.step()
    running_loss += loss.item()

    if it % 10 == 0:  # print every 2000 mini-batches
        print('[%5d] loss: %.6f' %
              (it + 1, running_loss / 2000))
        running_loss = 0.0
