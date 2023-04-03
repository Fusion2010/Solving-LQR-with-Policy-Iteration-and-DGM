"""
Deep Galerkin Method: https://arxiv.org/abs/1708.07469
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from E1_1 import SolveLQR



class DGM_Layer(nn.Module):

    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(DGM_Layer, self).__init__()

        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.gate_Z = self.layer(dim_x + dim_S, dim_S)
        self.gate_G = self.layer(dim_x + dim_S, dim_S)
        self.gate_R = self.layer(dim_x + dim_S, dim_S)
        self.gate_H = self.layer(dim_x + dim_S, dim_S)
        self.double()

    def layer(self, nIn, nOut):
        l = nn.Sequential(nn.Linear(nIn, nOut), self.activation)
        return l

    def forward(self, x, S):
        x_S = torch.cat([x, S], 1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)

        input_gate_H = torch.cat([x, S * R], 1)
        H = self.gate_H(input_gate_H)

        output = ((1 - G)) * H + Z * S
        return output


class Net_DGM(nn.Module):

    def __init__(self, dim_x, dim_S, activation='Tanh'):
        super(Net_DGM, self).__init__()

        self.dim = dim_x
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'LogSigmoid':
            self.activation = nn.LogSigmoid()
        else:
            raise ValueError("Unknown activation function {}".format(activation))

        self.input_layer = nn.Sequential(nn.Linear(dim_x + 1, dim_S), self.activation)

        self.DGM1 = DGM_Layer(dim_x=dim_x + 1, dim_S=dim_S, activation=activation)
        self.DGM2 = DGM_Layer(dim_x=dim_x + 1, dim_S=dim_S, activation=activation)
        self.DGM3 = DGM_Layer(dim_x=dim_x + 1, dim_S=dim_S, activation=activation)

        self.output_layer = nn.Linear(dim_S, 1)
        self.double()

    def forward(self, t, x):
        tx = torch.cat([t, x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx, S1)
        S3 = self.DGM2(tx, S2)
        S4 = self.DGM3(tx, S3)
        output = self.output_layer(S4)
        return output

    def get_gradient(output, x):
        grad = \
        torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True,
                            only_inputs=True)[0]
        return grad

    def get_laplacian(grad, x):
        hess_diag = []
        for d in range(x.shape[1]):
            v = grad[:, d].view(-1, 1)
            grad2 = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True,
                                        retain_graph=True)[0]
            hess_diag.append(grad2[:, d].view(-1, 1))
        hess_diag = torch.cat(hess_diag, 1)
        laplacian = hess_diag.sum(1, keepdim=True)
        return laplacian

def train_NN(max_updates = 100000,
             hidden_size = 100,
             activation = 'Tanh',
             learning_rate = 0.001,
             milestones = (10000,),
             gamma=0.1,
             batch_size = 1000,
             loss_fn = nn.MSELoss(),
             **kwags):

    H = kwags.get('H')
    M = kwags.get('M')
    R = kwags.get('R')
    C = kwags.get('C')
    D = kwags.get('D')
    T = kwags.get('T')
    Sigma = kwags.get('Sigma')
    Net = Net_DGM(2, hidden_size, activation = activation)

    optimizer = torch.optim.Adam(Net.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma)

    LQR1 = SolveLQR(H, M, Sigma, C, D, R, T)

    running_loss = 0
    episdoe = []
    loss_eps = []

    for it in tqdm(range(max_updates)):
        optimizer.zero_grad()

        input_domain = (torch.rand(batch_size, 1, 2) - 0.5)*6
        t = torch.from_numpy(np.linspace(0, 1, batch_size))
        target_functional = LQR1.get_value(t, input_domain)

        t_net = t.unsqueeze(1)
        input_domain_net = input_domain.squeeze()
        u_of_tx = Net(t_net, input_domain_net)
        loss = loss_fn(u_of_tx, target_functional)

        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        if (it+1) % (max_updates//20) == 0:  # print every 5%
            mean_loss = running_loss / (max_updates//20)
            episdoe.append(it)
            loss_eps.append(mean_loss)
            print('[%5d] loss: %.9f' % (it+1, mean_loss))

            running_loss = 0.0

    if kwags['visualize']:
        plt.plot(episdoe, loss_eps)
        plt.xlabel("Timesteps", fontsize = 20)
        plt.ylabel("Averaged Loss", fontsize = 20)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.tight_layout(pad = 0.3)

        plt.show()

kwags = {
    'H': np.identity(2),
    'M': np.identity(2),
    'R': np.identity(2),
    'C': 0.1*np.identity(2),
    'D': 0.1*np.identity(2),
    'T': 1,
    'Sigma': np.diag([0.05, 0.05]),
    'visualize': True,
}


train_NN(max_updates = 1000,
         hidden_size = 100,
         activation = 'Tanh',
         learning_rate = 0.001,
         milestones = (10000,),
         gamma=0.1,
         batch_size = 1000,
         loss_fn = nn.MSELoss(reduction = 'mean'),
         **kwags)