import torch
import torch.nn as nn
import numpy as np
from E1_1 import SolveLQR

from collections import namedtuple
from typing import Tuple





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
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

def get_laplacian(grad, x):
    hess_diag = []
    for d in range(x.shape[1]):
        v = grad[:,d].view(-1,1)
        grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
        hess_diag.append(grad2[:,d].view(-1,1))
    hess_diag = torch.cat(hess_diag,1)
    laplacian = hess_diag.sum(1, keepdim=True)
    return laplacian


max_updates = 10
Net = Net_DGM(2, 100, activation='Tanh')

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
H = torch.from_numpy(H)
M = torch.from_numpy(M)
C = torch.from_numpy(C)
D = torch.from_numpy(D)
Sig = Sigma * Sigma.T
tr = Sig.trace()

batch_size = 1000
running_loss = 0

input_domain = (torch.rand(batch_size, 2, requires_grad=True) - 0.5)*6
input_domain = input_domain.double()
alpha = torch.ones_like(input_domain)
print(torch.matmul(alpha.unsqueeze(1), M).T)


# for it in range(max_updates):
#     optimizer.zero_grad()
#
#     input_domain = (torch.rand(batch_size, 2, requires_grad=True) - 0.5)*6
#     input_domain = input_domain.double()
#     t = torch.rand(batch_size, 1, requires_grad=True).double()
#     alpha = torch.ones_like(input_domain)
#
#     u_of_tx = Net(t, input_domain)
#     grad_u_x = get_gradient(u_of_tx, input_domain)
#     grad_u_t = get_gradient(u_of_tx, t)
#     laplacian = get_laplacian(grad_u_x, input_domain)
#
#     target_functional = torch.zeros_like(u_of_tx)
#
#
#     pde = grad_u_t + 0.5 * tr * laplacian \
#           + (torch.matmul(torch.matmul(input_domain.detach().unsqueeze(1), H),grad_u_x.unsqueeze(2))\
#           + torch.matmul(torch.matmul(alpha.unsqueeze(1),M),grad_u_x.unsqueeze(2))
#           + torch.matmul(torch.matmul(input_domain.detach().unsqueeze(1),C),input_domain.detach().unsqueeze(2))\
#           + torch.matmul(torch.matmul(alpha.unsqueeze(1),D ),alpha.unsqueeze(2))).squeeze(1)
#
#     MSE_functional = loss_fn(pde, target_functional)
#
#     input_terminal = input_domain
#     #？？？？需要detach嘛
#     t = torch.ones(batch_size, 1) * T
#
#     u_of_tx = Net(t, input_terminal)
#     target_terminal = torch.matmul(torch.matmul(input_domain.detach().unsqueeze(1),C),input_domain.detach().unsqueeze(2)).squeeze()
#     MSE_terminal = loss_fn(u_of_tx, target_terminal)
#
#     loss = MSE_functional + MSE_terminal
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
