import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

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


max_updates = 1000
Net = Net_DGM(2, 100, activation = 'Tanh')

optimizer = torch.optim.Adam(Net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)

loss_fn = nn.MSELoss()

H = torch.eye(2).double()
M = torch.eye(2).double()
R = torch.eye(2).double()
C = torch.eye(2).double()* 0.1
D = torch.eye(2).double() * 0.1
T = 1
sigma = torch.diag(torch.tensor([0.5, 0.5]))
#LQR1 = SolveLQR(H, M, Sigma, C, D, R, T)
Sig = sigma * sigma.T
tr = Sig.trace()

batch_size = 1
running_loss = []

input_domain = (torch.rand(batch_size, 2, requires_grad=True) - 0.5)*6
input_domain = input_domain.double()
alpha = torch.ones_like(input_domain).double()

step = 0
error = []
for it in range(max_updates):
    optimizer.zero_grad()

    input_domain = (torch.rand(batch_size, 2, requires_grad=True) - 0.5)*6
    input_domain = input_domain.double()
    t = torch.rand(batch_size, 1, requires_grad=True).double()
    alpha = torch.ones_like(input_domain)

    u_of_tx = Net(t, input_domain)
    grad_u_x = get_gradient(u_of_tx, input_domain)
    grad_u_t = get_gradient(u_of_tx, t)
    laplacian = get_laplacian(grad_u_x, input_domain)

    target_functional = torch.zeros_like(u_of_tx)


    pde = grad_u_t + 0.5 * tr * laplacian \
              + (torch.mul(torch.matmul(input_domain, H.T), grad_u_x)
                 + torch.mul(torch.matmul(alpha, M.T), grad_u_x)
                 + torch.mul(torch.matmul(input_domain, C), input_domain)
                 + torch.mul(torch.matmul(alpha, D), alpha)
                 ).sum(dim = 1).unsqueeze(1)


    MSE_functional = loss_fn(pde, target_functional)

    input_terminal = input_domain
    t = torch.ones(batch_size, 1) * T

    u_of_tx = Net(t, input_terminal)
    target_terminal = torch.mul(torch.matmul(input_terminal, R), input_terminal).sum(dim = 1).unsqueeze(1)
    MSE_terminal = loss_fn(u_of_tx, target_terminal)

    loss = MSE_functional + MSE_terminal
    loss.backward()
    optimizer.step()
    scheduler.step()

    running_loss.append(loss.item())
    # Example: value function at Terminal time 1, Value of x [2, 2] with identity matrix R
    standard = torch.matmul(torch.tensor([2, 2]), torch.tensor([2, 2]))
    training = Net(torch.tensor([[1]]).double(), torch.tensor([[2, 2]]).double())
    error.append(torch.abs(training - standard).item())


print(Net(torch.tensor([[1]]).double(), torch.tensor([[2, 2]]).double()))
fig, ax = plt.subplots(1, 2)
time_list = np.arange(0, 1000, 1)
ax[0].plot(time_list, running_loss)
ax[1].plot(time_list, error)
plt.show()
