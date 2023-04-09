import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

from collections import namedtuple
from typing import Tuple
from E3_MC_fix_control import value_fix




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

sigma = torch.diag(torch.tensor([0.05, 0.05]))
sig = torch.mm(sigma, sigma.T)

def get_laplacian(grad, x):
    v1 = grad[:,0].view(-1,1)
    grad21 = \
        torch.autograd.grad(v1, x, grad_outputs=torch.ones_like(v1), only_inputs=True, create_graph=True, retain_graph=True)[0]
    v2 = grad[:,1].view(-1,1)
    grad22 = \
        torch.autograd.grad(v2, x, grad_outputs=torch.ones_like(v2), only_inputs=True, create_graph=True, retain_graph=True)[0]
    grad21 = grad21.unsqueeze(2)
    grad22 = grad22.unsqueeze(2)
    tr_hess = torch.matmul(sig.double(), torch.cat((grad21, grad22), 2).double()).diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
    return tr_hess.unsqueeze(1)

# def get_laplacian(grad, x):
#     hess_diag = []
#     for d in range(x.shape[1]):
#         v = grad[:,d].view(-1,1)
#         grad2 = torch.autograd.grad(v,x,grad_outputs=torch.ones_like(v), only_inputs=True, create_graph=True, retain_graph=True)[0]
#         hess_diag.append(grad2[:,d].view(-1,1))
#     hess_diag = torch.cat(hess_diag, 1)
#     laplacian = hess_diag.sum(1, keepdim=True)
#     return laplacian


max_updates = 800
Net = Net_DGM(2, 100, activation = 'LogSigmoid')

optimizer = torch.optim.Adam(Net.parameters(), lr=0.002)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(10000,), gamma=0.1)

loss_fn = nn.MSELoss()

H = torch.eye(2).double()
M = torch.eye(2).double()
R = torch.eye(2).double()
C = 0.8*torch.eye(2).double()
D = 0.1*torch.eye(2).double()
T = 1
sigma = torch.diag(torch.tensor([0.05, 0.05]))
#LQR1 = SolveLQR(H, M, Sigma, C, D, R, T)
Sig = torch.mm(sigma, sigma.T)
tr = Sig.trace()

batch_size = 10000
running_loss = []

step = 0
error_standard = []
error_MC = []

sum_loss = 0
sum_error = 0
episdoe = []
loss_eps = []
error_eps = []

for it in range(max_updates):
    step += 1
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

    # take value at t,x
    t0 = torch.zeros(1, 1, requires_grad=False).double()
    x0 = torch.ones(1, 2, requires_grad=False).double()
    value_dgm = Net(t0.detach(), x0.detach()).float()
    error = abs((value_dgm - value_fix)/value_fix).squeeze().detach().numpy()

    running_loss.append(loss.item())#per iteration
    sum_loss += loss.item()
    sum_error += error

    if (it + 1) % 10 == 0:  # print every 10 iterations
        mean_loss = sum_loss / 10
        mean_error = sum_error / 10
        episdoe.append(it)
        loss_eps.append(mean_loss) #per 10 iterations
        error_eps.append(mean_error) #per 10 iterations
        print('[%5d] Training loss: %.9f' % (it + 1, mean_loss))
        sum_loss = 0.0
        sum_error = 0.0

fig, ax = plt.subplots(1, 3)
time_list = np.arange(0, 800, 1)
ax[0].plot(time_list, running_loss)
ax[0].set_title('loss per iteration')
ax[1].plot(episdoe, loss_eps)
ax[1].set_title('loss per 10 iterations')
ax[2].plot(episdoe, error_eps)
ax[2].set_title('error against MC per 10 iterations')

plt.show()


# time_list = np.arange(0, 2000, 1)
# plt.plot(time_list, running_loss)
# plt.show()