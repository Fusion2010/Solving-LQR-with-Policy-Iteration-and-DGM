import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from E1_1 import SolveLQR
from E2_2 import FFN
from E2_1 import Net_DGM, DGM_Layer

class AgentDP:

    def __init__(self, batch_size, H, M, C, D,
                 learning_rate_critic,
                 learning_rate_policy):
        '''
        :param batch_size: sample size each iteration/improvement
        :param H: matrix H
        :param M: matrix M
        :param C: matrix C
        :param D: matrix D
        :param learning_rate_critic: learning rate for value function network
        :param learning_rate_policy: learning rate for policy function network
        '''
        self.policy_net = FFN([3, 100, 2])
        self.critic_net = Net_DGM(2, 100, activation = 'Tanh')

        self.batch_size = batch_size
        self.H = H
        self.M = M
        self.C = C
        self.D = D

        self.optim_critic = Adam(self.critic_net.parameters(), lr = learning_rate_critic)
        self.optim_policy = Adam(self.policy_net.parameters(), lr = learning_rate_policy)

    def Hamiltonian(self, t, x):
        '''
        :param t: one episode of t with batch size
        :param x: one episode of x with batch size
        :return: loss of critic network
        '''

        H = 0
        control = self.policy_net.forward(t, x)
        value = self.critic_net.forward(t, x)
        value.backward()
        value_partial = x.grad

        self.optim_critic.zero_grad()
        H += torch.matmul(value_partial.T, torch.matmul(self.H, x)) \
             + torch.matmul(value_partial.T, torch.matmul(self.M, control)) \
             + torch.matmul(x.T, torch.matmul(self.C, x)) \
             + torch.matmul(control.T, torch.matmul(self.D, control))

        H /= self.batch_size

        H.backward()
        self.optim_critic.step()

        return {'Hamiltonian loss: ', H}

