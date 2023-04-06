import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
from Networks import FFN, Net_DGM, DGM_Layer

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


class AgentDP:

    def __init__(self, batch_size, H, M, C, D, sigma, T,
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
        self.sigma = sigma
        self.tr = (self.sigma * self.sigma.T).trace()
        self.T = T

        self.optim_critic = Adam(self.critic_net.parameters(), lr = learning_rate_critic)
        self.optim_policy = Adam(self.policy_net.parameters(), lr = learning_rate_policy)

        self.loss_fn = torch.nn.MSELoss()

    def policy_improvement(self, t, x):
        '''
        :param t: one episode of t with batch size
        :param x: one episode of x with batch size
        :return: loss of critic network
        '''

        u_of_tx = self.critic_net(t, x)
        grad_u_x = get_gradient(u_of_tx, x)
        alpha = self.policy_net(t, x)

        self.optim_policy.zero_grad()

        H_loss = (torch.matmul(torch.matmul(x, self.H.T), grad_u_x.T)
                 + torch.matmul(torch.matmul(alpha, self.M.T), grad_u_x.T)
                 + torch.matmul(torch.matmul(x, self.C), x.T)
                 + torch.matmul(torch.matmul(alpha, self.D), alpha.T)).squeeze(1)

        H_loss /= self.batch_size

        H_loss.backward(retain_graph=True)
        self.optim_policy.step()

        return {'Hamiltonian loss:': H_loss.item()}

    def policy_evaluate(self, t, x):
        '''
        :param t: one episode of t with batch size
        :param x: one episode of x with batch size
        :return: loss of policy network
        '''

        alpha = self.policy_net(t, x)

        u_of_tx = self.critic_net(t, x)
        grad_u_x = get_gradient(u_of_tx, x)
        grad_u_t = get_gradient(u_of_tx, t)
        laplacian = get_laplacian(grad_u_x, x)

        target_functional = torch.zeros_like(u_of_tx)

        self.optim_critic.zero_grad()
        pde = grad_u_t + 0.5 * self.tr * laplacian \
              + (torch.matmul(torch.matmul(x, self.H.T), grad_u_x.T)
                 + torch.matmul(torch.matmul(alpha, self.M.T), grad_u_x.T)
                 + torch.matmul(torch.matmul(x, self.C), x.T)
                 + torch.matmul(torch.matmul(alpha, self.D), alpha.T)).squeeze(1)

        MSE_functional = self.loss_fn(pde, target_functional)

        input_terminal = x
        t = torch.ones(self.batch_size, 1) * self.T

        u_of_tx = self.critic_net(t, input_terminal)
        target_terminal = torch.matmul(torch.matmul(x.detach().unsqueeze(1), self.C),
                                       x.detach().unsqueeze(2)).squeeze()
        MSE_terminal = self.loss_fn(u_of_tx, target_terminal)

        B_loss = MSE_functional + MSE_terminal
        B_loss.backward(retain_graph=True)
        self.optim_critic.step()

        return {'Bellman Loss:': B_loss.item()}

H = torch.eye(2).double()
M = torch.eye(2).double()
R = torch.eye(2).double()
C = torch.eye(2).double()* 0.1
D = torch.eye(2).double() * 0.1
T = 1
sigma = torch.diag(torch.tensor([0.5, 0.5]))


# Initialize the dynamic programming agent
agent = AgentDP(1, H, M, C, D, sigma, T, 0.001, 0.001)

# Example of usage of update function
t = torch.rand(1, 1, requires_grad=True).double()
input_domain = (torch.rand(1, 2, requires_grad=True) - 0.5)*6
input_domain = input_domain.double()

def train_DP(Agent, criteria, max_steps):
    '''
    :param Agent:
    :param criteria:
    :param max_steps:
    :return:
    '''
    error = criteria * 1e6
    t = torch.rand(1, 1, requires_grad=True).double()
    x = (torch.rand(1, 2, requires_grad=True) - 0.5) * 6
    x = x.double()
    step = 1

    while (error > criteria) & (step < max_steps):

        value_last = Agent.critic_net(t, x)
        Agent.policy_evaluate(t, x)
        Agent.policy_improvement(t, x)
        value_improved = Agent.critic_net(t, x)
        error = torch.abs(value_improved - value_last)
        print({f'Step {step}': error})
        step += 1

    return t, x, value_improved.item()

print(train_DP(agent, 1e-4, 100))