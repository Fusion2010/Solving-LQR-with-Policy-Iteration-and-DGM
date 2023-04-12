import torch
from torch.optim import Adam
from torch import nn
import numpy as np
from Networks import FFN
from matplotlib import pyplot as plt
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
    grad = torch.autograd.grad(output, x, grad_outputs=torch.ones_like(output), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return grad

sigma = torch.diag(torch.tensor([0.05, 0.05]))
sig = torch.mm(sigma, sigma.T)

def get_laplacian(grad, x):
    # hessian actually but anyway
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





class AgentDP:

    def __init__(self, batch_size, H, M, C, D, R, sigma, T,
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
        self.policy_net = FFN([3, 100, 100, 2], activation = nn.LogSigmoid)
        self.critic_net = Net_DGM(2, 100, activation = 'LogSigmoid')

        self.batch_size = batch_size
        self.H = H
        self.M = M
        self.R = R
        self.C = C
        self.D = D
        self.sigma = sigma
        self.tr = torch.mm(self.sigma, self.sigma.T).trace()
        self.T = T

        self.optim_critic = Adam(self.critic_net.parameters(), lr = learning_rate_critic)
        self.optim_policy = Adam(self.policy_net.parameters(), lr = learning_rate_policy)

        self.critic_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim_critic , milestones=(10000,), gamma=0.1)
        self.policy_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optim_policy, milestones=(10000,), gamma=0.1)
        self.loss_fn = torch.nn.MSELoss()

    def policy_improvement(self, t, x):
        '''
        :param t: one episode of t with batch size
        :param x: one episode of x with batch size
        :return: loss of critic network
        '''
        self.optim_policy.zero_grad()
        self.optim_critic.zero_grad()

        u_of_tx = self.critic_net(t, x)
        grad_u_x = get_gradient(u_of_tx, x)
        alpha = self.policy_net(t, x)

        H_loss = (torch.mul(torch.matmul(input_domain, self.H.T), grad_u_x)
                 + torch.mul(torch.matmul(alpha, self.M.T), grad_u_x)
                 + torch.mul(torch.matmul(x, self.C), x)
                 + torch.mul(torch.matmul(alpha, self.D), alpha)
                 ).sum(dim = 1)

        H_loss = H_loss.mean(dim = 0)

        H_loss.backward(retain_graph = True)
        self.optim_policy.step()
        self.policy_scheduler.step()

        return H_loss.item()

    def policy_evaluate(self, t, x):
        '''
        :param t: one episode of t with batch size
        :param x: one episode of x with batch size
        :return: loss of policy network
        '''
        self.optim_policy.zero_grad()
        self.optim_critic.zero_grad()

        t = t
        input_domain = x

        # alpha = torch.ones_like(input_domain)

        alpha = self.policy_net(t, x).detach()

        u_of_tx = self.critic_net(t, input_domain)
        grad_u_x = get_gradient(u_of_tx, input_domain)
        grad_u_t = get_gradient(u_of_tx, t)
        laplacian = get_laplacian(grad_u_x, input_domain)
        # hessian actually
        target_functional = torch.zeros_like(u_of_tx)

        pde = grad_u_t + 0.5 * laplacian \
              + (torch.mul(torch.matmul(input_domain, self.H.T), grad_u_x)
                 + torch.mul(torch.matmul(alpha, self.M.T), grad_u_x)
                 + torch.mul(torch.matmul(input_domain, self.C), input_domain)
                 + torch.mul(torch.matmul(alpha, self.D), alpha)
                 ).sum(dim=1).unsqueeze(1)

        MSE_functional = self.loss_fn(pde, target_functional)

        input_terminal = input_domain
        t_terminal = torch.ones(self.batch_size, 1) * self.T

        u_terminal = self.critic_net(t_terminal, input_terminal)
        target_terminal = torch.mul(torch.matmul(input_terminal, self.R), input_terminal).sum(dim = 1).unsqueeze(1)
        MSE_terminal = self.loss_fn(u_terminal, target_terminal)

        B_loss = MSE_functional + MSE_terminal
        B_loss.backward(retain_graph = True)
        self.optim_critic.step()
        self.critic_scheduler.step()

        print({'Bellman Loss:': B_loss.item()})
        return B_loss.item()

H = torch.eye(2).double()
M = torch.eye(2).double()
R = torch.eye(2).double()
C = torch.eye(2).double() * 0.8
D = torch.eye(2).double() * 0.1
T = 1
# sigma = torch.diag(torch.tensor([0.0, 0.0]))
sigma = torch.diag(torch.tensor([0.05, 0.05]))

b_size = 3000
# Initialize the dynamic programming agent
agent = AgentDP(b_size, H, M, C, D, R, sigma, T, 2e-3, 3e-3)

# Example of usage of update function
t = torch.rand(b_size, 1, requires_grad=True).double()
input_domain = (torch.rand(b_size, 2, requires_grad=True) - 0.5)*6
input_domain = input_domain.double()
# print(agent.policy_improvement(t,input_domain))
u_of_tx = agent.critic_net(t, input_domain)
grad_u_x = get_gradient(u_of_tx, input_domain)
grad_u_t = get_gradient(u_of_tx, t)
alpha = agent.policy_net(t, input_domain)
# print(agent.policy_evaluate(t, input_domain))

H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.8*np.identity(2)
D = 0.1*np.identity(2)
SIG = np.diag([0.05, 0.05])
# SIG = np.diag([0.0, 0.0])
t_grid = torch.from_numpy(np.linspace(0, 1, 10000))

t0 = torch.tensor([0]).float()
x0 = torch.tensor([1.5, 1.5]).float()
standard_sol = SolveLQR(H, M, C, D, R, SIG, t_grid)
standard_v = standard_sol.get_value(t0, x0)
standard_a = standard_sol.get_controller(t0, x0).squeeze()


print(standard_v)


def train_DP(Agent,
             criteria,
             max_steps,
             stop_critic_start,
             stop_policy,
             epsilon_decay = True,
             example = False
             ):
    '''
    :param Agent:
    :param criteria:
    :param max_steps:
    :return:
    '''
    l2 = torch.nn.MSELoss(reduction='mean')
    error = criteria * 1e6

    step = 1
    error_list_v = []
    error_list_a = []


    while (error > criteria) & (step < max_steps):

        t_stop = torch.rand(b_size, 1, requires_grad=False).double()
        x_stop = (torch.rand(b_size, 2, requires_grad=False) - 0.5) * 6
        x_stop = x_stop.double()
        value_last = Agent.critic_net(t_stop, x_stop).squeeze(1)

        print('Training Value Network...')
        if not epsilon_decay:
            stop_critic = stop_critic_start
        else:
            if step == 1:
                stop_critic = stop_critic_start
            else:
                stop_critic -= 1

        error_critic = 1e3
        while error_critic >= stop_critic:
            t = torch.rand(b_size, 1, requires_grad=True).double()
            x = (torch.rand(b_size, 2, requires_grad=True) - 0.5) * 6
            x = x.double()
            error_critic = Agent.policy_evaluate(t, x)

        print('Training Policy Network...')
        diff_error_policy = 1e3
        error_policy0 = 0
        while diff_error_policy >= stop_policy:
            t = torch.rand(b_size, 1, requires_grad=True).double()
            x = (torch.rand(b_size, 2, requires_grad=True) - 0.5) * 6
            x = x.double()
            error_policy1 = Agent.policy_improvement(t, x)
            diff_error_policy = np.abs(error_policy1 - error_policy0)
            print(diff_error_policy)
            error_policy0 = error_policy1

        value_improved = Agent.critic_net(t_stop, x_stop).squeeze(1)
        error = l2(value_improved, value_last).item()
        step += 1

        print({f'Step {step}': error})
        # Example
        t_test = torch.tensor([[0]]).double()
        x_test = torch.tensor([[1.5, 1.5]]).double()
        value_test = Agent.critic_net(t_test, x_test)
        print(value_test.item())
        alpha_test = Agent.policy_net(t_test, x_test).squeeze()
        error_a = torch.norm(alpha_test-standard_a, p=2, dim=0).item()
        error_list_v.append((torch.abs(value_test-standard_v)).item())
        error_list_a.append(error_a)

    if example:
        plt.plot(np.arange(0, step - 1), error_list_v, label = 'error of value')
        plt.plot(np.arange(0, step - 1), error_list_a, label = 'l2 error of alpha')
        plt.xlabel("Iterations", fontsize = 18)
        plt.ylabel("Error", fontsize = 18)
        plt.xticks(fontsize = 15)
        plt.yticks(fontsize = 15)
        plt.legend()
        plt.show()

    return value_test.item()

stop_value_iteration = 10
stop_control_iteration = 5e-1
train_DP(agent, 3e-3, 10, stop_value_iteration, stop_control_iteration, epsilon_decay = True, example = True)

