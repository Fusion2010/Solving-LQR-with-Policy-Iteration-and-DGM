import numpy as np
import torch
from E1_1 import SolveLQR
from tqdm import tqdm

H_T = torch.tensor(np.identity(2)).float()
M_T = torch.tensor(np.identity(2)).float()
R_T = torch.tensor(torch.tensor(np.identity(2))).float()
C_T = torch.tensor(0.1*np.identity(2)).float()
D_T = torch.tensor(0.1*np.identity(2)).float()
T = 1
Sigma_T = torch.tensor(np.diag([0.05, 0.05])).float()
x = (torch.rand(1, 1, 2) - 0.5)*6

def value_function(solver, time, space):
    return solver.get_value(time, space)

def MC_controller(solver, time, space):
    return solver.get_controller(time, space)

def drift_ode(H, M, alpha, x, delta_t):
    return (torch.matmul(H, x.T) + torch.matmul(M, alpha.T)) * delta_t

def X_simu(solver, sigma, T, x_init = x, delta_t = 0.001):

    X = x_init
    t = 0
    x_list = []
    alpha_list = []

    while t < T:
        alpha_t = MC_controller(solver, torch.tensor([t]), X).squeeze().float()
        BM = torch.normal(mean = 0, std = np.sqrt(delta_t), size = (1, 2, 1))
        diffusion = torch.transpose(torch.matmul(sigma, BM), 1, 2)
        X = X + drift_ode(H_T, M_T, alpha_t, X.squeeze().float(), delta_t) + diffusion
        t += delta_t
        x_list.append(X.squeeze())
        alpha_list.append(alpha_t)

    return x_list, alpha_list

def objective_function(sample, control, delta_t, R_T):

    X_T = sample[-1]
    terminal = torch.matmul(torch.matmul(X_T, R_T), X_T.T)
    f = 0
    for i in range(len(sample)):
        f += (
                torch.matmul(torch.matmul(sample[i], C_T), sample[i].T) + torch.matmul(torch.matmul(control[i], C_T), control[i].T)
              ) * delta_t

    return f + terminal

def train_MC(episodes, t, x, measure = True, method = 'l1'):

    G = 0
    for eps in tqdm(range(1, episodes + 1)):
        sample, alpha_list = X_simu(lqr, Sigma_T, 1)
        r = objective_function(sample, alpha_list, 0.001, R_T)
        G = (G * (eps - 1) + r) / eps

    value = value_function(lqr, torch.tensor([t]), x)
    error = np.abs(G - value)
    if measure:
        print(f'The {method}-norm is evaluated by: {error.item()}')
    else:
        return G

H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.1*np.identity(2)
D = 0.1*np.identity(2)
T = 1
Sigma = np.diag([0.05, 0.05])
lqr = SolveLQR(H, M, Sigma, C, D, R, T)

input_domain = (torch.rand(1, 1, 2) - 0.5)*6
t = torch.from_numpy(np.linspace(0, 1, 1))
x_list, alpha_list = X_simu(lqr, Sigma_T, T, x)

# print(objective_function(x_list, alpha_list, 0.001, R_T))
# print(value_function(lqr, torch.tensor([0]), x))
# train_MC(100, 0, x)