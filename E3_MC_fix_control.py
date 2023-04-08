import numpy as np
import torch
from E1_1 import SolveLQR
from tqdm import tqdm
from matplotlib import pyplot as plt

class Monte_Carlo_fix:

    def __init__(self, model_para: list, time_grid, sample_size: int):

        '''
        Initialize LQR solver, model_para = [H, M, C, D, R, Sigma], T is the terminal of time grid
        '''
        self.H = model_para[0]
        self.M = model_para[1]
        self.C = model_para[2]
        self.D = model_para[3]
        self.R = model_para[4]
        self.Sigma = model_para[5]

        self.solver = SolveLQR(self.H, self.M,  self.C, self.D, self.R, self.Sigma, time_grid)

        '''
        Initialize tensors
        '''
        self.H_T = torch.tensor(self.H).float()
        self.M_T = torch.tensor(self.M).float()
        self.R_T = torch.tensor(self.R).float()
        self.C_T = torch.tensor(self.C).float()
        self.D_T = torch.tensor(self.D).float()
        self.T = 1
        self.Sigma_T = torch.tensor(self.Sigma).float()

        '''
        Initialize sample sizes and time space
        '''
        self.time_steps = time_grid
        self.delta_t = self.time_steps[1] - self.time_steps[0]
        self.sample_size = sample_size


    def value_function(self, t, x):
        return self.solver.get_value(t, x)

    def MC_controller(self, t, x):
        return torch.tensor([1, 1])

    def drift_ode(self, x_t, alpha):
        return (torch.matmul(self.H_T, x_t.T) + torch.matmul(self.M_T, alpha.T)) * self.delta_t

    def X_simu(self, position):
        X = position
        t = 0
        x_list = []
        alpha_list = []

        while t < self.T:
            alpha_t = self.MC_controller(torch.tensor([t]), X).squeeze().float()
            BM = torch.normal(mean = 0, std = np.sqrt(self.delta_t), size = (1, 2))
            diffusion = torch.transpose(torch.matmul(self.Sigma_T, BM.unsqueeze(2)), 1, 2)
            X = X + self.drift_ode(alpha_t, X.squeeze().float()) + diffusion
            t += self.delta_t
            x_list.append(X.squeeze())
            alpha_list.append(alpha_t)
        return x_list, alpha_list

    def objective_function(self, sample, control, delta_t):
        X_T = sample[-1]
        terminal = torch.matmul(torch.matmul(X_T, self.R_T), X_T.T)
        f = 0
        for i in range(len(sample)):
            f += (
                    torch.matmul(torch.matmul(sample[i], self.C_T), sample[i].T) + torch.matmul(torch.matmul(control[i], self.D_T), control[i].T)
                  ) * delta_t

        return f + terminal

    def MC_fix_control(self, t0, x):

        error_list = []
        value = self.value_function(t0, x)
        episodes = self.sample_size

        G = 0
        for eps in tqdm(range(1, episodes + 1)):
            sample, alpha_list = self.X_simu(x)
            r = self.objective_function(sample, alpha_list, self.delta_t)
            G = (G * (eps - 1) + r) / eps
            # error = np.abs((G - value)/ value)
        return G

H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.8*np.identity(2)
D = 0.1*np.identity(2)
SIG = np.diag([0.05, 0.05])
model_p = [H, M, C, D, R, SIG]

t0 = torch.tensor([0])
x = torch.tensor([[1, 1]]).float()
t_grid = torch.from_numpy(np.linspace(0, 1, 1000))
mc = Monte_Carlo_fix(model_p, t_grid, 100)
# mc at t = 0, x = [2,2]
value_fix = mc.MC_fix_control(t0, x)
print(value_fix)