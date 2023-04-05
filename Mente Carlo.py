import numpy as np
import torch
from E1_1 import SolveLQR
from tqdm import tqdm
from matplotlib import pyplot as plt

class Mente_Carlo:

    def __init__(self, time_steps: list, sample_size: int, method: str):

        '''
        Initialize LQR solver
        '''
        self.H = np.identity(2)
        self.M = np.identity(2)
        self.R = np.identity(2)
        self.C = 0.1 * np.identity(2)
        self.D = 0.1 * np.identity(2)
        self.T = 1
        self.Sigma = np.diag([0.05, 0.05])
        self.solver = SolveLQR(self.H, self.M, self.Sigma, self.C, self.D, self.R, self.T)

        '''
        Initialize tensors
        '''
        self.H_T = torch.tensor(np.identity(2)).float()
        self.M_T = torch.tensor(np.identity(2)).float()
        self.R_T = torch.tensor(torch.tensor(np.identity(2))).float()
        self.C_T = torch.tensor(0.1*np.identity(2)).float()
        self.D_T = torch.tensor(0.1*np.identity(2)).float()
        self.T = 1
        self.Sigma_T = torch.tensor(np.diag([0.05, 0.05])).float()
        self.x = torch.tensor([[3, 3]]).float()

        '''
        Initialize sample sizes and time space
        '''
        self.simu_method = method
        self.time_steps = time_steps
        self.delta_t = self.time_steps[1] - self.time_steps[0]
        self.sample_size = sample_size


    def value_function(self):
        return self.solver.get_value(self.time_steps, self.x)

    def MC_controller(self, t, x):
        return self.solver.get_controller(t, x)

    def drift_ode(self, x_t, alpha):
        return (torch.matmul(self.H_T, x_t.T) + torch.matmul(self.M_T, alpha.T)) * self.delta_t

    def X_simu(self):

        X = self.x
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

    def train_MC(self, measure = True, visualize = True):

        error_list = []
        value = self.value_function()
        episodes = self.sample_size

        G = 0
        for eps in tqdm(range(1, episodes + 1)):
            sample, alpha_list = self.X_simu()
            r = self.objective_function(sample, alpha_list, 0.001)
            G = (G * (eps - 1) + r) / eps

            error = np.abs(G - value)

            if measure:
                print(f'The l1-norm is evaluated by: {error.item()}')

            if visualize:
                error_list.append(error.item())
                if eps == episodes:
                    plt.plot(np.arange(1, episodes + 1), error_list)
                    plt.xlabel("Timesteps", fontsize=20)
                    plt.ylabel("Loss", fontsize=20)
                    plt.xticks(fontsize=15)
                    plt.yticks(fontsize=15)
                    plt.tight_layout(pad=0.3)

                    plt.show()


t = torch.from_numpy(np.linspace(0, 1, 1000))
mc = Mente_Carlo(t, 100, 'sample')
x_list, alpha_list = mc.X_simu()
# print(objective_function(x_list, alpha_list, 0.001, R_T))
# print(value_function(lqr, t, x))
mc.train_MC(measure = False)