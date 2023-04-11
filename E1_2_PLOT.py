import numpy
import numpy as np
import torch
from E1_1 import SolveLQR
from tqdm import tqdm
from matplotlib import pyplot as plt


class MC_plot:

    def __init__(self, model_para: list, time_steps: int, sample_size: int):

        '''
        Initialize LQR solver, model_para = [H, M, C, D, R, Sigma], T is the terminal of time grid
        '''
        self.H = model_para[0]
        self.M = model_para[1]
        self.C = model_para[2]
        self.D = model_para[3]
        self.R = model_para[4]
        self.Sigma = model_para[5]
        self.time_grid = np.linspace(0,1,10001)
        self.solver = SolveLQR(self.H, self.M,  self.C, self.D, self.R, self.Sigma, self.time_grid)

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
        self.time_steps = np.linspace(0,1,time_steps)
        if len(self.time_steps) == 1:
            self.delta_t = 1
        else:
            self.delta_t = self.time_steps[1] - self.time_steps[0]
        self.sample_size = sample_size


    def value_function(self, t, x):
        return self.solver.get_value(t, x)

    def MC_controller(self, t, x):
        return self.solver.get_controller(t, x)

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

    def train_MC(self, t0, x, relative = False):

        error_list = []
        value = self.value_function(t0, x)
        episodes = self.sample_size
        G = 0
        G_list = []
        for eps in tqdm(range(1, episodes + 1)):
            sample, alpha_list = self.X_simu(x)
            r = self.objective_function(sample, alpha_list, self.delta_t)
            G = (G * (eps - 1) + r) / eps
            G_list.append(G)
            if  relative == True:
                error = np.abs((G - value) / value)
            else:
                error = np.abs(G - value)
            error_list.append(error)
        return np.array(error_list)

H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.8*np.identity(2)
D = 0.1*np.identity(2)
SIG = np.diag([0.05, 0.05])
model_p = [H, M, C, D, R, SIG]


def plot_time(fix_sample, time_step_lists, test_t, test_domain, relative = False):
    sup_error_list = np.ones((len(test_domain), len(time_step_lists)))
    for i in range(len(time_step_lists)):
        t_steps = time_step_lists[i]
        MC_time = MC_plot(model_p, t_steps, fix_sample)
        for j in range(len(test_domain)):
            x = test_domain[j]
            error = MC_time.train_MC(test_t, x, relative)
            sup_error_list[j][i] = error[-1]
    return sup_error_list


def plot_sample(sample_list, fix_timestep, test_t, test_domain, relative = False):
    sup_error_list = np.ones((len(test_domain), len(sample_list)))
    for i in range(len(sample_list)):
        samples = sample_list[i]
        MC_sample = MC_plot(model_p, fix_timestep, samples)
        for j in range(len(test_domain)):
            x = test_domain[j]
            error = MC_sample.train_MC(test_t, x, relative)
            sup_error_list[j][i] = error[-1]
    return sup_error_list


# choose test set
t0 = torch.tensor([0])
x0 = torch.tensor([[0.5, 0]]).float()
x1 = torch.tensor([[1, 1]]).float()
x2 = torch.tensor([[0, 1.5]]).float()
x3 = torch.tensor([[-2., 2]]).float()
x4 = torch.tensor([[-2.5, -2.5]]).float()
test_x = [x0, x1, x2, x3, x4]

# plot parameter
test_lable = ["x = [0.5, 0]",
              "x = [1, 1]",
              "x = [0, 1.5]",
              "x = [-2, 2]",
              "x = [-2.5, -2.5]"]
test_color = ['coral',
              'skyblue',
              'khaki',
              'limegreen',
              'aquamarine']


# set variation
f_sample = 5000
f_timestep = 2000
t_step_lists = [1, 10, 50, 100, 500, 1000, 5000, 10000]
s_lists = [10, 50, 100, 500, 1000, 5000, 10000]

# plot execution function
def plot_exercise_1_2_2():
    t_analysis = plot_time(f_sample, t_step_lists, t0, test_x)
    plt.xlabel("Log of Time Steps", fontsize=18)
    plt.ylabel("Log of Absolute Error on Test Sets", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    for i in range(len(test_x)):
        plt.loglog(t_step_lists, t_analysis[i], color = test_color[i], label = test_lable[i])
    plt.legend()
    plt.show()

def plot_exercise_1_2_3():
    s_analysis = plot_sample(s_lists, f_timestep, t0, test_x)
    plt.xlabel("Log of Samples Number", fontsize=18)
    plt.ylabel("Log of Absolute Error on Test Sets", fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    for i in range(len(test_x)):
        plt.loglog(s_lists, s_analysis[i], color = test_color[i], label = test_lable[i])
    plt.legend()
    plt.show()

# plot_exercise_1_2_2()
plot_exercise_1_2_3()

