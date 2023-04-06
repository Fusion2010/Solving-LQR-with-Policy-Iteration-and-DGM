import numpy as np
import scipy as sp
from scipy.integrate import simpson
import matplotlib.pyplot as plt

import torch

from scipy.integrate import odeint


class SolveLQR:

    def __init__(self, h, m,  c, d, r, sigma, time_grid):
        #variables in lowercase are corresponding uppercase matrices or arrays in LQR
        self.h = h
        self.m = m
        self.c = c
        self.d = d
        self.r = r
        self.sigma = sigma
        self.time = np.array(time_grid)
        self.capt = self.time[-1]
        self.dt = self.time[1] - self.time[0]
        self.solution = self.sol_ricatti()

    def sol_ricatti(self):
        #solve on time_grid
        sol_s = [self.r]
        for i in range(len(self.time) - 1):
            s = sol_s[i]
            ds = - 2 * np.dot(self.h.T, s) \
                 + np.dot(s.dot(self.m), np.linalg.inv(self.d)).dot(self.m).dot(s) - self.c
            s -= ds * self.dt
            sol_s.append(s)
        sol_s = sol_s[::-1]
        #reversed solution corresponding to input time(increasing)
        return sol_s

    def get_value(self, time, space):
        # if type(time) != torch.Tensor:
        #     time = torch.tensor(time)
        # if type(space) != torch.Tensor:
        #     space = torch.tensor(space)
        #cnvert time tensor to corresponding index list, index on timegrid
        time_index_list = torch.div(time, self.dt).floor().tolist()
        v1 = torch.zeros_like(time)
        for i in range(len(time)):
            t0_index = int(time_index_list[i])
            integral = 0
            while t0_index < len(self.solution)-1:
                sr = np.array(self.solution[t0_index])
                integrand = np.dot(np.dot(self.sigma, self.sigma.T), sr)
                integral += integrand.trace() * self.dt
                t0_index += 1
            v1[i] = integral

        st = np.array([self.solution[int(index)] for index in time_index_list])
        #convert to np.array first to speed up
        st = torch.tensor(st).float()
        l = len(time)
        v0 = torch.matmul(torch.matmul(space, st), space.reshape(l, 2, 1))
        v = v0.squeeze() + v1.squeeze()
        return v

    def get_controller(self, time, space):
        time_index_list = torch.div(time, self.dt).floor().tolist()
        st = np.array([self.solution[int(index)] for index in time_index_list])
        #to speed up
        st = torch.tensor(st).float()
        a0 = torch.matmul(torch.tensor(- self.d), torch.tensor(self.m.T)).float()
        l = len(time)
        a1 = torch.matmul(st, space.reshape(l, 2, 1)).float()
        a = torch.matmul(a0, a1)
        return a


H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.1*np.identity(2)
D = 0.1*np.identity(2)
t_grid = np.linspace(0, 1, 1000)
Sigma = np.diag([0.05, 0.05])
x = torch.tensor([[3, 3]]).float()
t_grid = torch.from_numpy(np.linspace(0, 1, 10000))
# t = torch.tensor(t_grid)
# batch_size = 1000
# space =(torch.rand(batch_size, 1, 2) - 0.5)*6
LQR1 = SolveLQR(H, M, C, D, R, Sigma, t_grid)
#
v = LQR1.get_value(torch.tensor([1]).float(), torch.tensor([-3, 3]).float())
# print(v)
# a = LQR1.get_controller(t, space)
# plt.plot(v)
# plt.show()