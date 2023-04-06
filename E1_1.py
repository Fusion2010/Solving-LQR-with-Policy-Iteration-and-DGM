import numpy as np
import scipy as sp
from scipy.integrate import simpson
import matplotlib.pyplot as plt

import torch

from scipy.integrate import odeint


class SolveLQR:

    def __init__(self, h, m, sigma, c, d, r, time_grid):
        #variables in lowercase are corresponding uppercase matrices or arrays in LQR
        self.h = h
        self.m = m
        self.sigma = sigma
        self.c = c
        self.d = d
        self.r = r
        self.capt = time_grid[-1]
        self.dt = time_grid[1] - time_grid[0]
        self.time = time_grid
        self.solution = self.sol_ricatti()

    def sol_ricatti(self):
        #solve on time_grid
        sol_s = [self.r]
        for i in range(len(self.time) - 1):
            s = sol_s[i]
            ds = - 2 * self.h.T * s + s* self.m * np.linalg.inv(self.d) * self.m * s - self.c
            s -= ds * self.dt
            sol_s.append(s)
        sol_s = sol_s[::-1]
        #reversed solution corresponding to input time(increasing)
        return sol_s

    def get_value(self, time, space):
        #cnvert time tensor to corresponding index list, index on timegrid
        time_index_list = torch.div(time, self.dt).floor().tolist()
        v1 = torch.zeros_like(time)
        for i in range(len(time)):
            t0_index = time_index_list[i]
            integral = 0
            while t0_index < len(self.solution)-1:
                sr = np.array(self.solution[t0_index])
                integrand = np.dot(np.dot(self.sigma, self.sigma.T), sr)
                integral += integrand.trace() * self.dt
                t0_index += 1
            v1[i] = integral

        st = torch.tensor([self.solution[index] for index in time_index_list])
        l = len(time)
        v0 = torch.matmul(torch.matmul(space, st), space.reshape(l,2,1))
        v = v0.squeeze() + v1.squeeze()
        return v.unsqueeze(1)

    def get_controller(self, time, space):
        time_index_list = torch.div(time, self.dt).floor().tolist()
        st = torch.tensor([self.solution[index] for index in time_index_list])
        a0 = torch.matmul(- self.d, self.m.T)
        l = len(time)
        a1 = torch.matmul(st, space.reshape((l, 2, 1)))
        a = torch.matmul(a0, a1)
        return a


H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.1*np.identity(2)
D = 0.1*np.identity(2)
t_grid = np.linspace(0, 1, 10001)
Sigma = np.diag([0.05, 0.05])
t = torch.tensor(t_grid)
space = torch.tensor()
LQR1 = SolveLQR(H, M, Sigma, C, D, R, t_grid)
a = LQR1.get_value(t, [0, 0])
