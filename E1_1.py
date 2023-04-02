import numpy
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import torch

from scipy.integrate import odeint


class SolveLQR:

    def __init__(self, h, m, sigma, c, d, r, capt):
        #variables in lowercase are corresponding uppercase matrices or arrays in LQR
        self.h = h
        self.m = m
        self.sigma = sigma
        self.c = c
        self.d = d
        self.r = r
        self.capt = capt

    def ricatti_ode(self, s, t):
        #input and output of solution need to be  1*4 array limited to ode solver
        #reshape to matrix
        s = s.reshape([2, 2])
        ds = - 2 * self.h.T * s + s * self.m * np.linalg.inv(self.d) * self.m * s - C
        return ds.reshape(-1)

    def trace_sigsig_s(self, s, t):
        ds = np.linalg.matrix_rank(self.sigma*self.sigma.T*s)
        return ds

    def sol_ricatti(self, time):
        if type(time) != np.ndarray:
            time = time.numpy()
        r = self.r.reshape(-1)
        #input time be strictly monotone increasing numpy array
        time_backwards = time[::-1] #solving backwards by setting decreasing time sequence
        sol_s_backwards = odeint(self.ricatti_ode, r, time_backwards)
        sol_s = sol_s_backwards[::-1]
        l = len(sol_s)
        sol_s = sol_s.reshape(l,2,2)
        #reversed solution corresponding to input time(increasing)
        return sol_s

    def get_value(self, time, space):
        #create another ode and solve or just use data from sol_ricatti
        if type(time) != np.ndarray:
            time = time.numpy()
        #input time be strictly monotone increasing numpy array
        time_backwards = time[::-1] #solving backwards by setting decreasing time sequence
        integral_backwards = odeint(self.trace_sigsig_s, 0, time_backwards)
        integral = torch.from_numpy(integral_backwards[::-1].copy())
        s = torch.from_numpy(self.sol_ricatti(time).copy()).float()
        l = len(space)
        value = torch.bmm(torch.bmm(space, s), torch.reshape(space,(l,2,1))).squeeze(2) + integral
        #reversed solution corresponding to input time(increasing)
        return value

    def get_controller(self, time, space):
        if type(time) != np.ndarray:
            time = time.numpy()
        s = torch.from_numpy(self.sol_ricatti(time).copy()).float()
        l = len(space)
        a0 = torch.from_numpy(- self.d * self.m.T).float()
        a1 = torch.bmm(s, torch.reshape(space, (l,2,1)))
        a = torch.matmul(a0, a1).squeeze()
        return a


H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.1*np.identity(2)
D = 0.1*np.identity(2)
T = 1
Sigma = np.diag([0.05, 0.05])

LQR1 = SolveLQR(H, M, Sigma, C, D, R, T)