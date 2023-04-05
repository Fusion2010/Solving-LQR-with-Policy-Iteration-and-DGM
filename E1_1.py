import numpy as np
import scipy as sp
from scipy.integrate import simpson
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
        ds = - 2 * self.h.T * s + s * self.m * np.linalg.inv(self.d) * self.m * s - self.c
        return ds.reshape(-1)

    def sol_ricatti(self, time):
        if type(time) != np.ndarray:
            time = time.numpy()
        sol_s = [self.r]
        print(time)
        dt = time[1] - time[0]
        for i in range(len(time) - 1):
            s = sol_s[i]
            ds = - 2 * self.h.T * s + s* self.m * np.linalg.inv(self.d) * self.m * s - self.c
            s -= ds * dt
            sol_s.append(s)

        sol_s = sol_s[::-1]
        #reversed solution corresponding to input time(increasing)
        return sol_s

    def get_value(self, time, space):
        #time is a uniform time array or tensor
        #space is a trensor. i.e. tesor([[1,1]])
        #return value at (time[0], space)
        space = space.squeeze(0).float()
        s = self.sol_ricatti(time)
        integrand = self.sigma * self.sigma.T * s
        s0 = torch.from_numpy(s[0]).float()
        integral = 0
        dt = time[1]-time[0]
        for i in range(len(time) - 1):
            dy = integrand[i].trace() * dt
            integral += dy
        value = torch.mm(torch.mm(space.unsqueeze(0), s0), space.unsqueeze(1)).squeeze() + integral
        return value.float()

    def get_controller(self, time, space):
        s = torch.from_numpy(self.sol_ricatti(time).copy()).float()
        s = s[0]
        a0 = torch.from_numpy(- self.d * self.m.T).float()
        a1 = torch.mm(s, torch.reshape(space, (2,1)))
        a = torch.mm(a0, a1).squeeze()

        return a


H = np.identity(2)
M = np.identity(2)
R = np.identity(2)
C = 0.1*np.identity(2)
D = 0.1*np.identity(2)
T = 1
Sigma = np.diag([0.05, 0.05])

space = torch.tensor([0, 0])
LQR1 = SolveLQR(H, M, Sigma, C, D, R, T)
t = np.linspace(0, 1, 1000)

print(LQR1.sol_ricatti(t))