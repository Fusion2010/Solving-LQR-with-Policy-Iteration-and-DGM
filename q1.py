import numpy
import numpy as np
import scipy as sp
import matplotlib as mpt
import torch

from scipy.integrate import odeint


class SolveLQR:

    def __init__(self, h, m, sigma, c, d, r, capt):
        #variables in lowercase are corresponding uppercase matrices or arrays in LQR
        self.h = h
        self.m = m
        self.sig = sigma
        self.c = c
        self.d = d
        self.r = r
        self.capt = capt

    def ricatti_ode(self, s, t):
        #input and output of solution need to be  1*4 array limited to ode solver
        #reshape to matrix
        s = s.reshape([2, 2])
        ds = - 2 * self.h.T * s + s * self.m * self.d.I * self.m * s - C

        return ds.reshape(-1)

    def trace_sigsig_s(self, s, t):
        s = s.reshape([2, 2])
        ds = np.linalg.matrix_rank(self.sigma*self.sigma.T*s)
        return ds.reshape(-1)

    def sol_ricatti(self, time):
        if type(time) != np.ndarray:
            time = time.numpy()
        #input time be strictly monotone increasing numpy array
        time_backwards = time[::-1] #solving backwards by setting decreasing time sequence
        sol_s_backwards = odeint(self.ricatti_ode, self.r, time_backwards)
        sol_s = sol_s_backwards[::-1]
        #reversed solution corresponding to input time(increasing)
        return sol_s

    def get_value(self, time, space):
        #create another ode and solve or just use data from sol_ricatti
        if type(time) != np.ndarray:
            time = time.numpy()
        if type(space) != np.ndarray:
            space = space.numpy()
        #input time be strictly monotone increasing numpy array
        time_backwards = time[::-1] #solving backwards by setting decreasing time sequence
        integral_backwards = odeint(trace_sigma_sigmaT_s, 0, time_backwards)
        integral = integral_backwards[::-1]
        value = space.transpose(0,2,1)*self.sol_ricatti(time)*space + integral
        value = torch.from_numpy(value)
        #reversed solution corresponding to input time(increasing)
        return value

    def get_controller(self, time, space):
        if type(time) != np.ndarray:
            time = time.numpy()
        if type(space) != np.ndarray:
            space = space.numpy()
        s = np.array(self.get_value(self, time, space))
        #input time be strictly monotone increasing numpy array
        a = -self.D*self.M.T*s*space
        a = torch.from_numpy(a)
        #reversed solution corresponding to input time(increasing)
        return a


