# -*- coding: utf-8 -*-
import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt


class Parameters(object):
    pass


# ODE parameter
para = Parameters()  # instance of class Parameters
para.a = 1 # dxdt = a*x

# Simulation parameter
sim_para = Parameters()  # instance of class Parameters
sim_para.t0 = 0          # start time
sim_para.tf = 10         # final time
sim_para.dt = 0.04       # step-size


def ode(t, x, p):
    """Function of the robots kinematics

    Args:
        x        : state
        t        : time
        p(object): parameter container class

    Returns:
        dxdt: state derivative
    """
    x1, = x  # state vector

    # dxdt = f(x, u):
    dxdt = np.array([p.a*x1])

    # return state derivative
    return dxdt

def event(t, x):
    x_max = 5
    return np.abs(x)-x_max

event.terminal = True
# time vector
tt = np.arange(sim_para.t0, sim_para.tf + sim_para.dt, sim_para.dt)

# initial state
x0 = [1]

# simulation
sol = sci.solve_ivp(lambda t, x: ode(t, x, para), (sim_para.t0, sim_para.tf), x0, t_eval=tt, events=event)
x_traj = sol.y.T

plt.plot(sol.t, x_traj)
plt.xlabel(r't in s')
plt.ylabel(r'$x(t)$')
plt.show()

