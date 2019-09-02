# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan, pi
import scipy.integrate as sci
import scipy.linalg as scilin
import scipy.interpolate as sciinterp
import matplotlib.pyplot as plt
from Planner import PolynomialPlanner


class Parameters(object):
    pass


# Physical parameter
para = Parameters()  # instance of class Parameters
para.a = 1           # model parameter

# Simulation parameter
sim_para = Parameters()  # instance of class Parameters
sim_para.t0 = 0          # start time
sim_para.tf = 20         # final time
sim_para.dt = 0.05       # step-size

# Controller parameters
Q = np.diag([1, 1])
R = 1

# time vector
tt = np.arange(sim_para.t0, sim_para.tf + sim_para.dt, sim_para.dt)

# initial state
x0 = [-2.1, 0.2]


def calc_system_matrices(x, p):
    x1, x2 = x
    A = np.array([[0, p.a*cos(x2)],
                  [-2*x1, 0]])

    B = np.array([[0],
                  [1]])

    return A, B


def calc_trajectory():
    planner = PolynomialPlanner([-2, 0, 0], [2, 0, 0], 0, 10, 2)
    x1d_and_derivatives = planner.eval_vec(tt)
    x1d = x1d_and_derivatives[:, 0]
    x1d_dot = x1d_and_derivatives[:, 1]
    x1d_ddot = x1d_and_derivatives[:, 2]
    x2d = np.arcsin(x1d_dot / para.a)
    x2d_dot = x1d_ddot / np.sqrt(para.a ** 2 - np.square(x1d_dot))

    ud = x2d_dot + np.square(x1d)
    ud_interp = sciinterp.interp1d(tt, ud)

    xd = np.stack((x1d, x2d), axis=1)
    xd_interp = sciinterp.interp1d(tt, xd, axis=0)

    return xd_interp, ud_interp


xd_interp, ud_interp = calc_trajectory()

Ad, Bd = calc_system_matrices(np.array([0, 0]), para)

P = scilin.solve_continuous_are(Ad, Bd, Q, R)
K = -1/R * Bd.T @ P


def ode(t, x, p):
    """Function of the robots kinematics

    Args:
        x        : state
        t        : time
        p(object): parameter container class

    Returns:
        dxdt: state derivative
    """
    x1, x2 = x  # state vector
    u = control(t, x)  # control vector

    # dxdt = f(x, u):
    dxdt = np.array([p.a * sin(x2),
                     -x1**2 + u])

    # return state derivative
    return dxdt


def control(t, x):
    """Function of the control law

    Args:
        x: state vector
        t: time

    Returns:
        u: control signal

    """
    xd_t = xd_interp(t)
    ud_t = ud_interp(t)

    u = K @ (x - xd_t) + ud_t

    return u


# simulation
sol = sci.solve_ivp(lambda t, x: ode(t, x, para), (sim_para.t0, sim_para.tf), x0, t_eval=tt)
x_traj = sol.y.T
u_traj = np.array([control(tt[i], x_traj[i]) for i in range(len(tt))])

xd_traj = xd_interp(tt)
ud_traj = ud_interp(tt)

plt.figure()

plt.subplot(211)
plt.plot(tt, x_traj)
plt.plot(tt, xd_traj, '--')
plt.legend(['x1', 'x2', 'x1d', 'x2d'])
plt.grid()

plt.subplot(212)
plt.plot(tt, u_traj)
plt.plot(tt, ud_traj, '--')
plt.grid()

plt.show()