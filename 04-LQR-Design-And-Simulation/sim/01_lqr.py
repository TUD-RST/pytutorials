# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan, pi
import scipy.linalg as scilin
import matplotlib.pyplot as plt
from Planner import PolynomialPlanner
from typing import Literal


class Parameters(object):
    pass


# define simulation parameters
sim_para = Parameters()  # instance of class Parameters
sim_para.t0 = 0          # start time
sim_para.tf = 10         # final time
sim_para.dt = 0.1        # step-size
sim_para.x0 = [-2.1, 0.1] # initial value

# already prepare the time vector because we'll need it very soon
n_samples = int((sim_para.tf - sim_para.t0) / sim_para.dt) + 1
t_traj = sim_para.t0 + np.arange(n_samples) * sim_para.dt

# -- START SYSTEM SPECIFIC PART --
# define system functions
sys_para = Parameters()  # instance of class Parameters
sys_para.n = 2           # number of states
sys_para.m = 1           # number of inputs
sys_para.a = 1           # model parameter


def system_rhs(t, x, u, para):
    x1, x2 = x  # state vector

    # dxdt = f(x, u):
    dxdt = np.array([para.a * sin(x2),
                     -x1**2 + u])

    # return state derivative
    return dxdt


def system_matrices(t, x, u, para):
    x1, x2 = x
    A = np.array([[0, para.a * cos(x2)],
                  [-2 * x1, 0]])

    B = np.array([[0],
                  [1]])

    return A, B


# define controller parameters
Q = np.diag([1, 1])
R = np.array([[1]])
# S = np.array([[4.5, 0.1], [0.1, 1.2]])

# trajectory parameters
traj_para = Parameters()
traj_para.y0 = -2
traj_para.yf = 2
traj_para.t0 = 0
traj_para.tf = 10

# calculate trajectory
planner = PolynomialPlanner([traj_para.y0, 0, 0], [traj_para.yf, 0, 0], traj_para.t0, traj_para.tf, 2)

x1d_and_derivatives = planner.eval_vec(t_traj)
x1d = x1d_and_derivatives[:, 0]
x1d_dot = x1d_and_derivatives[:, 1]
x1d_ddot = x1d_and_derivatives[:, 2]
x2d = np.arcsin(x1d_dot / sys_para.a)
x2d_dot = x1d_ddot / np.sqrt(sys_para.a ** 2 - np.square(x1d_dot))

xd_traj = np.stack((x1d, x2d), axis=1)
ud_traj = x2d_dot + np.square(x1d)


# -- END SYSTEM SPECIFIC PART --

R_inv = scilin.inv(R)


# solve matrix riccati ODE
def triu_to_full(triu):
    n = int(round((np.sqrt(1+8*len(triu))-1)/2))
    mask = np.triu(np.ones((n, n), dtype=bool))

    full = np.empty((n, n))
    full[mask] = triu
    full = full.T
    full[mask] = triu

    return full


def full_to_triu(full):
    mask = np.triu(np.ones(full.shape, dtype=bool))
    return full[mask]


# Get initial value S for matrix Riccati ODE by solving algebraic Riccati equation
A_f, B_f = system_matrices(t_traj[-1], xd_traj[-1], ud_traj[-1], sys_para)
S = scilin.solve_continuous_are(A_f, B_f, Q, R)

Ptilde_triu_traj = np.empty((n_samples, int(sys_para.n*(sys_para.n + 1)/2)))
Ptilde_triu_traj[0, :] = full_to_triu(S)

K_traj = np.empty((n_samples, sys_para.m, sys_para.n))

# get trajectories for P and K via numerical integration
for i_tau in range(n_samples):
    i_t = n_samples - 1 - i_tau
    t_i = t_traj[i_t]
    tau_i = t_traj[i_tau]
    xd_i = xd_traj[i_t]
    ud_i = ud_traj[i_t]
    A_i, B_i = system_matrices(t_i, xd_i, ud_i, sys_para)
    Ptilde_triu_i = Ptilde_triu_traj[i_tau]
    Ptilde_i = triu_to_full(Ptilde_triu_i)

    K_traj[i_t] = R_inv @ B_i.T @ Ptilde_i

    if i_tau < n_samples - 1:
        dPtilde_dtau = - Ptilde_i @ B_i @ R_inv @ B_i.T @ Ptilde_i + Ptilde_i @ A_i + A_i.T @ Ptilde_i + Q
        dPtilde_dtau_triu = full_to_triu(dPtilde_dtau)

        Ptilde_triu_traj[i_tau + 1] = Ptilde_triu_i + sim_para.dt * dPtilde_dtau_triu  # one Euler step

# compute static LQR feedback
t_static = 5
i_static = 0
while i_static < len(t_traj) - 1 and t_traj[i_static] < t_static:  # find index of that time in time vector
    i_static += 1
x_static = xd_traj[i_static]
ud_static = ud_traj[i_static]
A_static, B_static = system_matrices(t_static, x_static, ud_static, sys_para)
P_static = scilin.solve_continuous_are(A_static, B_static, Q, R)
K_static = R_inv * B_static.T @ P_static

# main simulation loop
x_traj = np.empty((n_samples, sys_para.n))
x_traj[0] = sim_para.x0
u_traj = np.empty((n_samples, sys_para.m))

FeedbackMode = Literal["LTV", "LTI", "pseudoLTV"]
feedback_mode: FeedbackMode = "LTV"

for i in range(n_samples):
    t_i = t_traj[i]
    x_i = x_traj[i]

    xd_i = xd_traj[i]
    ud_i = ud_traj[i]

    if feedback_mode == "LTV":
        K_i = K_traj[i]
    elif feedback_mode == "LTI":
        K_i = K_static
    elif feedback_mode == "pseudoLTV":
        A_i, B_i = system_matrices(t_i, xd_i, ud_i, sys_para)
        P_i = scilin.solve_continuous_are(A_i, B_i, Q, R)
        K_i = R_inv * B_i.T @ P_i

    u_i = ud_i - K_i @ (x_i - xd_i)
    dxdt_i = system_rhs(t_traj[i], x_i, u_i, sys_para)

    u_traj[i] = u_i

    if i < n_samples - 1:
        x_traj[i + 1] = x_i + sim_para.dt * dxdt_i

# plotting
plt.figure()

plt.subplot(211)
plt.plot(t_traj, x_traj)
plt.plot(t_traj, xd_traj, '--')
plt.legend(["x1", "x2", "x1d", "x2d"])

plt.subplot(212)
plt.plot(t_traj, u_traj)
plt.plot(t_traj, ud_traj, '--')
plt.legend(["u", "ud"])

plt.figure()

plt.subplot(211)
plt.plot(t_traj, Ptilde_triu_traj[::-1, :])
plt.legend(["p11", "p12", "p22"])

plt.subplot(212)
plt.plot(t_traj, K_traj.reshape((n_samples, sys_para.n)))
plt.legend(["k1", "k2"])

plt.show()