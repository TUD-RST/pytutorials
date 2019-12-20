# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan, pi
import scipy.linalg as scilin
import matplotlib.pyplot as plt
from Planner import PolynomialPlanner
from typing import Literal
import pickle


class Parameters(object):
    pass


# define simulation parameters
sim_para = Parameters()  # instance of class Parameters
sim_para.t0 = 0          # start time
sim_para.tf = 2.5        # final time
sim_para.dt = 0.01       # step-size
sim_para.x0 = [0, 0, np.pi, 0] # initial value

# already prepare the time vector because we'll need it very soon
n_samples = int((sim_para.tf - sim_para.t0) / sim_para.dt) + 1
t_sim = sim_para.t0 + np.arange(n_samples) * sim_para.dt

# -- START SYSTEM SPECIFIC PART --
# LISTING_START defsystem
# define system functions
sys_para = Parameters()  # instance of class Parameters
sys_para.n = 4           # number of states
sys_para.m = 1           # number of inputs
sys_para.g = 9.81        # model parameter
sys_para.l = 0.5


def system_rhs(t, x, u, para):
    x1, x2, x3, x4 = x  # state vector

    # dxdt = f(x, u):
    dxdt = np.array([x2,
                     u,
                     x4,
                     (1/para.l)*(para.g*sin(x3)+u*cos(x3))])

    # return state derivative
    return dxdt


def system_matrices(t, x, u, para):
    x1, x2, x3, x4 = x
    A = np.array([[0, 1, 0, 0],
                  [0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1/para.l*(para.g*cos(x3)-u*sin(x3)), 0]])

    B = np.array([[0],
                  [1],
                  [0],
                  [cos(x3)/para.l]])

    return A, B
# LISTING_END defsystem


# define controller parameters
Q = np.diag([1, 1, 1, 1])
R = np.array([[1]])

# LISTING_START loadcsv
pendulum_csv = np.loadtxt("pendulum.csv", delimiter=",")

t_traj = pendulum_csv[:, 0].flatten()
n_traj_samples = len(t_traj)
xd_traj = pendulum_csv[:, 1:5]
ud_traj = pendulum_csv[:, 5]
# LISTING_END loadcsv
# -- END SYSTEM SPECIFIC PART --

R_inv = scilin.inv(R)


# solve matrix riccati ODE
# LISTING_START triuconvert
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
# LISTING_END triuconvert


# LISTING_START riccatiinit
# Get initial value S for matrix Riccati ODE by solving algebraic Riccati equation
A_f, B_f = system_matrices(t_traj[-1], xd_traj[-1], ud_traj[-1], sys_para)
S = scilin.solve_continuous_are(A_f, B_f, Q, R)
# LISTING_END riccatiinit
# LISTING_START riccatiint
Pbar_triu_traj = np.empty((n_traj_samples, int(sys_para.n*(sys_para.n + 1)/2)))  # allocate array for P
Pbar_triu_traj[0, :] = full_to_triu(S)  # initialize with Pbar(0) = S

K_traj = np.empty((n_traj_samples, sys_para.m, sys_para.n))  # allocate array for K

# get trajectories for P and K via numerical integration
for i_tau in range(n_traj_samples):  # iterate forward in tau direction
    i_t = n_traj_samples - 1 - i_tau  # index in t vector (counting down from last element)
    t_i = t_traj[i_t]
    tau_i = t_traj[i_tau]
    xd_i = xd_traj[i_t]
    ud_i = ud_traj[i_t]
    A_i, B_i = system_matrices(t_i, xd_i, ud_i, sys_para)
    Pbar_triu_i = Pbar_triu_traj[i_tau]  # indices for Pbar run forward in tau direction
    Pbar_i = triu_to_full(Pbar_triu_i)

    K_traj[i_t] = R_inv @ B_i.T @ Pbar_i

    if i_tau < n_traj_samples - 1:  # are we at the end yet? if not, compute next Pbar via numerical integration
        dPbar_dtau = - Pbar_i @ B_i @ R_inv @ B_i.T @ Pbar_i + Pbar_i @ A_i + A_i.T @ Pbar_i + Q
        dPbar_dtau_triu = full_to_triu(dPbar_dtau)

        Pbar_triu_traj[i_tau + 1] = Pbar_triu_i + sim_para.dt * dPbar_dtau_triu  # one Euler step
# LISTING_END riccatiint
# LISTING_START linsys
# compute static LQR feedback
t_static = 2
i_static = 0
while i_static < len(t_traj) - 1 and t_traj[i_static] < t_static:  # find index of that time in time vector
    i_static += 1
x_static = xd_traj[i_static]
ud_static = ud_traj[i_static]
A_static, B_static = system_matrices(t_static, x_static, ud_static, sys_para)
# LISTING_END linsys
# LISTING_START solveare
P_static = scilin.solve_continuous_are(A_static, B_static, Q, R)
K_static = R_inv * B_static.T @ P_static
# LISTING_END solveare

# LISTING_START sim
# main simulation loop
x_sim = np.empty((n_samples, sys_para.n))  # allocate array for state over time
x_sim[0] = sim_para.x0  # set initial state
u_sim = np.empty((n_samples, sys_para.m))  # allocate array for input over time
xd_log = np.empty((n_samples, sys_para.n))
ud_log = np.empty((n_samples, sys_para.m))
K_log = np.empty((n_samples, sys_para.m, sys_para.n))  # allocate array for feedback over time

FeedbackMode = Literal["LTV", "LTI", "pseudoLTV"]
feedback_mode: FeedbackMode = "LTV"

feedback_only_after_swingup = False

for i in range(n_samples):
    t_i = t_sim[i]
    x_i = x_sim[i]

    i_traj = min(i, n_traj_samples - 1)
    xd_i = xd_traj[i_traj]
    ud_i = ud_traj[i_traj]

    # switch between controller types
    if feedback_mode == "LTV":
        K_i = K_traj[i_traj]  # read feedback matrix from pre-computed Riccati solution
    elif feedback_mode == "LTI":
        K_i = K_static
    elif feedback_mode == "pseudoLTV":
        A_i, B_i = system_matrices(t_i, xd_i, ud_i, sys_para)
        P_i = scilin.solve_continuous_are(A_i, B_i, Q, R)  # retune feedback for current state from reference trajectory
        K_i = R_inv * B_i.T @ P_i

    if feedback_only_after_swingup and i < n_traj_samples:
        u_i = ud_i
    else:
        u_i = ud_i - K_i @ (x_i - xd_i)  # the actual control law u_tilde=-K*x_tilde

    u_sim[i] = u_i
    xd_log[i] = xd_i
    ud_log[i] = ud_i
    K_log[i] = K_i

    if i < n_samples - 1:  # have we reached the end yet? if not, integrate one step
        dxdt_i = system_rhs(t_sim[i], x_i, u_i, sys_para)
        x_sim[i + 1] = x_i + sim_para.dt * dxdt_i
# LISTING_END sim

# storing the results
store_dict = dict(t=t_sim, x=x_sim, xd=xd_log, u=u_sim, ud=ud_log, K=K_log, P_triu=Pbar_triu_traj[::-1, :])
pickle.dump(store_dict, open("log.p", "wb"))

# plotting
plt.figure()

plt.subplot(211)
plt.plot(t_sim, x_sim)
plt.plot(t_sim, xd_log, '--')

plt.subplot(212)
plt.plot(t_sim, u_sim)
plt.plot(t_sim, ud_log, '--')

plt.figure()

plt.plot(t_sim, K_log.reshape((n_samples, sys_para.n)))

plt.show()