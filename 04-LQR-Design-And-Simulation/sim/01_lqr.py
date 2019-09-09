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


def calc_system_matrices(x):
    x1, x2 = x
    A = np.array([[0, para.a * cos(x2)],
                  [-2 * x1, 0]])

    B = np.array([[0],
                  [1]])

    return A, B


def calc_trajectory_functions():
    planner = PolynomialPlanner([traj_para.y0, 0, 0], [traj_para.yf, 0, 0], traj_para.t0, 10, 2)

    def traj_fun(tt):
        if np.isscalar(tt):
            tt = np.array([tt])

        x1d_and_derivatives = np.array([list(planner.eval(t)) if t <= 10 else [traj_para.yf, 0, 0] for t in tt])

        x1d = x1d_and_derivatives[:, 0]
        x1d_dot = x1d_and_derivatives[:, 1]
        x1d_ddot = x1d_and_derivatives[:, 2]
        x2d = np.arcsin(x1d_dot / para.a)
        x2d_dot = x1d_ddot / np.sqrt(para.a ** 2 - np.square(x1d_dot))

        xd = np.stack((x1d, x2d), axis=1)
        ud = x2d_dot + np.square(x1d)

        if len(tt) == 1:
            xd = xd[0]
            ud = ud[0]

        return xd, ud

    return traj_fun


def calc_static_lqr(A, B, Q, R):
    P = scilin.solve_continuous_are(A, B, Q, R)
    K = 1 / R * B.T @ P

    return K

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

def calc_variant_lqr(Q, R, S):
    def riccati_rhs(tau, P_triu):
        P = triu_to_full(P_triu)

        xd, _ = traj_fun(traj_para.tf - tau)
        A, B = calc_system_matrices(xd)

        dP = (P @ B * 1/R) @ B.T @ P - A.T @ P - P @ A - Q

        dP_triu = full_to_triu(dP)
        return dP_triu

    t_traj = traj_para.t0 + np.arange(traj_para.n_samples) * traj_para.dt
    Ptf_triu = full_to_triu(S)

    sol = sci.solve_ivp(riccati_rhs, (traj_para.t0, traj_para.tf), Ptf_triu, t_eval=t_traj)
    P_triu_traj = sol.y.T[::-1, :]
    P_triu_interp = sciinterp.interp1d(t_traj, P_triu_traj, axis=0)

    def calc_feedback(t):
        P = triu_to_full(P_triu_interp(t))
        xd, _ = traj_fun(t)
        _, B = calc_system_matrices(xd)
        K = 1/R*B.T@P
        return K

    return calc_feedback


# Physical parameter
para = Parameters()  # instance of class Parameters
para.a = 1           # model parameter

# Simulation parameter
sim_para = Parameters()  # instance of class Parameters
sim_para.t0 = 0          # start time
sim_para.tf = 10         # final time
sim_para.dt = 0.01       # step-size
sim_para.n_samples = int((sim_para.tf - sim_para.t0) / sim_para.dt) + 1

# Controller parameters
Q = np.diag([1, 1])
R = 1

# time vector
tt = sim_para.t0 + np.arange(sim_para.n_samples) * sim_para.dt

# initial state
x0 = [-2.1, 0.2]

# Trajectory parameters
traj_para = Parameters()
traj_para.y0 = -2
traj_para.yf = 2
traj_para.t0 = 0
traj_para.tf = 20
traj_para.dt = 0.01
traj_para.n_samples = int((traj_para.tf - traj_para.t0) / traj_para.dt) + 1

traj_fun = calc_trajectory_functions()

Ad, Bd = calc_system_matrices(x0)

K = calc_static_lqr(Ad, Bd, Q, R)
print(K)
feedback_fun = calc_variant_lqr(Q, R, np.array([[-4.6, 0.123],
                                                [0.123, -1.12]]))

def ode(t, x):
    """Function of the robots kinematics

    Args:
        x        : state
        t        : time

    Returns:
        dxdt: state derivative
    """
    x1, x2 = x  # state vector
    u = control(t, x)  # control vector

    # dxdt = f(x, u):
    dxdt = np.array([para.a * sin(x2),
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
    xd, ud = traj_fun(t)

    K = feedback_fun(t)
    u = - K @ (x - xd) + ud

    return u


# simulation
sol = sci.solve_ivp(lambda t, x: ode(t, x), (sim_para.t0, sim_para.tf), x0, t_eval=tt)
x_traj = sol.y.T
u_traj = np.array([control(tt[i], x_traj[i]) for i in range(len(tt))])

xd_traj, ud_traj = traj_fun(tt)
K_traj = np.array([list(feedback_fun(t).flatten()) for t in tt])

# plotting
plt.figure()

plt.subplot(311)
plt.plot(tt, x_traj)
plt.plot(tt, xd_traj, '--')
plt.legend(['x1', 'x2', 'x1d', 'x2d'])
plt.grid()

plt.subplot(312)
plt.plot(tt, u_traj)
plt.plot(tt, ud_traj, '--')
plt.grid()

plt.subplot(313)
plt.plot(tt, K_traj)
plt.grid()

plt.show()