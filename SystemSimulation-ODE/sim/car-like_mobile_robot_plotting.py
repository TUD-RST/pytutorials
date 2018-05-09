# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def ode(x, t, prmtrs):
    """Function of the robots kinematics

    Args:
        x: state
        t: time
        prmtrs(object): parameter container class

    Returns:
        dxdt: state derivative
    """
    x1, x2, x3 = x  # state vector
    u1, u2 = control(x, t)  # control vector
    # dxdt = f(x, u)
    dxdt = np.array([u1 * cos(x3),
                     u1 * sin(x3),
                     1 / prmtrs.l * u1 * tan(u2)])

    # return state derivative
    return dxdt


def control(x, t):
    """Function of the control law

    Args:
        x: state vector
        t: time

    Returns:
        u: control vector

    """
    u = [1, 0.25]
    return u


def plot_data(x, u, t, fig_width, fig_height, save=False):
    """Plotting function of simulated state and actions

    Args:
        x(ndarray): state-vector trajectory
        u(ndarray): control vector trajectory
        t(ndarray): time vector
        fig_width: figure width in cm
        fig_height: figure height in cm
        save (bool) : save figure (default: False)
    Returns: None

    """
    # creating a figure with 2 subplots, that share the x-axis
    fig1, (ax1, ax2) = plt.subplots(2, sharex=True)

    # set figure size to desired values
    fig1.set_size_inches(fig_width / 2.54, fig_height / 2.54)

    # plot y_1 in subplot 1
    ax1.plot(t, x[:, 0], label='$y_1(t)$', lw=1, color='r')

    # plot y_2 in subplot 1
    ax1.plot(t, x[:, 1], label='$y_2(t)$', lw=1, color='b')

    # plot theta in subplot 2
    ax2.plot(t, x[:, 2], label=r'$\theta(t)$', lw=1, color='g')

    ax1.grid(True)
    ax2.grid(True)
    # set the labels on the x and y axis in subplot 1
    ax1.set_ylabel(r'm')
    ax1.set_xlabel(r't in s')
    ax2.set_ylabel(r'rad')
    ax2.set_xlabel(r't in s')

    # put a legend in the plot
    ax1.legend()
    ax2.legend()

    # automatically adjusts subplot to fit in figure window
    plt.tight_layout()

    # save the figure in the working directory
    if save:
        plt.savefig('state_trajectory.pdf')  # save output as pdf
        plt.savefig('state_trajectory.pgf')  # for easy export to LaTex
    return None


class Parameters(object):
    pass

prmtrs = Parameters()  # entity of class Parameters
prmtrs.l = 0.3  # define car length
prmtrs.w = prmtrs.l*0.3  # define car width

t0 = 0  # start time
tend = 10  # end time
dt = 0.04  # step-size

# time vector
tt = np.arange(t0, tend, dt)

# initial state
x0 = [0, 0, 0]

# simulation
x_traj = odeint(ode, x0, tt, args=(prmtrs, ))
u_traj = control(x_traj,tt)

# plot
plot_data(x_traj,u_traj,tt,12, 8, save=True)

plt.show()