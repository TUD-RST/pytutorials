# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan
import scipy.integrate as sci
import matplotlib.pyplot as plt


class Parameters(object):
    pass


# Physical parameter
para = Parameters()  # instance of class Parameters
para.l = 0.3         # define car length
para.w = para.l*0.3  # define car width

# Simulation parameter
sim_para = Parameters()  # instance of class Parameters
sim_para.t0 = 0          # start time
sim_para.tend = 10       # end time
sim_para.dt = 0.04       # step-size


def ode(x, t, p):
    """Function of the robots kinematics

    Args:
        x: state
        t: time
        p(object): parameter container class

    Returns:
        dxdt: state derivative
    """
    x1, x2, x3 = x  # state vector
    u1, u2 = control(x, t)  # control vector

    # dxdt = f(x, u):
    dxdt = np.array([u1 * cos(x3),
                     u1 * sin(x3),
                     1 / p.l * u1 * tan(u2)])

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
    u1 = np.maximum(0, 0.5 - 0.06*t)
    u2 = np.full(u1.shape, 0.25)
    return np.array([u1, u2]).T


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
    # creating a figure with 3 subplots, that share the x-axis
    fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    # set figure size to desired values
    fig1.set_size_inches(fig_width / 2.54, fig_height / 2.54)

    # plot y_1 in subplot 1
    ax1.plot(t, x[:, 0], label='$y_1(t)$', lw=1, color='r')

    # plot y_2 in subplot 1
    ax1.plot(t, x[:, 1], label='$y_2(t)$', lw=1, color='b')

    # plot theta in subplot 2
    ax2.plot(t, np.rad2deg(x[:, 2]), label=r'$\theta(t)$', lw=1, color='g')

    # plot control in subplot 3, left axis red, right blue
    ax3.plot(t, np.rad2deg(u[:, 0]), label=r'$v(t)$', lw=1, color='r')
    ax3.tick_params(axis='y', colors='r')
    ax33 = ax3.twinx()
    ax33.plot(t, np.rad2deg(u[:, 1]), label=r'$\phi(t)$', lw=1, color='b')
    ax33.spines["left"].set_color('r')
    ax33.spines["right"].set_color('b')
    ax33.tick_params(axis='y', colors='b')

    # Grids
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    # set the labels on the x and y axis and the titles
    ax1.set_title('Position coordinates')
    ax1.set_ylabel(r'm')
    ax1.set_xlabel(r't in s')
    ax2.set_title('Orientation')
    ax2.set_ylabel(r'deg')
    ax2.set_xlabel(r't in s')
    ax3.set_title('Velocity / steering angle')
    ax3.set_ylabel(r'm/s')
    ax33.set_ylabel(r'deg')
    ax33.set_xlabel(r't in s')

    # put a legend in the plot
    ax1.legend()
    ax2.legend()
    ax3.legend()
    li3, lab3 = ax3.get_legend_handles_labels()
    li33, lab33 = ax33.get_legend_handles_labels()
    ax3.legend(li3 + li33, lab3 + lab33, loc=0)

    # automatically adjusts subplot to fit in figure window
    plt.tight_layout()

    # save the figure in the working directory
    if save:
        plt.savefig('state_trajectory.pdf')  # save output as pdf
        plt.savefig('state_trajectory.pgf')  # for easy export to LaTex
    return None


# time vector
tt = np.arange(sim_para.t0, sim_para.tend + sim_para.dt, sim_para.dt)

# initial state
x0 = [0, 0, 0]

# simulation
x_traj = sci.odeint(ode, x0, tt, args=(para, ))
u_traj = control(x_traj, tt)

# plot
plot_data(x_traj, u_traj, tt, 12, 16, save=True)

plt.show()