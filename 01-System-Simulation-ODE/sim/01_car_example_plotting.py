# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan
import scipy.integrate as sci
import matplotlib.pyplot as plt
from typing import Type
from dataclasses import dataclass


# LISTING_START ParaClass
# Physical parameter
@dataclass
class Para:
    l: float = 0.3    # define car length
    w: float = l*0.3  # define car width
# LISTING_END ParaClass


# LISTING_START SimuPara
# Simulation parameter
@dataclass
class SimPara:
    t0: float = 0          # start time
    tf: float = 10         # final time
    dt: float = 0.04       # step-size
# LISTING_END SimuPara


# LISTING_START OdeFunDef
def ode(t, x, p: Type[Para]):
    """Function of the robots kinematics

    Args:
        x        : state
        t        : time
        p(object): parameter container class

    Returns:
        dxdt: state derivative
    """
    x1, x2, x3 = x  # state vector
    u1, u2 = control(t)  # control vector

    # dxdt = f(x, u):
    dxdt = np.array([u1 * cos(x3),
                     u1 * sin(x3),
                     1 / p.l * u1 * tan(u2)])

    # return state derivative
    return dxdt
# LISTING_END OdeFunDef


# LISTING_START ControlFunDef
def control(t):
    """Function of the control law

    Args:
        t: time

    Returns:
        u: control vector

    """
    u1 = np.maximum(0, 1.0 - 0.1 * t)
    u2 = np.full(u1.shape, 0.25)
    return np.array([u1, u2]).T
# LISTING_END ControlFunDef


# LISTING_START PlotFunDef
def plot_data(x, u, t, fig_width, fig_height, save=False):
    """Plotting function of simulated state and actions

    Args:
        x(ndarray) : state-vector trajectory
        u(ndarray) : control vector trajectory
        t(ndarray) : time vector
        fig_width  : figure width in cm
        fig_height : figure height in cm
        save (bool): save figure (default: False)
    Returns: None

    """
    # creating a figure with 3 subplots, that share the x-axis
    fig1, (ax1, ax2, ax3) = plt.subplots(3)

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
    ax3.set_xlabel(r't in s')

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
        # plt.savefig('state_trajectory.pgf')  # for easy export to LaTeX, needs a lot of extra packages
    return None
# LISTING_END PlotFunDef


# LISTING_START Simulation
# time vector
tt = np.arange(SimPara.t0, SimPara.tf + SimPara.dt, SimPara.dt)

# initial state
x0 = [0, 0, 0]

# simulation
sol = sci.solve_ivp(lambda t, x: ode(t, x, Para), (SimPara.t0, SimPara.tf), x0, t_eval=tt)
x_traj = sol.y.T
u_traj = control(tt)
# LISTING_END Simulation

# LISTING_START PlotResult
# plot
plot_data(x_traj, u_traj, tt, 12, 16, save=True)
plt.show()
# LISTING_END PlotResult
