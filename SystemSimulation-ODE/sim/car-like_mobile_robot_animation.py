# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation


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


def car_animation(x, u, t, prmtrs):
    """Animation function of the car-like mobile robot

    Args:
        x(ndarray): state-vector trajectory
        u(ndarray): control vector trajectory
        t(ndarray): time vector
        prmtrs(object): paramers

    Returns: None

    """
    dx = 1.5 * prmtrs.l
    dy = 1.5 * prmtrs.l
    fig2, ax = plt.subplots()
    ax.set_xlim([min(min(x_traj[:, 0] - dx), -dx),
                 max(max(x_traj[:, 0] + dx), dx)])
    ax.set_ylim([min(min(x_traj[:, 1] - dy), -dy),
                 max(max(x_traj[:, 1] + dy), dy)])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')

    x_traj_plot, = ax.plot([], [], 'b')  # state trajectory in the y1-y2-plane
    car, = ax.plot([], [], 'k', lw=2)  # car

    def car_plot(x, u):
        """Mapping from state x and action u to the position of the car elements

        Args:
            x: state vector
            u: action vector

        Returns:
            car:

        """
        wheel_length = 0.1 * prmtrs.l
        y1, y2, theta = x
        v, phi = u

        # define chassis lines
        chassis_y1 = [y1, y1 + prmtrs.l * cos(theta)]
        chassis_y2 = [y2, y2 + prmtrs.l * sin(theta)]

        # define lines for the front and rear axle
        rear_ax_y1 = [y1 + prmtrs.w * sin(theta), y1 - prmtrs.w * sin(theta)]
        rear_ax_y2 = [y2 - prmtrs.w * cos(theta), y2 + prmtrs.w * cos(theta)]
        front_ax_y1 = [chassis_y1[1] + prmtrs.w * sin(theta + phi),
                       chassis_y1[1] - prmtrs.w * sin(theta + phi)]
        front_ax_y2 = [chassis_y2[1] - prmtrs.w * cos(theta + phi),
                       chassis_y2[1] + prmtrs.w * cos(theta + phi)]

        # define wheel lines
        rear_l_wl_y1 = [rear_ax_y1[1] + wheel_length * cos(theta),
                        rear_ax_y1[1] - wheel_length * cos(theta)]
        rear_l_wl_y2 = [rear_ax_y2[1] + wheel_length * sin(theta),
                        rear_ax_y2[1] - wheel_length * sin(theta)]
        rear_r_wl_y1 = [rear_ax_y1[0] + wheel_length * cos(theta),
                        rear_ax_y1[0] - wheel_length * cos(theta)]
        rear_r_wl_y2 = [rear_ax_y2[0] + wheel_length * sin(theta),
                        rear_ax_y2[0] - wheel_length * sin(theta)]
        front_l_wl_y1 = [front_ax_y1[1] + wheel_length * cos(theta + phi),
                         front_ax_y1[1] - wheel_length * cos(theta + phi)]
        front_l_wl_y2 = [front_ax_y2[1] + wheel_length * sin(theta + phi),
                         front_ax_y2[1] - wheel_length * sin(theta + phi)]
        front_r_wl_y1 = [front_ax_y1[0] + wheel_length * cos(theta + phi),
                         front_ax_y1[0] - wheel_length * cos(theta + phi)]
        front_r_wl_y2 = [front_ax_y2[0] + wheel_length * sin(theta + phi),
                         front_ax_y2[0] - wheel_length * sin(theta + phi)]

        # empty value (to disconnect points, define where no line should be plotted)
        empty = [np.nan, np.nan]

        # concatenate set of coordinates
        data_y1 = [rear_ax_y1, empty, front_ax_y1, empty, chassis_y1,
                   empty, rear_l_wl_y1, empty, rear_r_wl_y1,
                   empty, front_l_wl_y1, empty, front_r_wl_y1]
        data_y2 = [rear_ax_y2, empty, front_ax_y2, empty, chassis_y2,
                   empty, rear_l_wl_y2, empty, rear_r_wl_y2,
                   empty, front_l_wl_y2, empty, front_r_wl_y2]

        # set data
        car.set_data(data_y1, data_y2)
        return car,

    def init():
        """Initialize plot objects that change during animation.
           Only required for blitting to give a clean slate.

        Returns:

        """
        x_traj_plot.set_data([], [])
        car.set_data([], [])
        return x_traj_plot, car

    def animate(i):
        """Defines what should be animated

        Args:
            i: frame number

        Returns:

        """
        k = i % len(t)
        ax.set_title('Time (s): ' + str(t[k]), loc='left')
        x_traj_plot.set_xdata(x[0:k, 0])
        x_traj_plot.set_ydata(x[0:k, 1])
        car_plot(x[k, :], control(x[k, :], t[k]))
        return x_traj_plot, car

    ani = animation.FuncAnimation(fig2, animate, init_func=init,
                                  frames=len(t) + 1,
                                  interval=(t[1] - t[0]) * 1000,
                                  blit=False)

    ani.save('animation.mp4', writer='ffmpeg', fps=1 / (t[1] - t[0]))
    plt.show()
    return None


class Parameters(object):
    pass


prmtrs = Parameters()  # entity of class Parameters
prmtrs.l = 0.3  # define car length
prmtrs.w = prmtrs.l * 0.3  # define car width

t0 = 0  # start time
tend = 10  # end time
dt = 0.04  # step-size

# time vector
tt = np.arange(t0, tend, dt)

# initial state
x0 = [0, 0, 0]

# simulation
x_traj = odeint(ode, x0, tt, args=(prmtrs,))
u_traj = control(x_traj, tt)

# plot
plot_data(x_traj, u_traj, tt, 12, 8, save=True)

# animation
car_animation(x_traj, u_traj, tt, prmtrs)

plt.show()
