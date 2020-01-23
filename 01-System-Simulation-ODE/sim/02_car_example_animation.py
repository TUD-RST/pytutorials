# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.animation as mpla
plt.rcParams['animation.ffmpeg_path'] = 'C:\\Progs\\ffmpeg\\bin\\ffmpeg.exe'


class Parameters(object):
    pass


# Physical parameter
para = Parameters()  # instance of class Parameters
para.l = 0.3         # define car length
para.w = para.l*0.3  # define car width

# Simulation parameter
sim_para = Parameters()  # instance of class Parameters
sim_para.t0 = 0          # start time
sim_para.tf = 10         # final time
sim_para.dt = 0.04       # step-size


def ode(t, x, p):
    """Function of the robots kinematics

    Args:
        x: state
        t: time
        p(object): parameter container class

    Returns:
        dxdt: state derivative
    """
    x1, x2, x3 = x  # state vector
    u1, u2 = control(t, x)  # control vector

    # dxdt = f(x, u):
    dxdt = np.array([u1 * cos(x3),
                     u1 * sin(x3),
                     1 / p.l * u1 * tan(u2)])

    # return state derivative
    return dxdt


def control(t, x):
    """Function of the control law

    Args:
        x: state vector
        t: time

    Returns:
        u: control vector

    """
    u1 = np.maximum(0, 1.0 - 0.1*t)
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
        #plt.savefig('state_trajectory.pgf')  # for easy export to LaTeX, needs a lot of extra packages
    return None


# LISTING_START CarAnimFunDef
def car_animation(x, u, t, p):
    """Animation function of the car-like mobile robot

    Args:
        x(ndarray): state-vector trajectory
        u(ndarray): control vector trajectory
        t(ndarray): time vector
        p(object): parameters

    Returns: None

    """
    # Setup two empty axes with enough space around the trajectory so the car
    # can always be completely plotted. One plot holds the sketch of the car,
    # the other the curve
    dx = 1.5 * p.l
    dy = 1.5 * p.l
    fig2, ax = plt.subplots()
    ax.set_xlim([min(min(x_traj[:, 0] - dx), -dx),
                 max(max(x_traj[:, 0] + dx), dx)])
    ax.set_ylim([min(min(x_traj[:, 1] - dy), -dy),
                 max(max(x_traj[:, 1] + dy), dy)])
    ax.set_aspect('equal')
    ax.set_xlabel(r'$y_1$')
    ax.set_ylabel(r'$y_2$')

    # Axis handles
    h_x_traj_plot, = ax.plot([], [], 'b')  # state trajectory in the y1-y2-plane
    h_car, = ax.plot([], [], 'k', lw=2)    # car
# LISTING_END CarAnimFunDef

# LISTING_START CarPlotFunDef
    def car_plot(x, u):
        """Mapping from state x and action u to the position of the car elements

        Args:
            x: state vector
            u: action vector

        Returns:

        """
        wheel_length = 0.1 * p.l
        y1, y2, theta = x
        v, phi = u

        # define chassis lines
        chassis_y1 = [y1, y1 + p.l * cos(theta)]
        chassis_y2 = [y2, y2 + p.l * sin(theta)]

        # define lines for the front and rear axle
        rear_ax_y1 = [y1 + p.w * sin(theta), y1 - p.w * sin(theta)]
        rear_ax_y2 = [y2 - p.w * cos(theta), y2 + p.w * cos(theta)]
        front_ax_y1 = [chassis_y1[1] + p.w * sin(theta + phi),
                       chassis_y1[1] - p.w * sin(theta + phi)]
        front_ax_y2 = [chassis_y2[1] - p.w * cos(theta + phi),
                       chassis_y2[1] + p.w * cos(theta + phi)]

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
        h_car.set_data(data_y1, data_y2)
# LISTING_END CarPlotFunDef

# LISTING_START InitFunDef
    def init():
        """Initialize plot objects that change during animation.
           Only required for blitting to give a clean slate.

        Returns:

        """
        h_x_traj_plot.set_data([], [])
        h_car.set_data([], [])
        return h_x_traj_plot, h_car
# LISTING_END InitFunDef        

# LISTING_START AnimateFunDef
    def animate(i):
        """Defines what should be animated

        Args:
            i: frame number

        Returns:

        """
        k = i % len(t)
        ax.set_title('Time (s): ' + '%.2f' % t[k], loc='left')
        h_x_traj_plot.set_xdata(x[0:k, 0])
        h_x_traj_plot.set_ydata(x[0:k, 1])
        car_plot(x[k, :], control(t[k], x[k, :]))
        return h_x_traj_plot, h_car
# LISTING_END AnimateFunDef        

# LISTING_START DoAnimate
    ani = mpla.FuncAnimation(fig2, animate, init_func=init, frames=len(t) + 1,
                             interval=(t[1] - t[0]) * 1000,
                             blit=False)

    file_format = 'mp4'
    ani.save('animation.'+file_format, writer='ffmpeg', fps=1 / (t[1] - t[0]))
# LISTING_END DoAnimate

    plt.show()
    return None


# LISTING_START DoSimulate
# time vector
tt = np.arange(sim_para.t0, sim_para.tf + sim_para.dt, sim_para.dt)

# initial state
x0 = [0, 0, 0]

# simulation
sol = sci.solve_ivp(lambda t, x: ode(t, x, para), (sim_para.t0, sim_para.tf), x0, t_eval=tt)
x_traj = sol.y.T
u_traj = control(tt, x_traj)

# plot
plot_data(x_traj, u_traj, tt, 12, 16, save=True)

# animation
car_animation(x_traj, u_traj, tt, para)

plt.show()
# LISTING_END DoSimulate