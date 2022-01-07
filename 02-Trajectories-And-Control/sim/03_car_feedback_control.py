# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan, arctan, arctan2
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.animation as mpla
from TrajGen import PolynomialTrajGen
from dataclasses import dataclass
from typing import Type, List

# Set up of ffmpeg for creating the animation
# Linux:
# Make sure ffmpeg is installed (e.g.: sudo apt install ffmpeg)
#
# Windows:
# 1. Download & install ffmepg from http://www.ffmpeg.org/download.html
# 2. Uncomment the following line and set path appropriately
#plt.rcParams['animation.ffmpeg_path'] = 'C:\\Progs\\ffmpeg\\bin\\ffmpeg.exe'


# Real Physical Parameters
@dataclass
class PhysPara:
    l: float = 0.3      # define car length
    w: float = l * 0.3  # define car width


# LISTING_START DefineControllerPara
@dataclass
class ControllerPara:
    k01: float = 3
    k02: float = 2
    k12: float = 10
    l: float = 0.5 * PhysPara.l
# LISTING_END DefineControllerPara


# Simulation Parameters
# LISTING_START DefineSimPara
@dataclass
class SimPara:
    t0: float = 0                   # start time
    tf: float = 10                  # final time
    dt: float = 0.04                # step-size
    tt = np.arange(0, tf + dt, dt)  # time vector
    x0d = [0, 0, 0]                 # reference inital state at t0
    xfd = [5, 5, 0]                 # reference final state at tf
    x0real = [0.25, 0.5, 0.25]      # real inital state at t0
# LISTING_END DefineSimPara


def setup_trajectories(sp: Type[SimPara]) -> List[PolynomialTrajGen]:
    """Setup the trajectory objects

    Args:
        sp(SimPara): The simulation parameter

    Returns:
        A list holding two objects of type PolynomialTrajGen. The first
        one is f, the second g.

    """

    # Start and final time of transistion
    t0: float = sp.t0 + 1
    tf: float = sp.tf - 1

    # boundary conditions for y1
    y1_a = np.array([sp.x0d[0], 0])
    y1_b = np.array([sp.xfd[0], 0])

    # boundary conditions for y2
    y2_a = np.array([sp.x0d[1], tan(sp.x0d[2]), 0])
    y2_b = np.array([sp.xfd[1], tan(sp.xfd[2]), 0])

    # From Y2A to Y2B on the interval [Y1A[0], Y1B[0]]
    f_traj = PolynomialTrajGen(y2_a, y2_b, y1_a[0], y1_b[0], 2)

    # from Y1A to Y1B on the interval [t0, tf]
    g_traj = PolynomialTrajGen(y1_a, y1_b, t0, tf, 1)

    return [f_traj, g_traj]


def ode(x, t, p: Type[PhysPara], cp: Type[ControllerPara]):
    """Function of the robots kinematics

    Args:
        x: state
        t: time
        p(PhysPara):        Real physical parameter
        cp(ControllerPara): Parameter used for simulation and control

    Returns:
        dxdt: state derivative
    """
    x1, x2, x3 = x  # state vector
    u1, u2 = control(x, t, cp)  # control vector

    # dxdt = f(x, u):
    dxdt = np.array([u1 * cos(x3),
                     u1 * sin(x3),
                     1 / p.l * u1 * tan(u2)])

    # return state derivative
    return dxdt


# LISTING_START ControlFunc
def control(x, t, p: Type[ControllerPara]):
    """Function of the control law

    Args:
        x (ndarray, int): state vector
        t (int): time
        p (ControllerPara): Parameter

    Returns:
        u (ndarry): control vector

    """
# LISTING_END ControlFunc

    # state vector
    y1 = x[0]
    y2 = x[1]
    theta = x[2]

    dy2 = sin(theta)

    # LISTING_START DefineRefTraj
    # reference trajectories yd, yd', yd''
    # evaluate the planned trajectories at time t
    g_t = g_traj_gen.eval(t)  # y1 = g(t)
    f_y1 = f_traj_gen.eval(g_t[0])  # y2 = f(y1) = f(g(t))

    y1d = g_t[0]
    dy1d = 1/(np.sqrt(1 + f_y1[1] ** 2))

    y2d = f_y1[0]
    dy2d = f_y1[1]/(np.sqrt(1 + f_y1[1] ** 2))
    ddy2d = f_y1[2]/(1 + f_y1[1] ** 2)
    # LISTING_END DefineRefTraj

    # LISTING_START CalcStabInputs
    # stabilizing inputs
    w1 = dy1d - p.k01 * (y1 - y1d)
    w2 = ddy2d - p.k12 * (dy2 - dy2d) - p.k02 * (y2 - y2d)
    # LISTING_END CalcStabInputs

    # LISTING_START ControlLaw
    # control laws
    ds = g_t[1] * np.sqrt(1 + (f_y1[1]) ** 2)  # desired velocity
    u1 = ds*np.sqrt(w1**2+dy2**2)
    u2 = arctan2(p.l * (w2 * w1), 1)

    return np.array([u1, u2]).T
    # LISTING_END ControlLaw


def plot_data(x, xref, u, t, fig_width, fig_height, save=False):
    """Plotting function of simulated state and actions

    Args:
        x(ndarray):     state-vector trajectory
        xref(ndarray):  reference state-vector trajectory
        u(ndarray):     control vector trajectory
        t(ndarray):     time vector
        fig_width:      figure width in cm
        fig_height:     figure height in cm
        save (bool) :   save figure (default: False)
    Returns: None

    """
    # creating a figure with 3 subplots, that share the x-axis
    fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    # set figure size to desired values
    fig1.set_size_inches(fig_width / 2.54, fig_height / 2.54)

    # plot y_1 in subplot 1
    ax1.plot(t, x[:, 0], label='$y_1(t)$', lw=1, color='r')
    ax1.plot(t, xref[:, 0], label='$y_{1,d}(t)$', lw=1, color=(0.5, 0, 0))

    # plot y_2 in subplot 1
    ax1.plot(t, x[:, 1], label='$y_2(t)$', lw=1, color='b')
    ax1.plot(t, xref[:, 1], label='$y_{2,d}(t)$', lw=1, color=(0, 0, 0.5))

    # plot theta in subplot 2
    ax2.plot(t, np.rad2deg(x[:, 2]), label=r'$\theta(t)$', lw=1, color=(0, 0.7, 0))
    ax2.plot(t, np.rad2deg(xref[:, 2]), label=r'$\theta_d(t)$', lw=1, color='g')

    # plot control in subplot 3, left axis red, right blue
    ax3.plot(t, u[:, 0], label=r'$v(t)$', lw=1, color='r')
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
        plt.savefig('state_trajectory.pgf')  # for easy export to LaTex
    return None


def car_animation(x, u, t, p: Type[PhysPara]):
    """Animation function of the car-like mobile robot

    Args:
        x(ndarray): state-vector trajectory
        u(ndarray): control vector trajectory
        t(ndarray): time vector
        p(object): Parameters

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

    def draw_the_car(cur_x, cur_u):
        """Mapping from state x and action cur_u to the position of the car elements

        Args:
            cur_x: The current state vector
            cur_u: The current action vector

        Returns:

        """
        wheel_length = 0.1 * p.l
        y1, y2, theta = cur_x
        v, phi = cur_u

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

    def init():
        """Initialize plot objects that change during animation.
           Only required for blitting to give a clean slate.

        Returns:

        """
        h_x_traj_plot.set_data([], [])
        h_car.set_data([], [])
        return h_x_traj_plot, h_car

    def animate(i):
        """Defines what should be animated

        Args:
            i: frame number

        Returns:

        """
        k = i % len(t)
        ax.set_title('Time (s): {:.2f}'.format(t[k]), loc='left')
        h_x_traj_plot.set_xdata(x[0:k, 0])
        h_x_traj_plot.set_ydata(x[0:k, 1])
        draw_the_car(x[k, :], u[k, :])
        return h_x_traj_plot, h_car

    ani = mpla.FuncAnimation(fig2, animate, init_func=init, frames=len(t) + 1,
                             interval=(t[1] - t[0]) * 1000,
                             blit=False)

    ani.save('animation.mp4', writer='ffmpeg', fps=1 / (t[1] - t[0]))
    plt.show()
    return None


# Setup the trajectories, we hold it as global objects here
[f_traj_gen, g_traj_gen] = setup_trajectories(SimPara)

# simulation
sol = sci.solve_ivp(lambda t, x: ode(x, t, PhysPara, ControllerPara), (0.0, SimPara.tf),
                    SimPara.x0real, method='RK45', t_eval=SimPara.tt)
x_traj = sol.y.T  # size(sol.y) = len(x)*len(tt) (.T -> transpose)
u_traj = np.zeros([len(SimPara.tt), 2])
for i in range(0, len(SimPara.tt)):
    u_traj[i] = control(x_traj[i], SimPara.tt[i], ControllerPara)

# get reference trajectories
y1D = g_traj_gen.eval_vec(SimPara.tt)
y2D = f_traj_gen.eval_vec(y1D[:, 0])

x_ref = np.zeros_like(x_traj)
x_ref[:, 0] = y1D[:, 0]
x_ref[:, 1] = y2D[:, 0]
x_ref[:, 2] = arctan(y2D[:, 1])

# plot result
plot_data(x_traj, x_ref, u_traj, SimPara.tt, 12, 16, save=True)

# animation
car_animation(x_traj, u_traj, SimPara.tt, PhysPara)

plt.show()
