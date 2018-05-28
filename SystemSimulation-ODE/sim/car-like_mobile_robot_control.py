# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan, arctan, pi
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation
import path_planner


def ode(t, x, prmtrs):
    """Function of the robots kinematics

    Args:
        x: state
        t: time
        prmtrs(object): parameter container class

    Returns:
        dxdt: state derivative
    """
    x1, x2, x3, x4 = x  # state vector
    u1, u2 = control(x, t)  # control vector
    # dxdt = f(x, u)
    dxdt = np.array([x4 * cos(x3),
                     x4 * sin(x3),
                     1 / prmtrs.l * x4 * tan(u2),
                     u1])

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
    yd = path_planner.path(y0, yend, t0, tend, gamma, t)
    y1d = yd[0, 0]
    y2d = yd[0, 1]
    dy1d = yd[0, 2]
    dy2d = yd[0, 3]
    ddy1d = yd[0, 4]
    ddy2d = yd[0, 5]
    y1 = x[0]
    y2 = x[1]
    v = path_planner.abssign(x[3])*max(abs(x[3]),0.001)
    k02 =  -1e3
    k12 =  0
    k01 = k02
    k11 = k12
    dy1 = v*cos(x[2])
    dy2 = v*sin(x[2])
    w1 = ddy2d - k12 * (dy2 - dy2d) - k02 * (y2 - y2d)
    w2 = ddy1d - k11 * (dy1 - dy1d) - k01 * (y1 - y1d)
    u2 = arctan(prmtrs.l * (w2 * dy2 - dy1 * w1) / (dy1 ** 2 + dy2 ** 2) ** (3. / 2))
    u2 = min(abs(u2),1.5)*np.sign(u2)
    u1 = 1 / prmtrs.l * (v ** 2) * tan(x[2]) * tan(u2) - w2 / cos(x[2])
    u = [u1, u2]
    return u


def plot_data(x, xref, u, t, fig_width, fig_height, save=False):
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
    fig1, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)

    # set figure size to desired values
    fig1.set_size_inches(fig_width / 2.54, fig_height / 2.54)

    # plot y_1 in subplot 1
    ax1.plot(t, x[:, 0], label='$y_1(t)$', lw=1, color='b')
    ax1.plot(t, xref[:, 0], label='$y_{1,d}(t)$', lw=1, color='r')

    # plot y_2 in subplot 1
    ax2.plot(t, x[:, 1], label='$y_2(t)$', lw=1, color='b')
    ax2.plot(t, xref[:, 1], label='$y_{2,d}(t)$', lw=1, color='r')

    # plot theta in subplot 2
    ax3.plot(t, x[:, 2], label=r'$\theta(t)$', lw=1, color='g')

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    # set the labels on the x and y axis in subplot 1
    ax1.set_ylabel(r'm')
    ax1.set_xlabel(r't in s')
    ax2.set_ylabel(r'm')
    ax2.set_xlabel(r't in s')
    ax3.set_ylabel(r'rad')
    ax3.set_xlabel(r't in s')

    # put a legend in the plot
    ax1.legend()
    ax2.legend()
    ax3.legend()

    # automatically adjusts subplot to fit in figure window
    plt.tight_layout()

    # save the figure in the working directory
    if save:
        plt.savefig('state_trajectory.pdf')  # save output as pdf
        plt.savefig('state_trajectory.pgf')  # for easy export to LaTex
    return None


def car_animation(x, xref, u, t, prmtrs):
    """Animation function of the car-like mobile robot

    Args:
        x(ndarray): state-vector trajectory
        u(ndarray): control vector trajectory
        t(ndarray): time vector
        prmtrs(object): parameters

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
    x_ref_plot, = ax.plot([], [], 'r')  # reference trajectory in the y1-y2-plane

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
        x_ref_plot.set_data([], [])
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
        x_ref_plot.set_xdata(xref[0:k, 0])
        x_ref_plot.set_ydata(xref[0:k, 1])
        car_plot(x[k, :], u[k, :])
        return x_traj_plot, x_ref_plot, car

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
prmtrs.traj = []
prmtrs.u = []
prmtrs.t = []
prmtrs.y1 = [0]
prmtrs.t = [0]

t0 = 0  # start time
tend = 10  # end time
dt = 0.04  # step-size

# time vector
tt = np.arange(t0, tend, dt)

# initial state
x0 = [0, 0.0, 0.0, 0.001, 0.0]

# y1, y2, theta, v, phi
xend = [5.0, 0.0, 0.0, 0.001, 0.0]
dv = 0
dy10 = x0[3]*cos(x0[2])
dy1end = xend[3]*cos(xend[2])
dy20 = x0[3]*sin(x0[2])
dy2end = xend[3]*sin(xend[2])
ddy10 = dv*cos(x0[0])-x0[3]**2*tan(x0[4])*sin(x0[2])/prmtrs.l
ddy1end = dv*cos(xend[0])-xend[3]**2*tan(xend[0])*sin(xend[2])/prmtrs.l
ddy20 = dv*sin(x0[0])+x0[3]**2*tan(x0[4])*cos(x0[2])/prmtrs.l
ddy2end = dv*sin(xend[0])+xend[3]**2*tan(xend[0])*cos(xend[2])/prmtrs.l

gamma = 2

y0 = [x0[0:2],[dy10, dy20], [ddy10, ddy20]]
yend = [xend[0:2],[dy1end, dy2end], [ddy1end, ddy2end]]
# simulation
sol = solve_ivp(lambda t, x: ode(t, x, prmtrs), (t0, tend), x0[0:4], method='RK45',t_eval=tt)
x_traj = sol.y.T  # size=len(x)*len(t)
x_traj = x_traj[:,0:3]
u_traj = np.zeros([len(tt), 2])

for i in range(0, len(tt)):
    u_traj[i,:] = control(sol.y[:,i], tt[i])
yd = path_planner.path(y0, yend, t0, 5, gamma, tt)
x_ref = yd[:,0:2]

u_traj[:,0]=sol.y.T[:,-1]

# plot
plot_data(x_traj, x_ref, u_traj, sol.t, 12, 8, save=False)
# animation
car_animation(x_traj, x_ref, u_traj, sol.t, prmtrs)
#plt.plot(sol.t,sol.y.T)

plt.show()

print(x_traj[:,0])