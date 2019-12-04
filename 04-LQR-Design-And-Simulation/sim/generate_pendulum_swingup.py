"""
Adapted from PyTrajectory's ex0_InvertedPendulumSwingUp
"""
from pytrajectory import TransitionProblem
import numpy as np
from sympy import cos, sin
from numpy import pi

import sys
import matplotlib as mpl
from pytrajectory.visualisation import Animation


def f(xx, uu, uuref, t, pp):
    """ Right hand side of the vectorfield defining the system dynamics

    :param xx:       state
    :param uu:       input
    :param uuref:    reference input (not used)
    :param t:        time (not used)
    :param pp:       additionial free parameters  (not used)

    :return:        xdot
    """
    x1, x2, x3, x4 = xx  # system variables
    u1, = uu             # input variable

    l = 0.5     # length of the pendulum
    g = 9.81    # gravitational acceleration

    # this is the vectorfield
    ff = [x2,
          u1,
          x4,
          (1/l)*(g*sin(x3)+u1*cos(x3))]

    return ff


# then we specify all boundary conditions
a = 0.0
xa = [0.0, 0.0, pi, 0.0]

b = 2.0
xb = [0.0, 0.0, 0.0, 0.0]

ua = [0.0]
ub = [0.0]

from pytrajectory import log
log.console_handler.setLevel(20)

# now we create our Trajectory object and alter some method parameters via the keyword arguments

first_guess = {'seed': 20}
S = TransitionProblem(f, a, b, xa, xb, ua, ub, first_guess=first_guess, kx=2, eps=5e-2, use_chains=False, sol_steps=1300)

# time to run the iteration
S.solve()


# now that we (hopefully) have found a solution,
# we can visualise our systems dynamic

# therefore we define a function that draws an image of the system
# according to the given simulation data
def draw(xt, image):
    # to draw the image we just need the translation `x` of the
    # cart and the deflection angle `phi` of the pendulum.
    x = xt[0]
    phi = xt[2]

    # next we set some parameters
    car_width = 0.05
    car_heigth = 0.02

    rod_length = 0.5
    pendulum_size = 0.015

    # then we determine the current state of the system
    # according to the given simulation data
    x_car = x
    y_car = 0

    x_pendulum = -rod_length * sin(phi) + x_car
    y_pendulum = rod_length * cos(phi)

    # now we can build the image

    # the pendulum will be represented by a black circle with
    # center: (x_pendulum, y_pendulum) and radius `pendulum_size
    pendulum = mpl.patches.Circle(xy=(x_pendulum, y_pendulum), radius=pendulum_size, color='black')

    # the cart will be represented by a grey rectangle with
    # lower left: (x_car - 0.5 * car_width, y_car - car_heigth)
    # width: car_width
    # height: car_height
    car = mpl.patches.Rectangle((x_car-0.5*car_width, y_car-car_heigth), car_width, car_heigth,
                                fill=True, facecolor='grey', linewidth=2.0)

    # the joint will also be a black circle with
    # center: (x_car, 0)
    # radius: 0.005
    joint = mpl.patches.Circle((x_car,0), 0.005, color='black')

    # and the pendulum rod will just by a line connecting the cart and the pendulum
    rod = mpl.lines.Line2D([x_car,x_pendulum], [y_car,y_pendulum],
                            color='black', zorder=1, linewidth=2.0)

    # finally we add the patches and line to the image
    image.patches.append(pendulum)
    image.patches.append(car)
    image.patches.append(joint)
    image.lines.append(rod)

    # and return the image
    return image


# now we can create an instance of the `Animation` class
# with our draw function and the simulation results

# save the simulation data (solution of IVP) to csv
tt, xx, uu = S.sim_data
export_array = np.hstack((tt.reshape(-1, 1), xx, uu))
np.savetxt("pendulum.csv", export_array, delimiter=",")


# first column: time
# next n columns: state (here n = 4)
# last m columns: input (here m = 1)

# to plot the curves of some trajectories along with the picture
# we also pass the appropriate lists as arguments (see documentation)
A = Animation(drawfnc=draw, simdata=S.sim_data,
              plotsys=[(0,'x'), (2,'phi')], plotinputs=[(0,'u')])

# as for now we have to explicitly set the limits of the figure
# (may involves some trial and error)
xmin = np.min(S.sim_data[1][:,0]); xmax = np.max(S.sim_data[1][:,0])
A.set_limits(xlim=(xmin - 0.5, xmax + 0.5), ylim=(-0.6,0.6))

A.animate()

# then we can save the animation as a `mp4` video file or as an animated `gif` file
A.save('pendulum.mp4')
