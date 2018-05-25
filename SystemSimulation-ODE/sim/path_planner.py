# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation


def path(y0, yend, t0, tend, gamma, t):

    """

    Args:
        y0(ndarray):
        yend(ndarray):
        t0:
        tend:
        gamma:
        t(ndarray):

    Returns: yd

    """

    yd = np.zeros([len(t), len(y0)*(gamma + 1)])

    for k in range(0,len(t)):
        phi = prototype_fct((t[k] - t0) / (tend - t0), gamma)
        if t[k] < t0:
            yd[k, 0] = y0

        elif t[k] > tend:
            yd[k, 0] = yend

        else:
            yd[k, 0] = y0 + (yend - y0) * phi[0]
            for i in range(1, gamma + 1):
                yd[k, i] = (1/(tend-t0))**i * (y0 - yend) * phi[i]
    return yd


def prototype_fct(t, gamma):

    """Prototype function, that is used in the path planner and returns a polynomial and its derivatives up to order gamma.

    Args:
        t: time
        gamma: differential index
    Returns: phi (vector of phi and its successive derivatives)

    """
    phi = np.zeros([gamma + 1, 1])

    summation = sum([bin_coeff(gamma, k) * (-1) ** k * t ** (k + gamma + 1) / (gamma + k + 1)
                     for k in range(0, gamma + 1)])
    phi[0] = faculty(2 * gamma + 1) / (faculty(gamma) ** 2) * summation

    # calculate it's derivatives up to order (gamma-1)

    for p in range(1, gamma + 1):
        summation = sum(
            [bin_coeff(gamma, k) * (-1) ** k * t ** (k + gamma + 1 - p) / (gamma + k + 1) * prod_iter(gamma, k, p)
             for k in range(0, gamma + 1)])
        phi[p] = faculty(2 * gamma + 1) / (faculty(gamma) ** 2) * summation

    return phi


def faculty(x):
    result = 1
    for i in range(2, x + 1):
        result *= i
    return result


def bin_coeff(n, k):
    result = faculty(n) / (faculty(k) * faculty(n - k))
    return result


def prod_iter(gamma, k, p):
    result = 1
    for i in range(1, p + 1):
        result *= (gamma + k + 2 - i)
    return result


def test(t):
    phi = np.zeros([4, 1])
    phi[0] = 35 * t ** 4 - 84 * t ** 5 + 70 * t ** 6 - 20 * t ** 7
    phi[1] = 140 * t ** 3 - 420 * t ** 4 + 420 * t ** 5 - 140 * t ** 6
    phi[2] = 420 * t ** 2 - 1680 * t ** 3 + 2100 * t ** 4 - 840 * t ** 5
    phi[3] = 840 * t ** 1 - 5040 * t ** 2 + 8400 * t ** 3 - 4200 * t ** 4
    result = np.sum(prototype_fct(t, 3) - phi) < 1e-10
    return result


if test(0) and test(0.5) and test(1):
    print("Test passed")


t = np.linspace(0,1,101)
print(t)
y0 = np.array([0])
yend = np.array([1])
t0 = t[0]
tend = t[-1]
gamma = 2
traj = path(y0,yend,t0,tend,gamma,t)
print(traj)
plt.plot(t,traj[:,0])
plt.plot(t,traj[:,1])
plt.plot(t,traj[:,2])
plt.show()