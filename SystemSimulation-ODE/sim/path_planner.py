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
        t(ndarray,int):

    Returns: yd

    """
    if isinstance(y0, (list, np.ndarray)):
        n = len(y0[0])
    else:
        n = 1
        y0 = [y0]

    if isinstance(t, (list, np.ndarray)):
        m = len(t)
    else:
        m = 1
        t = [t]

    y0 = np.matrix(y0).T
    yend = np.matrix(yend).T
    yd = np.matrix(np.zeros([m, n*(gamma + 1)]))
    for k in range(0,m):
        phi = prototype_fct((t[k] - t0) / (tend - t0), gamma + 1)
        if t[k] < t0:
            yd[k, 0:n] = y0[:,0].T
            for i in range(1, gamma + 1):
                yd[k, i*n:i*n+2] = y0[:,i].T
        elif t[k] > tend:
            yd[k, 0:n] = yend[:,0].T
            for i in range(1, gamma + 1):
                yd[k, i*n:i*n+2] = yend[:,i].T
        else:
            #yd[k, 0:n] = (y0[:,0] + (yend[:,0] - y0[:,0]) * phi[0]).T
            for i in range(0, gamma + 1):
                yd[k, i*n:i*n+2] = y0[:,i].T + sum((bin_coeff(i,j) * (yend[:,i-j] - y0[:,i-j]) * (1/(tend-t0))**j * phi[j]).T for j in range(0, i + 1))
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

def abssign(x):
    if np.sign(x) == 0:
        return 1
    else:
        return np.sign(x)
#if test(0) and test(0.5) and test(1):
#    print("Test passed")


#t = np.linspace(0,1,101)
#y0 = [0.1, 0]
#yend = [1, 0.5]
#t0 = t[0]
#tend = t[-1]
#gamma = 3
#traj = path(y0,yend,t0,tend,gamma,t)
#plt.plot(t,traj[:,])
#plt.show()