# -*- coding: utf-8 -*-
import numpy as np
from numpy import cos, sin, tan
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import animation


def prototype_fct(t, gamma):
    """Prototype function, that is needed for trajectory planning

    Args:
        t: time
        gamma: differential index
    Returns: phi (vector of phi and its successive derivatives)

    """
    phi = []

    summation = sum([bin_coeff(gamma, k) * (-1) ** k * t ** (k + gamma + 1) / (gamma + k + 1)
                     for k in range(0, gamma + 1)])
    print(summation)
    phi.append(faculty(2 * gamma + 1) / (faculty(gamma) ** 2) * summation)

    # calculate it's derivatives up to order (gamma-1)

    for p in range(1, gamma+1):
        summation = sum([bin_coeff(gamma, k) * (-1) ** k * t ** (k + gamma + 1 - p) / (gamma + k + 1) * prod_iter(gamma, k, p)
                         for k in range(0, gamma + 1)])
        phi.append(faculty(2 * gamma + 1) / (faculty(gamma) ** 2) * summation)

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
    phi = []
    phi.append(35 * t ** 4 - 84 * t ** 5 + 70 * t ** 6 - 20 * t ** 7)
    phi.append(140 * t ** 3 - 420 * t ** 4 + 420 * t ** 5 - 140 * t ** 6)
    phi.append(420 * t ** 2 - 1680 * t ** 3 + 2100 * t ** 4 - 840 * t ** 5)
    phi.append(840 * t ** 1 - 5040 * t ** 2 + 8400 * t ** 3 - 4200 * t ** 4)
    return phi

print(prototype_fct(0.5,3))
print(test(0.5))
#print(prod_iter(1,2,2))
