# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import abc # abstract base class
import math

class Planner(object):
    """ Base class for a trajectory planner.

    Attributes:
        YA (int, float, ndarray): start value
        YB (int, float, ndarray): final value
        t0 (int, float): start time
        tf (int, float): final time
        d (int): trajectory is smooth up to the d-th derivative
    """

    def __init__(self, YA, YB, t0, tf, d):
        self.YA = YA
        self.YB = YB
        self.t0 = t0
        self.tf = tf
        self.d = d

    @abc.abstractmethod
    def eval(self):
        return


class PolynomialPlanner(Planner):
    """Planner subclass that uses a polynomial approach for trajectory generation

    Attributes:
        c (ndarray): parameter vector of polynomial
    """

    def __init__(self, YA, YB, t0, tf, d):
        super(PolynomialPlanner, self).__init__(YA, YB, t0, tf, d)
        self.c = self.coefficients()

    def eval(self,t):
        """Evaluates the planned trajectory at time t.

        Args:
            t (int, float): time

        Returns:
            Y (ndarray): y and its derivatives at t
        """
        if t < self.t0:
            Y = self.YA
        elif t > self.tf:
            Y = self.YB
        else:
            Y = np.dot(self.TMatrix(t),self.c)
        return Y


    def eval_vec(self,tt):
        """Samples the planned trajectory

        Args:
            tt (ndarray): time vector

        Returns:
            Y (ndarray): y and its derivatives at the sample points

        """
        Y = np.zeros([len(tt),len(self.YA)])
        for i in range(0,len(tt)):
            Y[i] = self.eval(tt[i])
        return Y


    def TMatrix(self, t):
        """Computes the T matrix at time t

        Args:
            t (int, float): time

        Returns:
            T (ndarray): T matrix

        """

        d = self.d
        n = d+1 # first dimension of T
        m = 2*d+2 # second dimension of T

        T = np.zeros([n, m])

        for i in range(0, m):
            T[0, i] = t ** i / math.factorial(i)
        for j in range(1, n):
            T[j, j:m] = T[0, 0:m-j]
        return T


    def coefficients(self):
        """Calculation of the polynomial parameter vector

        Returns:
            c (ndarray): parameter vector of the polynomial

        """
        t0 = self.t0
        tf = self.tf

        Y = np.append(self.YA, self.YB)

        T0 = self.TMatrix(t0)
        Tf = self.TMatrix(tf)

        T = np.append(T0, Tf, axis=0)

        # solve the linear equation system for c
        c = np.linalg.solve(T, Y)
        return c


# example
YA = np.array([0,0]) # t = t0
YB = np.array([1,0]) # t = tf
t0 = 1 # start time of transition
tf = 2 # final time of transition
tt = np.linspace(0, 3, 500) # 0 to 3 in 500 steps
d = 1 # smooth derivative up to order d
yd = PolynomialPlanner(YA, YB, t0, tf, d)

# display the parameter vector
print("c = ", yd.c)

# sample the generated trajectory
Y = yd.eval_vec(tt)

#plot the trajectory
plt.plot(tt, Y)
plt.title('Planned trajectory')
plt.legend([r'$y_d(t)$', r'$\dot{y}_d(t)$'])
plt.xlabel(r't in s')
plt.ylabel(r'$y(t)$')
plt.show()