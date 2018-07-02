import numpy as np
import abc # abstract base class
import math
import scipy as sp
from scipy import special

class Planner(object):
    """ Base class for a trajectory planner.

    Attributes:
        YA (int, float, ndarray): start value
        YB (int, float, ndarray): final value
        t0 (int, float): start time
        tf (int, float): final time
        d (int): trajectory is smooth up at least to the d-th derivative
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

    def eval(self, t):
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
            Y = np.dot(self.TMatrix(t), self.c)
        return Y


    def eval_vec(self, tt):
        """Samples the planned trajectory

        Args:
            tt (ndarray): time vector

        Returns:
            Y (ndarray): y and its derivatives at the sample points

        """
        Y = np.zeros([len(tt), len(self.YA)])
        for i in range(0, len(tt)):
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
        print(Y)

        T0 = self.TMatrix(t0)
        Tf = self.TMatrix(tf)

        T = np.append(T0, Tf, axis=0)

        # solve the linear equation system for c
        c = np.linalg.solve(T, Y)
        return c


class PrototypePlanner(Planner):
    """Planner subclass that uses a polynomial approach for trajectory generation"""

    def eval(self, t):
        """Evaluates the planned trajectory at time t.

                Args:
                    t (int, float): time

                Returns:
                    Y (ndarray): y and its derivatives at t
                """

        phi = self.prototype_fct((t - self.t0) / (self.tf - self.t0))
        if t < self.t0:
            Y = self.YA
        elif t > self.tf:
            Y = self.YB
        else:
            Y = np.zeros([(self.d + 1)])
            for i in range(0, self.d + 1):
                Y[i] = self.YA[i] + sum(
                    (self.bin_coeff(i, j) * (self.YB[i - j] - self.YA[i - j]) * (1 / (self.tf - self.t0)) ** j * phi[j]) for j in
                    range(0, i + 1))
        return Y.T


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


    def prototype_fct(self, t):
        """Prototype function, that is used in the path planner and returns a polynomial and its derivatives up to order gamma.

        Args:
            t: time
        Returns: phi (vector of phi and its successive derivatives)

        """
        phi = np.zeros([self.d + 1])

        summation = sum([self.bin_coeff(self.d, k) * (-1) ** k * t ** (k + self.d + 1) / (self.d + k + 1)
                         for k in range(0, self.d + 1)])
        phi[0] = self.faculty(2 * self.d + 1) / (self.faculty(self.d) ** 2) * summation

        # calculate it's derivatives up to order (d-1)

        for p in range(1, self.d + 1):
            summation = sum(
                [self.bin_coeff(self.d, k) * (-1) ** k * t ** (k + self.d + 1 - p) / (self.d + k + 1) * self.prod_iter(k, p)
                 for k in range(0, self.d + 1)])
            phi[p] = self.faculty(2 * self.d + 1) / (self.faculty(self.d) ** 2) * summation

        return phi


    def faculty(self, x):
        """Calcualtes the faculty of x"""
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result


    def bin_coeff(self, n, k):
        """Calculates the binomial coefficient of n over k"""
        result = self.faculty(n) / (self.faculty(k) * self.faculty(n - k))
        return result


    def prod_iter(self, k, p):
        """Calculates the iterative product"""
        result = 1
        for i in range(1, p + 1):
            result *= (self.d + k + 2 - i)
        return result


class GevreyPlanner(Planner):
    """Planner that uses a Gevrey function approach and plans trajectories that are infinitely differentiable.
               /   0                                        t < t0
    phi(t) =  |    1/2(1 + tanh( (2T-1) / (4T(1-T))^s ))    t in [t0, tf]
               \   1                                        t > tf

    T = (t-t0)/(tf-t0)

               /   yA                       t < t0
    y_d(t) =  |    yA + (yB - yA)*phi(t)    t in [t0, tf]
               \   yB                       t > tf
    """


    def __init__(self, YA, YB, t0, tf, d, s):
        super(GevreyPlanner, self).__init__(YA, YB, t0, tf, d)
        self.s = s

    def eval(self, t):
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
            T = min(max((t-self.t0)/(self.tf-self.t0),0.001),0.999)
            phi = self.Y(T)
            Y = np.zeros([(self.d + 1)])
            for i in range(0, self.d + 1):
                Y[i] = self.YA[i] + sum(
                    (sp.special.binom(i, j) * (self.YB[i - j] - self.YA[i - j]) * (1 / (self.tf - self.t0)) ** j * phi[j])
                    for j in
                    range(0, i + 1))
        return Y


    def eval_vec(self, tt):
        """Samples the planned trajectory

        Args:
            tt (ndarray): time vector

        Returns:
            Y (ndarray): y and its derivatives at the sample points

        """
        Y = np.zeros([len(tt), len(self.YA)])
        for i in range(0, len(tt)):
            Y[i] = self.eval(tt[i])
        return Y

    def Y(self, t):
        Y = np.zeros([self.d + 1])
        Y[0] = 1/2*(1 + self.y(t, 0))
        for i in range(1, self.d + 1):
            Y[i] = 1/2*self.y(t, i)
        return Y


    def y(self, t, n):
        """Calculates phi(t) = 1/2(1 + tanh( 2(2T-1) / (4T(1-T))^s )) and it's derivatives up to order d"""
        s = self.s
        if n == 0:
            y = np.tanh(2*(2*t - 1) / ((4*t*(1 - t))**s))
        elif n == 1:
            y = self.a(t, 2)*(1 - self.y(t, 0)**2)
        else:
            y = sum(sp.special.binom(n - 1, k)*self.a(t, k + 2)*self.z(t, n - 1 - k) for k in range(0, n))
        return y


    def a(self, t, n):
        s = self.s
        if n == 0:
            a = ((4*t*(1 - t))**(1 - s))/(2*(s - 1))
        elif n == 1:
            a = 2*(2*t - 1) / ((4*t*(1 - t))**s)
        else:
            a = 1/(t*(1 - t))*((s - 2+n)*(2*t - 1)*self.a(t, n - 1) + (n - 1)*(2*s - 4 + n)*self.a(t, n - 2))
        return a


    def z(self, t, n):
        if n == 0:
            z = (1-self.y(t, 0)**2)
        else:
            z = - sum(sp.special.binom(n, k)*self.y(t, k)*self.y(t, n - k) for k in range(0, n + 1))
        return z





