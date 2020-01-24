# LISTING_START Planner
import numpy as np
import abc # abstract base class
import math
import scipy as sp
from scipy import special

class Planner(object):
    """ Base class for a trajectory planner.

    Attributes:
        YA (int, float, ndarray): start value (size = d+1)
        YB (int, float, ndarray): final value (size = d+1)
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
# LISTING_END Planner

#LISTING_START PolynomialPlanner
class PolynomialPlanner(Planner):
    """Planner subclass that uses a polynomial approach for trajectory generation

    Attributes:
        c (ndarray): parameter vector of polynomial

    """
#LISTING_END PolynomialPlanner

# LISTING_START PolynomialPlannerInitFunDef
    def __init__(self, YA, YB, t0, tf, d):
        super(PolynomialPlanner, self).__init__(YA, YB, t0, tf, d)
        self.c = self.coefficients()
# LISTING_END PolynomialPlannerInitFunDef

# LISTING_START PolynomialPlannerEvalFunDef
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
# LISTING_END PolynomialPlannerEvalFunDef


# LISTING_START PolynomialPlannerEvalVecFunDef
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
# LISTING_END PolynomialPlannerEvalVecFunDef


# LISTING_START TMatrixFunDef
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
# LISTING_END TMatrixFunDef

# LISTING_START CoeffFunDef
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
# LISTING_END CoeffFunDef


class PrototypePlanner(Planner):
    """Planner subclass that uses a polynomial approach for trajectory generation"""

    def __init__(self, YA, YB, t0, tf, d):
        super(PrototypePlanner, self).__init__(YA, YB, t0, tf, d)
        # check if values are 0
        if any(self.YA[1:]!=0) or any(self.YB[1:]!=0):
            print('Boundary conditions of the derivatives set to 0. All given values are ignored.')

    def eval(self, t):
        """Evaluates the planned trajectory at time t.

                Args:
                    t (int, float): time

                Returns:
                    Y (ndarray): y and its derivatives at t
                """

        phi = self.prototype_fct((t - self.t0) / (self.tf - self.t0))
        Y = np.zeros([(self.d + 1)])
        if t < self.t0:
            Y[0] = self.YA[0]
        elif t > self.tf:
            Y[0] = self.YB[0]
        else:
            Y[0] = self.YA[0] + (self.YB[0]-self.YA[0])*phi[0]
            for i in range(1,self.d+1):
                Y[i] = (1/(self.tf-self.t0))**i * (self.YB[0]-self.YA[0])*phi[i]
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


    def prototype_fct(self, t):
        """Prototype function, that is used in the path planner and returns a polynomial and its derivatives up to order gamma.

        Args:
            t: time
        Returns: phi (vector of phi and its successive derivatives)

        """
        phi = np.zeros([self.d + 1])

        summation = sum([sp.special.binom(self.d, k) * (-1) ** k * t ** (k + self.d + 1) / (self.d + k + 1)
                         for k in range(0, self.d + 1)])
        phi[0] = self.faculty(2 * self.d + 1) / (self.faculty(self.d) ** 2) * summation

        # calculate it's derivatives up to order (d-1)

        for p in range(1, self.d + 1):
            summation = sum(
                [sp.special.binom(self.d, k) * (-1) ** k * t ** (k + self.d + 1 - p) / (self.d + k + 1) * self.prod_iter(k, p)
                 for k in range(0, self.d + 1)])
            phi[p] = self.faculty(2 * self.d + 1) / (self.faculty(self.d) ** 2) * summation

        return phi


    def faculty(self, x):
        """Calcualtes the faculty of x"""
        result = 1
        for i in range(2, x + 1):
            result *= i
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

    based on: "J. Rudolph, J. Winkler, F. Woittenek: Flatness Based Control of Distributed Parameter Systems:
    Examples and Computer Exercises from Various Technological Domains" Pages 88ff.
    """


    def __init__(self, YA, YB, t0, tf, d, s):
        super(GevreyPlanner, self).__init__(YA, YB, t0, tf, d)
        self.s = s
        if any(self.YA[1:]!=0) or any(self.YB[1:]!=0):
            print('Boundary conditions of the derivatives set to 0. All given values are ignored.')

    def eval(self, t):
        """Evaluates the planned trajectory at time t.

        Args:
            t (int, float): time

        Returns:
            Y (ndarray): y and its derivatives at t
        """
        Y = np.zeros([(self.d + 1)])
        if t < self.t0:
            Y[0] = self.YA[0]
        elif t > self.tf:
            Y[0] = self.YB[0]
        else:
            T = min(max((t-self.t0)/(self.tf-self.t0),0.001),0.999)
            phi = self.phi(T)
            Y = np.zeros_like(phi)
            Y[0] = self.YA[0] + (self.YB[0] - self.YA[0]) * phi[0]
            for i in range(1, self.d + 1):
                Y[i] = (1 / (self.tf - self.t0)) ** i * (self.YB[0] - self.YA[0]) * phi[i]
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

    def phi(self, t):
        """Calculates phi = 1/2*(1 + tanh( 2(2t-1) / (4t(1-t))^s )) ) and it's derivatives up to order d"""
        phi = np.zeros([self.d + 1])
        phi[0] = 1/2*(1 + self.y(t, 0))
        for i in range(1, self.d + 1):
            phi[i] = 1/2*self.y(t, i)
        return phi


    def y(self, t, n):
        """Calculates y = tanh( 2(2t-1) / (4t(1-t))^s )) and it's derivatives up to order n"""
        s = self.s
        if n == 0:
            # eq. A.3
            y = np.tanh(2*(2*t - 1) / ((4*t*(1 - t))**s))
        elif n == 1:
            # eq. A.5
            y = self.a(t, 2)*(1 - self.y(t, 0)**2)
        else:
            # eq. A.7
            y = sum(sp.special.binom(n - 1, k)*self.a(t, k + 2)*self.z(t, n - 1 - k) for k in range(0, n))
        return y


    def a(self, t, n):
        s = self.s
        if n == 0:
            # eq. A.4
            a = ((4*t*(1 - t))**(1 - s))/(2*(s - 1))
        elif n == 1:
            # eq. for da/dt
            a = 2*(2*t - 1) / ((4*t*(1 - t))**s)
        else:
            # eq. for the n-th derivative of a
            a = 1/(t*(1 - t))*((s - 2+n)*(2*t - 1)*self.a(t, n - 1) + (n - 1)*(2*s - 4 + n)*self.a(t, n - 2))
        return a


    def z(self, t, n):
        if n == 0:
            # eq. A.6
            z = (1-self.y(t, 0)**2)
        else:
            # eq. for n-th derivative of z
            z = - sum(sp.special.binom(n, k)*self.y(t, k)*self.y(t, n - k) for k in range(0, n + 1))
        return z
