# LISTING_START TrajGen
import numpy as np
import scipy.special as sps
import abc  # abstract base class


class TrajGen(abc.ABC):
    """ Base class for a trajectory generator.

    Attributes:
        y_a (int, float, ndarray): start value (size = d+1)
        y_b (int, float, ndarray): final value (size = d+1)
        t_0 (int, float): start time
        t_f (int, float): final time
        d (int): trajectory is smooth up at least to the d-th derivative
    """

    def __init__(self, y_a, y_b, t_0, t_f, d):
        self.YA = y_a
        self.YB = y_b
        self.t0 = t_0
        self.tf = t_f
        self.d = d

    @abc.abstractmethod
    def eval(self, t):
        pass

    @abc.abstractmethod
    def eval_vec(self, t):
        pass
# LISTING_END TrajGen

# LISTING_START FactorialFunc
    @classmethod
    def factorial(cls, x):
        """Calcualtes the faculty of x"""
        result = 1
        for i in range(2, x + 1):
            result *= i
        return result
# LISTING_END FactorialFunc


# LISTING_START PolynomialTrajGen
class PolynomialTrajGen(TrajGen):
    """TrajGen subclass that uses a polynomial approach for trajectory generation

    Attributes:
        c (ndarray): parameter vector of polynomial

    """
    def __init__(self, y_a, y_b, t_0, t_f, d):
        super().__init__(y_a, y_b, t_0, t_f, d)
        self.c = self.coefficients()
# LISTING_END PolynomialTrajGen

# LISTING_START PolynomialTrajGenEvalFunDef
    def eval(self, t):
        """Evaluates the planned trajectory at time t.

        Args:
            t (int, float): time

        Returns:
            y (ndarray): y and its derivatives at t
        """
        if t < self.t0:
            y = self.YA
        elif t > self.tf:
            y = self.YB
        else:
            y = np.dot(self.t_matrix(t), self.c)
        return y
# LISTING_END PolynomialTrajGenEvalFunDef

# LISTING_START PolynomialTrajGenEvalVecFunDef
    def eval_vec(self, tt):
        """Samples the planned trajectory

        Args:
            tt (ndarray): time vector

        Returns:
            y (ndarray): y and its derivatives at the sample points

        """
        y = np.zeros([len(tt), len(self.YA)])
        for i in range(0, len(tt)):
            y[i] = self.eval(tt[i])
        return y
# LISTING_END PolynomialTrajGenEvalVecFunDef

# LISTING_START TMatrixFunDef
    def t_matrix(self, t):
        """Computes the T matrix at time t

        Args:
            t (int, float): time

        Returns:
            t_mat (ndarray): T matrix

        """

        d = self.d
        n = d+1    # first dimension of T
        m = 2*d+2  # second dimension of T

        t_mat = np.zeros([n, m])

        for i in range(0, m):
            t_mat[0, i] = t ** i / self.factorial(i)
        for j in range(1, n):
            t_mat[j, j:m] = t_mat[0, 0:m-j]
        return t_mat
# LISTING_END TMatrixFunDef

# LISTING_START CoeffFunDef
    def coefficients(self):
        """Calculation of the polynomial parameter vector

        Returns:
            c (ndarray): parameter vector of the polynomial

        """
        t0 = self.t0
        tf = self.tf

        y = np.append(self.YA, self.YB)

        t0_mat = self.t_matrix(t0)
        tf_mat = self.t_matrix(tf)

        t_mat = np.append(t0_mat, tf_mat, axis=0)

        # solve the linear equation system for c
        c = np.linalg.solve(t_mat, y)

        return c
# LISTING_END CoeffFunDef


class PrototypeTrajGen(TrajGen):
    """TrajGen subclass that uses a polynomial approach for trajectory generation"""

    def __init__(self, y_a, y_b, t_0, t_f, d):
        super().__init__(y_a, y_b, t_0, t_f, d)
        # check if values are 0
        if any(self.YA[1:] != 0) or any(self.YB[1:] != 0):
            print('Boundary conditions of the derivatives set to 0. All given values are ignored.')

    def eval(self, t):
        """Evaluates the planned trajectory at time t.

                Args:
                    t (int, float): time

                Returns:
                    y (ndarray): y and its derivatives at t
                """

        phi = self.prototype_fct((t - self.t0) / (self.tf - self.t0))
        y = np.zeros([(self.d + 1)])
        if t < self.t0:
            y[0] = self.YA[0]
        elif t > self.tf:
            y[0] = self.YB[0]
        else:
            y[0] = self.YA[0] + (self.YB[0]-self.YA[0])*phi[0]
            for i in range(1, self.d+1):
                y[i] = (1/(self.tf-self.t0))**i * (self.YB[0]-self.YA[0])*phi[i]
        return y

    def eval_vec(self, tt):
        """Samples the planned trajectory

        Args:
            tt (ndarray): time vector

        Returns:
            y (ndarray): y and its derivatives at the sample points

        """
        y = np.zeros([len(tt), len(self.YA)])
        for i in range(0, len(tt)):
            y[i] = self.eval(tt[i])
        return y

    def prototype_fct(self, t):
        """ Implementation of the prototype function

        Args:
            t: time
        Returns: phi (vector of phi and its successive derivatives)

        """
        phi = np.zeros([self.d + 1])

        summation = sum([sps.binom(self.d, k) * (-1) ** k * t ** (k + self.d + 1) / (self.d + k + 1)
                         for k in range(0, self.d + 1)])
        phi[0] = self.factorial(2 * self.d + 1) / (self.factorial(self.d) ** 2) * summation

        # calculate it's derivatives up to order (d-1)

        for p in range(1, self.d + 1):
            summation = sum(
                [sps.binom(self.d, k) * (-1) ** k * t ** (k + self.d + 1 - p) / (self.d + k + 1) * self.prod_iter(k, p)
                 for k in range(0, self.d + 1)])
            phi[p] = self.factorial(2 * self.d + 1) / (self.factorial(self.d) ** 2) * summation

        return phi

    def prod_iter(self, k, p):
        """Calculates the iterative product"""
        result = 1
        for i in range(1, p + 1):
            result *= (self.d + k + 2 - i)
        return result


class GevreyTrajGen(TrajGen):
    """TrajGen that uses a Gevrey function approach and plans trajectories that are infinitely differentiable.
              |   0                                        t < t0
    phi(t) =  |   1/2(1 + tanh( (2T-1) / (4T(1-T))^s ))    t in [t0, tf]
              |   1                                        t > tf

    T = (t-t0)/(tf-t0)

              |   yA                       t < t0
    y_d(t) =  |   yA + (yB - yA)*phi(t)    t in [t0, tf]
              |   yB                       t > tf

    based on: "J. Rudolph, J. Winkler, F. Woittenek: Flatness Based Control of Distributed Parameter Systems:
    Examples and Computer Exercises from Various Technological Domains" Pages 88ff.
    """

    def __init__(self, y_a, y_b, t_0, t_f, d, s):
        super().__init__(y_a, y_b, t_0, t_f, d)
        self.s = s
        if any(self.YA[1:] != 0) or any(self.YB[1:] != 0):
            print('Boundary conditions of the derivatives set to 0. All given values are ignored.')

    def eval(self, t):
        """Evaluates the planned trajectory at time t.

        Args:
            t (int, float): time

        Returns:
            y (ndarray): y and its derivatives at t
        """
        y = np.zeros([(self.d + 1)])
        if t < self.t0:
            y[0] = self.YA[0]
        elif t > self.tf:
            y[0] = self.YB[0]
        else:
            t_val = min(max((t - self.t0) / (self.tf - self.t0), 0.001), 0.999)
            phi = self.phi(t_val)
            y = np.zeros_like(phi)
            y[0] = self.YA[0] + (self.YB[0] - self.YA[0]) * phi[0]
            for i in range(1, self.d + 1):
                y[i] = (1 / (self.tf - self.t0)) ** i * (self.YB[0] - self.YA[0]) * phi[i]
        return y

    def eval_vec(self, tt):
        """Samples the planned trajectory

        Args:
            tt (ndarray): time vector

        Returns:
            y (ndarray): y and its derivatives at the sample points

        """
        y = np.zeros([len(tt), len(self.YA)])
        for i in range(0, len(tt)):
            y[i] = self.eval(tt[i])
        return y

    def phi(self, t):
        """Calculates phi = 1/2*(1 + tanh( 2(2t-1) / (4t(1-t))^s )) ) + its deriv. up to order d"""
        phi = np.zeros([self.d + 1])
        phi[0] = 1/2*(1 + self.y(t, 0))
        for i in range(1, self.d + 1):
            phi[i] = 1/2*self.y(t, i)
        return phi

    def y(self, t, n):
        """Calculates y = tanh( 2(2t-1) / (4t(1-t))^s )) and its derivatives up to order n"""
        s = self.s
        if n == 0:
            # eq. A.3
            y = np.tanh(2*(2*t - 1) / ((4*t*(1 - t))**s))
        elif n == 1:
            # eq. A.5
            y = self.a(t, 2)*(1 - self.y(t, 0)**2)
        else:
            # eq. A.7
            y = sum(sps.binom(n - 1, k)*self.a(t, k + 2) *
                    self.z(t, n - 1 - k) for k in range(0, n))
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
            a = 1/(t*(1 - t))*((s - 2+n)*(2*t - 1)*self.a(t, n - 1) + (n - 1) *
                               (2*s - 4 + n)*self.a(t, n - 2))
        return a

    def z(self, t, n):
        if n == 0:
            # eq. A.6
            z = (1-self.y(t, 0)**2)
        else:
            # eq. for n-th derivative of z
            z = - sum(sps.binom(n, k)*self.y(t, k)*self.y(t, n - k) for k in range(0, n + 1))
        return z
