# -*- coding: utf-8 -*-
import abc # abstract base class
import numpy as np
import math
class Planner(object):

    def __init__(self, y0, yT, t0, tT, d):
        self.y0 = y0 # initial value
        self.yT = yT # final value
        self.t0 = t0 # start time
        self.tT = tT # end time
        self.d = d  # trajectory is smooth up to the d-th derivative

    @abc.abstractmethod
    def evaluate(self):
        return


class PolynomialPlanner(Planner):
    def eval(self,t):
        # TODO: check if t is scalar
        if t<self.t0:
            y = self.y0
        elif t>self.tT:
            y = self.yT
        else:
            y = np.dot(self.TMatrix(t),self.coefficients())
        return y

    def TMatrix(self, t):
        d = self.d
        n = d+1
        m = 2*d+2
        T = np.zeros([n, m])
        for i in range(0, m):
            T[0, i] = t ** i / math.factorial(i)
        for j in range(1, n):
            T[j, j:m] = T[0, 0:m-j]
        return T


    def coefficients(self):
        # TODO: should be called, when instance is constructed
        t0 = self.t0
        tT = self.tT
        Y = np.append(self.y0, self.yT)
        T0 = self.TMatrix(t0)
        TT = self.TMatrix(tT)
        T = np.append(T0, TT, axis=0)
        c = np.linalg.solve(T, Y)
        return c



y0 = np.array([0.5, 1])
yT = np.array([1, 0])
t0 = 0.2
tT = 1
d = 1
pp = PolynomialPlanner(y0,yT,t0,tT,d)

print(pp.TMatrix(t0))
print(pp.TMatrix(tT))
print(pp.coefficients())
print(pp.eval(t0))
print(pp.eval(tT))
print(pp.eval(0.5*(tT-t0)))
