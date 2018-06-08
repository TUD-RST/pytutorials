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
    def evaluate(self,t):
        # TODO: check if t is scalar
        if t<self.t0:
            y = self.y0
        elif t>self.tT:
            y = self.yT
        else:
            y = 1
        return y

    def TMatrix(self, t):
        d = self.d
        T = np.zeros_like(np.matrix(d,2*d+1))
        for j in range(0, d):
            for i in range(0, 2*d+1):
                T[j,i] = t**(i-j)/math.factorial(i-j)
        return T


    def coefficients(self):
        # TODO: should be called, when instance is constructed
        t0 = self.t0
        tT = self.tT
        Y = np.stack(self.y0, self.yT)
        T = np.stack(self.TMatrix(t0), self.TMatrix(tT))
        c = np.linalg.inv(T)*Y
        return c


