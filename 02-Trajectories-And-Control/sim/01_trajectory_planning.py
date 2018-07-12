# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Planner import PolynomialPlanner, PrototypePlanner, GevreyPlanner

# example 1
YA = np.array([0, 0, 0]) # t = t0
YB = np.array([1, 0, 0]) # t = tf
t0 = 0 # start time of transition
tf = 1 # final time of transition
tt = np.linspace(t0-1, tf+1, 500) # -1 to 4 in 500 steps
d = 2 # smooth derivative up to order d
yd = PolynomialPlanner(YA, YB, t0, tf, d)

# display the parameter vector
print("c = ", yd.c)

# sample the generated trajectory
Y = yd.eval_vec(tt)

#plot the trajectory
plt.figure(1)
plt.plot(tt, Y)
plt.title('Planned trajectory')
plt.legend([r'$y_d(t)$', r'$\dot{y}_d(t)$',r'$\ddot{y}_d(t)$'])
plt.xlabel(r't in s')
plt.grid(True)

# example 2
yd2 = PrototypePlanner(YA, YB, t0, tf, d)

# sample the generated trajectory
Y2 = yd2.eval_vec(tt)

#plot the trajectory
plt.figure(2)
plt.plot(tt, Y2)
plt.title('Planned trajectory')
plt.legend([r'$y_d(t)$', r'$\dot{y}_d(t)$',r'$\ddot{y}_d(t)$'])
plt.xlabel(r't in s')
plt.grid(True)

# example 3
s = 1.1
yd3 = GevreyPlanner(YA, YB, t0, tf, d, s)

# sample the generated trajectory
Y3 = yd3.eval_vec(tt)
#plot the trajectory
plt.figure(3)
plt.plot(tt, Y3)
plt.title('Planned trajectory')
plt.legend([r'$y_d(t)$', r'$\dot{y}_d(t)$',r'$\ddot{y}_d(t)$'])
plt.xlabel(r't in s')
plt.grid(True)
plt.show()