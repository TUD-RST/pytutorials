# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from Planner import PolynomialPlanner, PrototypePlanner, GevreyPlanner

# example 1 - polynomial planner
YA = np.array([0, 0, 0]) # t = t0
YB = np.array([1, 0, 0]) # t = tf

t0 = 0 # start time of transition
tf = 1 # final time of transition
tt = np.linspace(t0, tf, 100) # -1 to 4 in 500 steps

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

# example 2 - prototype planner
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

# example 3 - Gevrey planner
s1 = 1.1
s2 = 1.9
yd3 = GevreyPlanner(YA, YB, t0, tf, d, s1)
yd4 = GevreyPlanner(YA, YB, t0, tf, d, s2)
# sample the generated trajectory
Y3 = yd3.eval_vec(tt)
Y4 = yd4.eval_vec(tt)
#plot the trajectory
plt.figure(3)
plt.plot(tt, Y3)
plt.title('Planned trajectory')
plt.legend([r'$y_d(t)$', r'$\dot{y}_d(t)$',r'$\ddot{y}_d(t)$'])
plt.xlabel(r't in s')
plt.grid(True)

plt.figure(4)
plt.plot(tt, Y3[:,0],tt, Y4[:,0])
plt.title(r'$y = \varphi^{(0)}_{\sigma,T}(t)$')
plt.legend([r'$\sigma = 1.1$', r'$\sigma = 1.9$'])
plt.xlabel(r't in s')
plt.grid(True)

plt.figure(5)
plt.plot(tt, Y3[:,1],tt, Y4[:,1])
plt.title(r'$y = \varphi^{(1)}_{\sigma,T}(t)$')
plt.legend([r'$\sigma = 1.1$', r'$\sigma = 1.9$'])
plt.xlabel(r't in s')
plt.grid(True)