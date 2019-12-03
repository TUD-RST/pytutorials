# Pytutorial 4 - LQR Design and Simulation

Todo:

- [ ] Figure out what plots I need for academic example
  - [x] What are the trajectory and initial value conditions, that show the difference between different linearization points most clearly? Answer: Trajectory from -2 to 2 (-2.1, 0.1), Trajectory from 2 to -2 (2.1, 0.1)
  - [ ] Plots: 
- [ ] Figure out structure of text
- [ ] Figure out code structure to generate those plots
- [ ] Make figures
- [ ] ...



## Introduction

The goal of this tutorial is to teach the usage of the programming language _Python_ as a tool for developing and simulating control systems. The following topics are covered:

- Flatness based feedforward control using existing trajectory generators
- Feedback control using LQR for linear time-invariant (LTI) system
- Demonstration of problematic situations
- Feedback control using LQR linear time-variant (LTV) system

Later in this tutorial this process is applied to design control strategies for the cart-pole system from a previous tutorial.

## Implementing the system

In order to demonstrate the design methods discussed in the following, a simple academic example from ==??== will be used. The system state $x=(x_1, x_2)$ has two components and the scalar input is called $u$. Written in state-space-form, the system equation then is:
$$
\dot x = F(x, u) = 
\begin{pmatrix}
a \sin(x_2)\\
-x_1^2+u\\
\end{pmatrix}\,
$$
Later, the jacobians
$$
A(x^*, u^*) := \left.\frac{\partial F}{\partial x}\right\vert_{(x^*, u^*)}
= \begin{pmatrix}
0 & a \cos(x^*_2)\\
-2 x^*_1 & 0
\end{pmatrix}
$$
and
$$
B(x^*, u^*) := \left.\frac{\partial F}{\partial u}\right\vert_{(x^*, u^*)}
= \begin{pmatrix}
0 \\ 1
\end{pmatrix}
$$
will also be needed.

Implementing this system in Python then simply means expressing these terms as functions containing these computations. 

==`defsystem`==

Notable are mainly the usage of Numpy-Arrays for all vectors and the `Parameters` class, please refer to a previous tutorial if unsure about these aspects.

## Trajectory planning and feed-forward

Similar to previous tutorials, feedforward control design can be simplified significantly by exploiting the flatness property of this system. Specifically, the flat output of this system is $y=x_1$. Recall, this means a desired trajectory $t \mapsto x_1^*(t)$ can be freely chosen (as long as it is sufficiently often differentiable). Then, the trajectories for all other system variables (states and inputs) is calculated analytically without integration.

To this effect, the first system equation in ==??== is solved for $x_2$, yielding
$$
x_2 = \arcsin\left(\frac{\dot x_1}{a}\right)
$$
which also introduces the constraint $\forall t: |\dot x_1(t)| \leq |a|$ to the trajectory planning.

After differentiating ==??== w.r.t. time, resulting in
$$
\dot x_2 = \frac{\ddot x_1}{\sqrt{a^2-\dot x_1^2}}\,,
$$
a term for $u$ is obtained by solving the second component of ==??==:
$$
u = \dot x_2 + x_1^2 =  \frac{\ddot x_1}{\sqrt{a^2-\dot x_1^2}} + x_1^2\, .
$$
For the Python implementation, the trajectory planner from a previous tutorial is reused to obtain a polynomial that transitions from $y(t_0) = y_0$ to $y(t_f) = y_f$. This function is then evaluated at the time values stored in vector `t_traj`.

==`plantraj`==

The previously derived formulas are then translated into NumPy operations to obtain the values for $x_2$ and $u$ at every time step.

==`flatness`==

## Reminder: LQR for LTI systems

- for a non-linear system $\dot x = f(x, u)$, we can linearize at some operating point $(x^*, u^*)$ to get $\dot{\tilde x}(t) = A \tilde x(t) + B \tilde u(t)$
- `linsys`
- for this __LTI__ system we can then design an LQR controller as we know it
- $J = \int_0^\infty \tilde x^T(t) Q x(t) + \tilde u^T(t) R u(t) \, \mathrm d t$
- this means solving the Riccati equation $P A + A^T P - P B R^{-1} B^T P + Q = 0$
- feedback is then $\tilde u = - K \tilde x$ with $K = R^{-1} B^T P$
- `solveare`
- now let's show some results, only one trajectory since that's less confusing. -2 to 2 is most convincing
- Diagram: works with -2 to 2 and linearizing point at t=0, doesn't work with t=tf/2
- System dynamics is obviously too different when moving far away from linearization point, we might be lucky, we might not
- ad-hoc solution: linearize the system at every time step and re-solve ARE
- diagram: t=tf/2
- seems to work, but no mathematical foundation

## LQR for LTV systems

- alternative: actual LTV system $\dot{\tilde x}(t) = A(t)\tilde x(t) + B(t) u(t)$
- $J=\tilde x^T(t_f) S \tilde x(t_f) +\int_0^{t_f} \tilde x^T(t) Q x(t) + \tilde u^T(t) R u(t) \, \mathrm d t$

- optimal feedback is then obtained by solving

- $$
  \frac{\mathrm d P(t)}{\mathrm d t}= -P(t)A(t) + A(t)^T P(t) - P(t) B(t) R^{-1} B(t)^T P(t) + Q(t)
  $$

- 

## Practical example: cart-pole system