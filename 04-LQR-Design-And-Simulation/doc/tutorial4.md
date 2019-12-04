# Pytutorial 4 - LQR Design and Simulation

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
\end{pmatrix}\,.
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

Before an LQR controller can be designed, the system in question must be linear. As a rough approximation, Taylor linearization is used to obtain a linear time-invariant (LTI) system which is valid in the proximity to the operating point $(x^*, u^*)$. In new coordinates $\tilde x=x - x^*$ and $\tilde u = u - u^*$ as well as by using the Jacobians ==??==, the new state-space-form is then
$$
\dot{\tilde x}(t) = A(x^*, u^*) \tilde x(t) + B(x^*,u^*) \tilde u(t)\, .
$$
In \py, the operating point is defined by a time $t^*$ along the planned trajectory. The corresponding array index is first determined and then used to get the values $(x^*, u^*)$, with which the system matrices are computed.

==`linsys`==

Now an LQR controller will be designed by the book. This means defining a cost-function
$$
J = \int_0^\infty \tilde x^T(t) Q x(t) + \tilde u^T(t) R u(t) \, \mathrm d t
$$
with the tuning matrices $Q$ and $R$ usually chosen as diagonal. Minimizing this function leads to the algebraic Riccati equation
$$
P A + A^T P - P B R^{-1} B^T P + Q = 0
$$
having to be solved for $P$. In \py the SciPy package fortunately contains the routine `solve_continuous_are` which does exactly that.

The cost function then is minimal if a constant state feedback $\tilde u = - K \tilde x$ is applied, with
$$
K = \begin{pmatrix} k_1 & k_2 \end{pmatrix} = R^{-1} B^T P\, .
$$
These computations -- save for the applying the actual control law, which happens in the main simulation loop -- can now be compactly written as

==`solveare`==

Now it is time to look at some simulation results. Shown in Figure ==??== on the left side is the combination of feedforward control and feedback control, designed for a system linearized around the planned trajectory values at $t^*=0$. The initial state $x(0)$ is slightly offset from the planned initial position $x^*(0)$, but the controller manages to track the trajectory anyway.

However, this choice of linearization point was arbitrary and lucky. When e. g. $t^*=5$ is picked instead, the closed loop system immediately starts moving away from the reference value and tracking capability is lost. This is demonstrated on the right side of Figure ==??==.

==DIAGRAM==

In fact, simply sticking to $t^*=0$ also does not work. For other reference trajectories, e. g. for $y(0)=2$ and $y(10)=-2$ (the reverse trajectory if you will), choosing the same linearization time fails. In that case $t^*=10$ must instead be used.

It appears that tracking arbitrary trajectories with this approach is not feasible, since the linearized system can never sufficiently represent the more involved non-linear dynamics and a "one size fits all" constant state feedback is therefore bound to fail.

One might be tempted to improve the controller by simply re-linearizing the system and re-computing the feedback matrix at every time step. For this specific case this approach actually works and is very easy to implement when a normal LQR controller already exists. Simulation results are shown in Figure ==??==, along with a plot of the changing feedback matrix entries over time.

==DIAGRAM==

Remember this is dangerous territory though. No stability guarantees can be made here, because this ad-hoc solution corresponds to a linear system approximation with time-variant system matrices
$$
\dot{\tilde x}(t) = A(t)\tilde x(t) + B(t) u(t)\, .
$$
For a linear time-variant system, the closed loop system matrix $A(t) - B(t)K(t)$ must only contain eigenvalues $s$ with $\mathrm{Re}\,(s) < 0$, but also be __constant__ in order for the closed loop to be stable! This property is not ensured in this controller design, using it is therefore not recommended.

## LQR for LTV systems

- define $A(t):=A^*(x^*(t), u^*(t))$ and $B(t):=B^*(x^*(t),u^*(t))$

- alternative: actual LTV system $\dot{\tilde x}(t) = A(t)\tilde x(t) + B(t) u(t)$

- $J=\tilde x^T(t_f) S \tilde x(t_f) +\int_0^{t_f} \tilde x^T(t) Q x(t) + \tilde u^T(t) R u(t) \, \mathrm d t$

- optimal feedback is then obtained by solving IVP

- $$
  \frac{\mathrm d P(t)}{\mathrm d t}= -P(t)A(t) - A(t)^T P(t) + P(t) B(t) R^{-1} B(t)^T P(t) - Q(t)
  $$

- with $P(t_f) = S$

- $P(t)$ is actually symmetric, so we don't need full matrix ODE, only for entries in upper triangular half

- converting to and from upper triu vector is done as such

- ==`triuconvert`==

- what S to choose? free to do whatever, but for setpoint transition it makes sense to arrive at the feedback we would have for an LTI LQR in the endpoint --> solving ARE

- ==`riccatiinit`==

- issue: we (and software) is used to solving ODEs in forward time direction

- solution: time reversal $P(t) =P(t_f - \tau)=:\bar P(\tau)$

- chain rule $\frac{\mathrm d P(t)}{\mathrm d t} = -\frac{\mathrm d \bar P(\tau)}{\mathrm d \tau}$

- new IVP $\frac{\mathrm d \bar P(\tau)}{\mathrm d \tau}= -\bar P(\tau) B(T-\tau) R^{-1} B(T-\tau)^T \bar P(\tau) + \bar P(\tau)A(T-\tau)+A(T-\tau)^T \bar P(\tau) + Q$ with $P(t_f)=\bar P(0)=S$

- now we can finally implement this. this is done "offline" using system and trajectory information. resulting K(t) is stored

- here we use simple euler-1

- ==`riccatiint`==

- pay attention to indices

- now only important remaining part is simulation loop

- looks similar, also just numerically integrating ODE

- again uses fixed step Euler-1

- ==`sim`==

- now let's look at result

- ==DIAGRAM==

- visually indistinguishable from previous result :/ still, this is the safe option for less benevolent systems

## Practical example: cart-pole system