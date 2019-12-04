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

Fortunately a more formal way to solve the problem demonstrated in the previous section exists. A time-variant linear system description provides a good compromise between reflecting the changing system properties and still being manageable using some of the tools known for linear systems.

For ease of notation, first define the time-variant system matrices
$$
A(t):=A^*(x^*(t), u^*(t)) \quad \text{and} \quad B(t):=B^*(x^*(t),u^*(t))
$$
and analogous to the previous section the corresponding state-space system
$$
\dot{\tilde x}(t) = A(t)\tilde x(t) + B(t) u(t)\, .
$$
Intuitively, this linear approximation stems not from linearization at one single operating point, but from linearization along the reference trajectory. Therefore this approach is valid as long as it can be assumed that in the closed loop the system closely tracks the desired states.

The first step in the LQR design process again is to define a cost function to be minimized. Similar to the previous section this function is now
$$
J=\tilde x^T(t_f) S \tilde x(t_f) +\int_0^{t_f} \tilde x^T(t) Q x(t) + \tilde u^T(t) R u(t) \, \mathrm d t\, .
$$
Note these two differences: for one the integration interval has changed from an infinite horizon to the finite length of the planned trajectory, also an additional end cost term with the weighting matrix $S$ was added.

To obtain the optimal state feedback, the initial value problem (IVP) defined by the matrix Riccati differential equation
$$
\frac{\mathrm d P(t)}{\mathrm d t}= -P(t)A(t) - A(t)^T P(t) + P(t) B(t) R^{-1} B(t)^T P(t) - Q(t)
$$
and
$$
P(t_f) = S
$$
must be solved.

After performing all the matrix operations this would mean simulating $n^2$ scalar ODEs. It can be shown though that $P(t)$ is symmetric, which reduces the number of distinct entries to $\frac{n(n+1)}{2}$ in the upper half. To extract these entries into a one-dimensional vector for simulation purposes, as well as to reconstruct the full matrix, the following \py code is used.

==`triuconvert`==

The initial value $S$ is technically a tuning parameter and can be freely chosen. However, for trajectories that transfer between two setpoints, such as in the example shown here, one should consider the following suggestion. If the final setpoint is to be stabilized after the transfer phase ends, it is desirable for the final state feedback of the time-variant section to be identical to the feedback designed for the LTI system resulting from linearization around the final point. This ensures a smooth transition between the two phases.

In practice this means solving the algebraic Riccati equation ==??== for the reference state at $t_f$ and using the solution as the initial value $S$. The implementation is as follows:

==`riccatiinit`==

The last quirk of the IVP ==??== is the "initial value" not being defined at time zero. While this mathematically does not make much of a difference, both our intuition and most existing software tools expect that the system "runs" in forward flowing time. The time reversal
$$
P(t) =P(t_f - \tau)=:\bar P(\tau)
$$
remedies this issue. This new function $\bar P(\tau)$ is almost the same as $P(t)$, only with a redefined argument, so that an increasing $\tau$ corresponds to $t$ decreasing from $t_f$ in the original function. Utilizing the chain rule $\frac{\mathrm d P(t)}{\mathrm d t} = -\frac{\mathrm d \bar P(\tau)}{\mathrm d \tau}$ yields a new IVP
$$
\frac{\mathrm d \bar P(\tau)}{\mathrm d \tau}= -\bar P(\tau) B(T-\tau) R^{-1} B(T-\tau)^T \bar P(\tau) + \bar P(\tau)A(T-\tau)+A(T-\tau)^T \bar P(\tau) + Q
$$
where the initial value is now the more familiar
$$
\bar P(0)=S=P(t_f)\, .
$$
As only system and reference trajectory information are required, the IVP can be solved offline. Values for the feedback matrix
$$
K(t) = R^{-1} B(t)^T P(t)
$$
are stored for the fixed time steps and then used later in the closed loop. Simulation of ==??== can be performed with a numerical integration algorithm of choice, a simple fixed step width Euler algorithm was used here.

==`riccatiint`==

The main difficulty is keeping the array indices corresponding to $t$ and $\tau$ straight, otherwise the code is structurally identical to the simulation of any dynamic system.

Finally, the actual simulation of the closed loop happens in the main simulation loop. The method of numerical integration is identical here.

==`sim`==

Simulation results are shown in Figure ==??==.

==DIAGRAM==

Unfortunately the final state trajectory is visually indistinguishable from the previous result. Still, for less benevolent systems than this academic example this method is the safer option, as it is applicable whenever the feedforward control is good enough to not deviate from the reference too far.

## Practical example: cart-pole system