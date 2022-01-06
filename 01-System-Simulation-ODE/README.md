<!-- LTeX: language=en-US -->
# SystemSimulation-ODE

This tutorial shows how to simulate a dynamic system described by a set of ordinary differential equations and how to illustrate the results by plots and animations.

## The following topics are covered by the tutorial

- Simulation of a dynamical system using odeint or solve_ivp of the SciPy integration package
- Presentation of the results using MatplotLib
- Creation of animations based on the simulation results in order impress decision makers
- Implementation of a flatness based feedforward and feedback control

## Prerequisites (software)

- Python >= 3.7 incl. NumPy, Scipy, and MatplotLib

## Prerequisites (knowledge)

- Basic knowledge in the Python programming language, esp. lists, tupels and lambda-functions

## How to build the tutorial

In the `doc` directory of this repository run 

``` bash
lualatex tutorial1.tex
biber tutorial1
makeglossaries tutorial1
lualatex tutorial1.tex
```
