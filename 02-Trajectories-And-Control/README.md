<!-- LTeX: language=en-US -->

# Trajectory planning and control

This tutorial shows how to implement a trajectory generator and how to establish a flatness
based feedforward and feedback control

## The following topics are covered by the tutorial

- Implementation of different trajectory generators in a class hierarchy
- Flatness based feedforward control
- Flatness based feedback control
- Some topics of object-oriented programming in Python (as needed)

## Prerequisites (software)

- Python >= 3.7 incl. NumPy, Scipy, and MatplotLib

## Prerequisites (knowledge)

- Basic knowledge in the Python programming language, esp. lists, tupels and lambda-functions
- [Tutorial 01-System-Simulation-ODE](../01-System-Simulation-ODE/README.md)

## How to build the tutorial

In the `doc` directory of this repository run

``` bash
lualatex tutorial2.tex
biber tutorial2
lualatex tutorial2.tex
```
