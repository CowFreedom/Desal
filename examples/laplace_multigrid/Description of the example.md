# solve_poisson_multigrid #


This example numerically solves

<img src="https://latex.codecogs.com/svg.image?\Delta u=0" title="\frac{\partial \textbf{u}}{\partial t}=v\Delta\textbf{u}" />

on a rectangular grid

<img src="https://latex.codecogs.com/svg.image?D\subset&space;\mathbb{R}^2:=\left&space;[&space;0,\text{width}&space;\right&space;]&space;\times&space;\left&space;[&space;0,\text{height}&space;\right&space;]" title="D\subset \mathbb{R}^2:=\left [ 0,\text{width} \right ] \times \left [ 0,\text{height} \right ]" />

with Dirichlet boundary conditions

<img src="https://latex.codecogs.com/svg.image?u\left&space;(t,&space;\textbf{x}&space;\right&space;)=k,&space;\quad&space;\quad&space;k,t\in&space;\mathbb{R},\textbf{x}&space;\in&space;\partial&space;D" title="\textbf{u}\left (t, \textbf{x} \right )=k, \quad \quad k,t\in \mathbb{R},\textbf{x} \in \partial D" />

The Laplace operator is represented by a first order approximation of the form

<img src="https://latex.codecogs.com/svg.image?\Delta {u}_{n}&space;=\frac{1}{h_xh_y}&space;\textbf{M}{u}_{n}&plus;O\left&space;(&space;h^{-2}&space;\right&space;)" title="\left ( \textbf{I}-v\cdot dt \Delta\right)\textbf{u}_{n} = \left ( \textbf{I}-\frac{v\cdot dt}{h_xh_y} \textbf{M}\right)\textbf{u}_{n}+O\left ( h^{-2} \right )" />

The name of this example is derived from the circumstance that at each time point <img src="https://latex.codecogs.com/svg.image?t" title="t" /> a Poisson equation is solved. This linear system of equations is solved with a V-Cycle
multigrid algorithm.

## What do I see

Calculated is the heat distribution on a rectangle with Dirichlet boundary conditions. The result can be printed or saved in a csv file (see code).
Because the long term distribution is calculated, all grid points attain the boundary value.

## Why do I want to see this

Play around with different hyperparameters of the V-Cycle multigrid solver to understand its behavior with respect to the problem geometry and boundary conditions.

## Usage
A build script *buildtask_gcc.cmd* is included in this folder. On Windows, you can use it to compile 
the example. Alternatively, you can open the build script, look for the compilation statements and compile it manually.

