# solve_poisson_multigrid #


This example numerically solves

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;u}{\partial&space;t}=v\Delta u" title="\frac{\partial \textbf{u}}{\partial t}=v\Delta\textbf{u}" />

on a rectangular grid

<img src="https://latex.codecogs.com/svg.image?D\subset&space;\mathbb{R}^2:=\left&space;[&space;0,\text{width}&space;\right&space;]&space;\times&space;\left&space;[&space;0,\text{height}&space;\right&space;]" title="D\subset \mathbb{R}^2:=\left [ 0,\text{width} \right ] \times \left [ 0,\text{height} \right ]" />

with initial conditions

<img src="https://latex.codecogs.com/svg.image?u(0,\mathbf{x}&space;)=a(0)&plus;&space;a(0)-0.5\left&space;(&space;1&plus;\text{tan}(r-(x_1-c_1)^2-(x_2-c_2)^2)&space;\right&space;),\quad&space; \textbf{x} \in \partial D" title="u(0,\mathbf{x} )=a(t)+ a(t)-0.5\left ( 1+\text{tan}(r-(x_1-c_1)^2-(x_2-c_2)^2) \right ) \textbf{x} \in D" />

and boundary conditions

<img src="https://latex.codecogs.com/svg.image?u\left&space;(t,&space;\textbf{x}&space;\right&space;)=a(t),&space;\quad&space;\quad&space;t\in&space;\mathbb{R},\textbf{x}&space;\in&space;\partial&space;D" title="\textbf{u}\left (t, \textbf{x} \right )=a(t), \quad t\in \mathbb{R},\textbf{x} \in \partial D" />

Common names for this problem are diffusion or heat equation. The heat equation can be solved numerically by 

<img src="https://latex.codecogs.com/svg.image?\left&space;(&space;&space;\textbf{I}-\frac{v\cdot&space;dt}{h_xh_y}&space;\textbf{M}\right){u}_{n}={u}_{n&plus;1}" title="\left ( \textbf{I}-\frac{v\cdot dt}{h_xh_y} \textbf{M}\right)\textbf{u}_{n}=\textbf{u}_{n+1}" />

with identity matrix <img src="https://latex.codecogs.com/svg.image?\textbf{I}" title="\textbf{I}" />.
The Laplace operator is represented by a first order approximation of the form

<img src="https://latex.codecogs.com/svg.image?\Delta {u}_{n}&space;=\frac{1}{h_xh_y}&space;\textbf{M}{u}_{n}&plus;O\left&space;(&space;h^{-2}&space;\right&space;)" title="\left ( \textbf{I}-v\cdot dt \Delta\right)\textbf{u}_{n} = \left ( \textbf{I}-\frac{v\cdot dt}{h_xh_y} \textbf{M}\right)\textbf{u}_{n}+O\left ( h^{-2} \right )" />



The name of this example is derived from the circumstance that at each time point <img src="https://latex.codecogs.com/svg.image?t" title="t" /> a Poisson equation is solved. This linear system of equations is solved with a V-Cycle
multigrid algorithm.

## What do I see

Calculated is the heat distribution on a rectangle with the given initial and boundary conditions.

<img src="img/animation_poisson_changing_boundary0.svg" title="rods1" width="400"/>
<em>Temperature distribution on a rod at time <img src="https://latex.codecogs.com/svg.image?t=0" title="t=0" /> juxtaposed with  sinosoidal boundary conditions <img src="https://latex.codecogs.com/svg.image?a(t)=\text{sin}(ct)." title="boundary_conditions" /></em>

<img src="img/animation_poisson_changing_boundary500.svg" title="rods2" width="400"/>
<em>Temperature distribution on the same rod at time <img src="https://latex.codecogs.com/svg.image?t=6.25." title="t=6.25" /></em>


## Why do I want to see this

Play around with different hyperparameters of the V-Cycle multigrid solver to understand its behavior with respect to the problem geometry and boundary conditions.

## Usage
A build script *buildtask_gcc.cmd* is included in this folder. On Windows, you can use it to compile 
the example. Alternatively, you can open the build script, look for the compilation statements and compile it manually.

