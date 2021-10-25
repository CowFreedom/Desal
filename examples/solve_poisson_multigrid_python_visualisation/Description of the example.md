# solve_poisson_multigrid #


This example numerically solves

<img src="https://latex.codecogs.com/svg.image?\frac{\partial&space;\textbf{u}}{\partial&space;t}=v\Delta\textbf{u}" title="\frac{\partial \textbf{u}}{\partial t}=v\Delta\textbf{u}" />

on a rectangular grid

<img src="https://latex.codecogs.com/svg.image?D\subset&space;\mathbb{R}^2:=\left&space;[&space;0,\text{width}&space;\right&space;]&space;\times&space;\left&space;[&space;0,\text{height}&space;\right&space;]" title="D\subset \mathbb{R}^2:=\left [ 0,\text{width} \right ] \times \left [ 0,\text{height} \right ]" />

with Dirichlet boundary conditions

<img src="https://latex.codecogs.com/svg.image?\textbf{u}\left&space;(t,&space;\textbf{x}&space;\right&space;)=k,&space;\quad&space;\quad&space;k,t\in&space;\mathbb{R},\textbf{x}&space;\in&space;\partial&space;D" title="\textbf{u}\left (t, \textbf{x} \right )=k, \quad \quad k,t\in \mathbb{R},\textbf{x} \in \partial D" />.

Common names for this problem are diffusion or heat equation. The heat equation can be solved numerically by a first order approximation of the form

<img src="https://latex.codecogs.com/svg.image?\left&space;(&space;&space;\textbf{I}-\frac{v\cdot&space;dt}{h_xh_y}&space;\textbf{M}\right)\textbf{u}_{n}=\textbf{u}_{n&plus;1}" title="\left ( \textbf{I}-\frac{v\cdot dt}{h_xh_y} \textbf{M}\right)\textbf{u}_{n}=\textbf{u}_{n+1}" />

whereas the Laplace operator is discretized as

<img src="https://latex.codecogs.com/svg.image?\left&space;(&space;&space;\textbf{I}-v\cdot&space;dt&space;\Delta\right)\textbf{u}_{n}&space;=&space;\left&space;(&space;&space;\textbf{I}-\frac{v\cdot&space;dt}{h_xh_y}&space;\textbf{M}\right)\textbf{u}_{n}&plus;O\left&space;(&space;h^{-2}&space;\right&space;)" title="\left ( \textbf{I}-v\cdot dt \Delta\right)\textbf{u}_{n} = \left ( \textbf{I}-\frac{v\cdot dt}{h_xh_y} \textbf{M}\right)\textbf{u}_{n}+O\left ( h^{-2} \right )" />

with identity matrix <img src="https://latex.codecogs.com/svg.image?\textbf{I}" title="\textbf{I}" /> and finite difference matrix <img src="https://latex.codecogs.com/svg.image?\textbf{M}" title="\textbf{M}" />.

The name of this example is derived from the circumstance that at each time point <img src="https://latex.codecogs.com/svg.image?t" title="t" /> a Poisson equation is solved. This linear system of equations is solved with a V-Cycle
multigrid algorithm.

## What do I see

Calculated is the heat distribution on a rectangle with Dirichlet boundary conditions. The results are also written into the output folder.
different problem sizes and multigrid stages.

<img src="img/rods1.svg" title="rods1" width="400"/>
<em>Temperature distribution on trod at time <img src="https://latex.codecogs.com/svg.image?t=0" title="t=0" /> and <img src="https://latex.codecogs.com/svg.image?t=0" title="t=50" /> with  <img src="https://latex.codecogs.com/svg.image?v=0.25" title="v=0.25" />. </em>


## Why do I want to see this

Play around with different hyperparameters of the V-Cycle multigrid solver to understand its behavior with respect to the problem geometry and boundary conditions.

## Usage
A build script *buildtask_gcc.cmd* is included in this folder. On Windows, you can use it to compile 
the example. Alternatively, you can open the build script, look for the compilation statements and compile it manually.

