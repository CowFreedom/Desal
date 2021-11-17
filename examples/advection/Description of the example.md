# solve_poisson_multigrid #


This example computes advection numerically as

<img src="https://latex.codecogs.com/svg.image?\mathbf{u} \left(t+\delta t, \mathbf{x})\right)=\mathbf{u} \left(t, \mathbf{x}-\delta t \mathbf{u}(t,\mathbf{x})\right)" title="\frac{\partial \textbf{u}}{\partial t}=v\Delta\textbf{u}" />

on a rectangular grid

<img src="https://latex.codecogs.com/svg.image?D\subset&space;\mathbb{R}^2:=\left&space;[&space;0,\text{width}&space;\right&space;]&space;\times&space;\left&space;[&space;0,\text{height}&space;\right&space;]" title="D\subset \mathbb{R}^2:=\left [ 0,\text{width} \right ] \times \left [ 0,\text{height} \right ]" />


## What do I see

Output to file is the transport of a quantity along a rectangular region at different time stages.

## Usage
A build script *buildtask_gcc.cmd* is included in this folder. On Windows, you can use it to compile 
the example. Alternatively, you can open the build script, look for the compilation statements and compile it manually.

