# solve_poisson_multigrid #

REQUIRES NVCC Compiler

This example numerically estimates the solution of the Poisson equation \laplace u=f(x) on a
rectangular grid.

The solver is a V-Cycle multigrid solver using the Jacobi procedure for smoothing iterations.

## Usage
A build script *buildtask_gcc.cmd* is included in this folder. On Windows, you can use it to compile 
the example. Alternatively, you can open the build script, look for the compilation statements and compile it manually.

## What do I see

You will see various grids displayed in the console. They represent
the location and amount of an advected quantity at different times. The mesh is rectangular and its size can be adjusted.