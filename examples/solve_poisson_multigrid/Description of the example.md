# solve_poisson_multigrid #

REQUIRES NVCC Compiler

This example numerically estimates the solution of the equation I-dt*v*\laplace u=b on a
rectangular grid.

The solver is a V-Cycle multigrid solver using a weighted Jacobi procedure for smoothing iterations.

## Usage
A build script *buildtask_gcc.cmd* is included in this folder. On Windows, you can use it to compile 
the example. Alternatively, you can open the build script, look for the compilation statements and compile it manually.

## What do I see

You will see a table indicating the l2 error of the estimation with respect to
different problem sizes and multigrid stages.