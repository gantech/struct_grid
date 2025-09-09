# Learning CUDA programming to run on structured curvilinear meshes


The goal of this series of exercises is to build a computational aerodynamics solver with the following requirements

1. Solve the steady state incompressible RANS equations on a structured mesh around an airfoil in under 15 seconds. 
2. The airfoil mesh must be an O-type mesh with a single structured block in two dimensions (x,y).
3. The solver must run on Nvidia GPU's using the CUDA programming model in C++.


Here's what exists so far:

1. Capability to solve the Laplace heat equation with a nonlinear source term that drives a desired solution using method of manufactured solutions. Uses an object oriented approach to selecting different solvers for the Laplace heat problem and compare them.

2. Algorithms to calculate gradient using Gauss method as well as a Laplacian on curvilinear structured meshes.