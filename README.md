# Learning CUDA programming to run on structured curvilinear meshes


The goal of this series of exercises is to build a computational aerodynamics solver with the following requirements

1. Solve the steady state incompressible RANS equations on a structured mesh around an airfoil in under 15 seconds. 
2. The airfoil mesh must be an O-type mesh with a single structured block in two dimensions (x,y).
3. The solver must run on Nvidia GPU's using the CUDA programming model in C++.

Since I have no knowledge of programming on GPU's or writing a computational aerodynamics solver from scratch, I have done the following to learn CUDA programming and build expertise in various toolsets required to achieve this objective.

Alternating Direction Implicit solution of linear systems of equations
----------------------------------------------------------------------

The solution of linear system of equations is at the heart of this whole exercise and is expected to consume the largest amount of computational time. To prepare myself, I wrote a simple code in python that solves the laplace equation $k_c \nabla^2 T = k_c(2x + 3 y^2)$. The solution to this set of equations is $T = 300.0 + x^2 + y^3$. These equations are to be solved a on a rectangular domain $(0,0) \leq (x,y) \leq (1,3)$. The boundary conditions are Dirichlet on the left ($x=0$) and bottom ($y=0$) boundaries and specified gradient boundary conditions on the right ($x=1$) and top ($y=3$). This file can be found at [python/try_adi.py] and produces the following contour plot.



