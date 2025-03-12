# Learning CUDA programming to run on structured curvilinear meshes


The goal of this series of exercises is to build a computational aerodynamics solver with the following requirements

1. Solve the steady state incompressible RANS equations on a structured mesh around an airfoil in under 15 seconds. 
2. The airfoil mesh must be an O-type mesh with a single structured block in two dimensions (x,y).
3. The solver must run on Nvidia GPU's using the CUDA programming model in C++.

Since I have no knowledge of programming on GPU's or writing a computational aerodynamics solver from scratch, I have done the following to learn CUDA programming and build expertise in various toolsets required to achieve this objective.

Alternating Direction Implicit solution of linear systems of equations
----------------------------------------------------------------------

The solution of linear system of equations is at the heart of this whole exercise and is expected to consume the largest amount of computational time. To prepare myself, I wrote a simple code in python that solves the laplace equation $k_c \nabla^2 T = k_c(2 + 2 y / 9 )$. The solution to this set of equations is $T = 300.0 + x^2 + (y/3)^3$. These equations are to be solved a on a rectangular domain $(0,0) \leq (x,y) \leq (1,3)$. The boundary conditions are Dirichlet on the left ($x=0$) and bottom ($y=0$) boundaries and specified gradient boundary conditions on the right ($x=1$) and top ($y=3$). The finite volume method is used to discretize the equations on $nx \times ny$ cells. The discretization of the grid is expected to be such that $dx = dy$, so you would need $ny = 3 \; nx$. The discretized equation at each cell becomes
$$
    k_c \left (T_{i,j-1} + T_{i,j+1} + T_{i-1,j} + T_{i+1,j} - 4.0 * T_{i,j} \right ) = k_c \left ( 2.0 + 2.0 * y / 9.0 \right ) dx \; dy.
$$
These equations are solved using the Newton-Raphson method and Alternating Direction Implicity scheme as follows. The residual $R$ at each cell is
$$
 R_{i,j} =  k_c \left (T_{i,j-1} + T_{i,j+1} + T_{i-1,j} + T_{i+1,j} - 4.0 * T_{i,j} \right ) - k_c \left ( 2.0 + 2.0 * y / 9.0 \right ) dx \; dy.
$$
The Newton-Raphson iteration for each compuation of the residual $R$ with a given field $T$ produces an update $\Delta T$ as 
$$
\frac{\partial R}{\partial T} \Delta T = -R.
$$
The Jacobian $ \frac{\partial R}{\partial T} is stored in a 2-D array $J$ of shape $(nx,ny,5)$ where the last index stores the derivatives of the residual $R_{ij}$ with respect to the state variable $T$ at $i,j$, $i-1,j$, $i+1,j$, $i,j-1$, $i,j+1$ in that order. Looking back, I should've probly ordered the $J$ array as $(5,nx,ny)$.

This file can be found at [python/try_adi.py](python/try_adi.py). 

TODO

1. Add plot of residual convergence with iterations and a contour plot of the solution.


ADI in CUDA
-----------

Now, I implement the same ADI algorithm in CUDA C++. Since multi-dimensional arrays are not natively available in CUDA, I use a column major order to store arrays $idx[i,j] = j * nx + i$. For the Jacobian vector $j_idx[i,j,k] = (j * nx + i) * 5 + k$. Since there are no for loops, the block sizes are defined using `#define TILE_SIZE 16`. The same block sizes are used for both $i$ and $j$ directions.

This can be found at [src/try_adi.cu](src/try_adi.cu). 


Moving to curvilinear coordinates
---------------------------------

The logical next step is to make sure that we can compute simple gradients on curvilinear meshes. After thorough consideration of using equations transformed to a cartesian reference coordinate system using Jacobian transformations, I decided to forgo this approach and stick to traditional methods using Green-Gauss theorem as 
$$
 \int_V \frac{\partial \phi}{\partial x} \; \textrm{d}V = \int_S \phi \; ( \hat{i} \cdot \hat{n} ) \; \textrm{d}S.
$$
In discrete terms in two dimensions the partial derivative at the cell center will become,
$$
\frac{\partial \phi}{\partial x} = \frac{1}{A} \sum_{\textrm{faces}} \phi_f  ( \hat{i} \cdot \vec{S} ),
$$
where $\vec{S}$ is the area of the faces and $A$ is the area of the cell. In this first implementation, the value of the field at the cell $\phi_f$ is simply written as the average of the values at the neighboring cell centers as
$$
\phi_f = \frac{1}{2} ( \phi_C + \phi_{\textrm{nei}}).
$$
I read the coordinates into `h_pts` and transfer to device. A field with a known gradient is initialized on device as $\phi = x^2 + y^3$. The reference gradient field is evaluated on device at the cell centers as well. The  output is written into the legacy vtk format. This implementation can be found at [src/airfoil/airfoil.cu](src/airfoil/airfoil.cu).




