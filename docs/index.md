---
title: StructGridCUDA Documentation
language_tabs:
  - cpp: C++/CUDA
  - bash: Shell

toc_footers:
  - <a href='https://github.com/gantech/struct_grid'>View on GitHub</a>

search: true
---

# Learning CUDA programming to run on structured curvilinear meshes

I see the available FLOPs numbers on modern GPU's with so many zeros and they don't line up with the actual run times of computational aerodynamics solvers in vogue with the exception of few modern GPU-first solvers like Flexcompute and Luminary cloud. I think we should expect transformational reductions in solve times by several orders of magnitude. I want to take a crack at making this happen. The worst that could happen is I'll fail! Meh...I'll learn something in the process. Strategically, I think it's best to create a proof of concept on this small problem with hope for large impact down the line. With that thought process written up and out of the way, the goal of this repo is to implement a series of exercises towards building a computational aerodynamics solver with the following requirements:

1. Solve the steady state incompressible RANS equations on a structured mesh around an airfoil in under 15 seconds. 
2. The airfoil mesh must be an O-type mesh with a single structured block in two dimensions (x,y).
3. The solver must run on Nvidia GPU's using the CUDA programming model in C++.

Since I have no prior knowledge of programming on GPU's or writing a computational aerodynamics solver from scratch, I have done the following to learn CUDA programming and build expertise in various toolsets required to achieve this objective including linear solvers. I don't have forever to get to MVP to learn everything needed in a linear fashion. So this has been a fairly "nonlinear" process for lack of a better word.

1. Solve the Laplace heat equation with a nonlinear source term that drives a desired solution using method of manufactured solutions. For some reason Alternating Direction Implicit schemes have always attracted my attention for computational aerodynamics on RANS grids. I assumed that a direct solver in the wall-normal direction on stretched grids will help overcome the stiffness of the assembled RANS equations in the most efficient manner. So I implemented this first.

2. Implement algorithms to calculate gradient using Gauss method as well as a Laplacian on curvilinear mesh around airfoil. Details are available [here](airfoil.md)

3. Realize that achieving computational efficiency on CUDA requires a different thinking and read up on blog posts by [Simon Boehm](https://siboehm.com/articles/22/CUDA-MMM), [Alex Armbruster](https://alexarmbr.github.io/2024/08/10/How-To-Write-A-Fast-Matrix-Multiplication-From-Scratch-With-Tensor-Cores.html), [Aman Salykova](https://salykova.github.io/sgemm-gpu). Realize CFD algorithms will always be memory bound using arithmetic intensity calculations. Implement [convolution in CUDA](https://gantech.github.io/convolution_cuda/) as a stepping stone to implementing CFD algorithms.

3. Finish implementing an object oriented approach to selecting different solvers for the Laplace heat problem and compare them. Details are available [here](laplace_heat.md).

4. Work in progress: Implementing an incompressible laminar flow CFD solver to solve the lid-driven cavity problem.

5. Future work: Implement laminar flow incompressible solver around airfoil.

6. Future work: Add turbulent flow capability for solver around airfoil.




