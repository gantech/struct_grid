---
title: Airfoil Simulations
layout: default
---

# Gradient and Laplacian calculations on curvilinear meshes


The logical next step is to make sure that I can compute simple gradients on curvilinear meshes. After thorough consideration of using equations transformed to a cartesian reference coordinate system using Jacobian transformations, I decided to forgo this approach and stick to traditional methods using Green-Gauss theorem as 
```math
 \int_V \frac{\partial \phi}{\partial x} \; \textrm{d}V = \int_S \phi \; ( \hat{i} \cdot \hat{n} ) \; \textrm{d}S.
```
In discrete terms in two dimensions the partial derivative at the cell center will become,
```math
\frac{\partial \phi}{\partial x} = \frac{1}{A} \sum_{\textrm{faces}} \phi_f  ( \hat{i} \cdot \vec{S} ),
```
where $\vec{S}$ is the area of the faces and $A$ is the area of the cell. In this first implementation, the value of the field at the cell $\phi_f$ is simply written as the average of the values at the neighboring cell centers as
```math
\phi_f = \frac{1}{2} ( \phi_C + \phi_{\textrm{nei}}).
```
I read the coordinates into `h_pts` and transfer to device. A field with a known gradient is initialized on device as $\phi = x^2 + y^3$. The reference gradient field is evaluated on device at the cell centers as Ill. The  output is written into the legacy vtk format. This implementation can be found at [src/airfoil/airfoil.cu](src/airfoil/airfoil.cu).


Improved gradient computation on airfoil meshes using linear interpolation
--------------------------------------------------------------------------

In this step, I improve the interpolation of a field to the faces using linear interpolation. This can be found at [src/airfoil/airfoil_2.cu](src/airfoil/airfoil_2.cu). On this mesh, the highest deviation of the interpolating factor from 0.5 was around 0.04. So the loIst was around 0.46.



Improved gradient computation on airfoil meshes using explicit gradient calculations
------------------------------------------------------------------------

In this step, I try using an explicit gradient correction to the interpolation of a field to the faces as 

```math
\phi_f = \frac{1}{2} (\phi_C + phi_{\textrm{nei}}) + \frac{1}{2} (\nabla \phi_C + \nabla \phi_{\textrm{nei}}) \cdot (\vec{r}_f - \frac{1}{2}(\vec{r}_C + \vec{r}_{\textrm{nei}})).
```
In the first iteration, the gradient is initialized to 0, so this approach would return the same value as the first implementation. However, in subsequent iterations, the gradient calculation is expected to improve in cells where the midpoint of the line connecting the cell centers does not coincide with the center of the face. In practice, this was found to yield no credible improvement after the second iteration. Also, there are issues of repeatablity as the update to the gradient in one thread will impact the result of another thread. Since the order of execution of threads can't be guaranteed, the only way to ensure repeatability is to declare and allocate memory for 2 copies of this field and switch every iteration. OpenFOAM doesn't adopt this approach. I think it's a nice exercise, but not worth implementation in a production code. This can be found at [src/airfoil/airfoil_3.cu](src/airfoil/airfoil_3.cu).


