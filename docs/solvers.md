---
title: Linear Solvers
layout: default
---

# Linear Solvers

StructGridCUDA implements several linear solvers optimized for GPU execution, each with different strengths for specific problems.

## Available Solvers

### Jacobi Solver

A simple point-iterative method that updates each grid point based on its neighbors:

- **Strengths**: Easy to parallelize, good for diagonal-dominant systems
- **Weaknesses**: Slow convergence for ill-conditioned systems
- **Use Case**: Laplace heat equation, pressure correction

### ADI (Alternating Direction Implicit)

A directional splitting method that solves implicitly along each coordinate direction:

- **Strengths**: Fast convergence for certain problems, requires only tridiagonal solves
- **Weaknesses**: Limited to structured grids, not optimal for all problem types
- **Use Case**: Parabolic problems, transient heat conduction

### CG (Conjugate Gradient)

An iterative method that minimizes the residual using conjugate search directions:

- **Strengths**: Fast convergence for symmetric positive definite systems
- **Weaknesses**: Requires good preconditioning for ill-conditioned problems
- **Use Case**: Pressure Poisson equation, structural mechanics

### Multigrid

A hierarchical solver that addresses multiple error frequencies using grid coarsening:

- **Strengths**: Near-optimal convergence for elliptic problems
- **Weaknesses**: Complex implementation, requires careful tuning
- **Use Case**: Steady-state problems, pressure solvers

## Implementation

All solvers implement a common interface:

```cpp
class LinearSolver {
public:
    virtual void solve(double *x, double *b, int max_iter, double tol) = 0;
    virtual void setup(int nx, int ny, int nz = 1) = 0;
    // Other methods...
};
```

This allows for easy swapping of solvers depending on problem requirements.

## Performance Comparison

| Solver | Relative Speed | Parallelization | Memory Usage |
|--------|---------------|-----------------|--------------|
| Jacobi | Slowest       | Excellent       | Low          |
| ADI    | Medium        | Good            | Low          |
| CG     | Fast          | Good            | Medium       |
| Multigrid | Fastest    | Complex         | High         |

## Usage Example

```cpp
// Create solver (e.g., Jacobi)
JacobiSolver solver;

// Set up the solver
solver.setup(nx, ny);

// Solve the system Ax = b
solver.solve(x, b, 1000, 1e-6);
```

## GPU Optimization

The solvers leverage CUDA-specific optimizations:

1. **Tiled execution** for improved memory access patterns
2. **Shared memory** usage to reduce global memory bandwidth
3. **Warp-level primitives** for efficient reductions
4. **Stream compaction** for sparse operations

These optimizations result in significant speedups compared to CPU implementations.
