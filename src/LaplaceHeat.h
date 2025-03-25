#ifndef LAPLACE_HEAT
#define LAPLACE_HEAT

#include "LinearSolver.h"
#include "Jacobi.h"

namespace LaplaceHeat {

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_const(double *T, double val, int nx, int ny, double dx, double dy);

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_ref(double *T, int nx, int ny, double dx, double dy);

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny, double dx, double dy);

// Kernel function for calculation of Jacobian and Residual - No tiling or shared memory
__global__ void compute_r_j(double *T, double *J, double *R, int nx, int ny, double dx, double dy, double kc);

// Kernel function for calculation of Residual - No tiling or shared memory
__global__ void compute_r(double *T, double * J, double *R, int nx, int ny, double dx, double dy, double kc);

// Kernel function for computation of Matrix Vector product J * v
// This function is likely needed to be passed to any LinearSolver class
__global__ void compute_matvec(double * v, double * J, double * result, int nx, int ny);

// Kernel function for Jacobi smoother - No tiling or shared memory
__global__ void jacobi(double *deltaT, double * deltaT1, double *J, double *R, int nx, int ny);

class LaplaceHeat {
public:

    // Constructor
    LaplaceHeat(int nx, int ny, double kc);

    // Destructor
    ~LaplaceHeat();

    __host__ void initialize_const(double val);

    __host__ void initialize_ref();

    __host__ void update();

    __host__ double compute_r_j();

    __host__ double compute_r();

    __host__ void compute_matvec(double *v, double * result);

    __host__ void solve(int nsteps);

    int nx;
    int ny;
    double dx;
    double dy;
    double kc;

    double * T;
    double * nlr;
    double * deltaT;
    double * J;

    LinearSolver::LinearSolver * solver;

private:

    dim3 grid_size;
    dim3 block_size;

    dim3 grid_size_1d;
    dim3 block_size_1d = 1024;

};

}

#endif // LAPLACE_HEAT