#ifndef LID_DRIVEN_CAVITY
#define LID_DRIVEN_CAVITY

#include <string>
#include "LinearSolver.h"
#include "Jacobi.h"
#include "ADI.h"
#include <thrust/device_vector.h>

namespace LidDrivenCavityNS {

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_const(double *T, double val, int nx, int ny);

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, double alpha, int nx, int ny);

// Kernel function for calculation of Jacobian and Residual - No tiling or shared memory
__global__ void compute_mom_r_j(double *T, double *J, double *R, int nx, int ny, double dx, double dy, double kc);

// Kernel function for calculation of Residual - No tiling or shared memory
__global__ void compute_cont_r_j(double *T, double * J, double *R, int nx, int ny, double dx, double dy, double kc);

class LidDrivenCavity {
public:

    // Constructor
    LidDrivenCavity(int nx, int ny, double nu_inp);

    // Destructor
    ~LidDrivenCavity();

    __host__ void initialize_const(double * arr, double val);

    __host__ void update_mom(double alpha_mom = 1.0);

    __host__ void update_pres(double alpha_p = 1.0);

    __host__ double compute_mom_r_j();

    __host__ double compute_cont_r_j();

    __host__ void solve(int nsteps);

    __host__ void solve_mom();

    __host__ void solve_cont();

    int nx;
    int ny;
    double dx;
    double dy;
    double nu;

    double * umom;
    double * vmom;
    double * pres;
    double * phi;
    double * a_inv;
    double * u_nlr;
    double * v_nlr;
    double * cont_nlr;
    double * deltaU;
    double * deltaV;
    double * deltaP;
    double * Jmom;
    double * Jcont;

    thrust::device_ptr<double> t_unlr;
    thrust::device_ptr<double> t_vnlr;
    thrust::device_ptr<double> t_cont_nlr;

    LinearSolverNS::LinearSolver * solver_u;
    LinearSolverNS::LinearSolver * solver_v;
    LinearSolverNS::LinearSolver * solver_p;

private:

    dim3 grid_size;
    dim3 block_size;

    dim3 grid_size_1d;
    dim3 block_size_1d = 1024;

};

}

#endif // LID_DRIVEN_CAVITY