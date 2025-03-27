#include "CG.h"
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <cmath>

namespace CGNS {

typedef exec_matvec()
// Functor to square the elements
struct square {
    __device__ double operator()(double a) {
        return a * a;
    }
};

// Kernel function for initialization - No tiling or shared memory
// This is copied from LaplaceHeat - TODO: Put this into a utils library.
// Unfortunately cudaMemset only works on bytes 
__global__ void initialize_const(double *T, double val, int ntot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ntot ) 
        T[idx] = val;
}

// Kernel function to update deltaT and R using alpha in Conjugate Gradient method
__global__ void update_deltat_r(double *deltaT, double * pvec, double * R, double * jpvec, double alpha, int ntot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ntot ) {
        deltaT[idx] += alpha * pvec[idx];
        R[idx] -= alpha * jpvec[idx];
    }
}

// Kernel function to update search direction using beta in Conjugate Gradient method
__global__ void update_searchdir(double * pvec, double * R, double beta, int ntot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ntot )
        pvec[idx] = r[idx] + beta * pvec[idx];
}


    CG::CG(int nx, int ny, double * J, double *T, double *deltaT, double *R):
    LinearSolver::LinearSolver(nx, ny, J, T, deltaT, R),
    ntot(nx * ny)
    {
        cudaMalloc(&pvec, ntot * sizeof(double));
        cudaMalloc(&Jpvec, ntot * sizeof(double));

        cudaMemcpy(pvec, R, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        initialize_const(deltaT, 0.0, ntot);

        t_pvec = thrust::device_ptr(pvec);
        t_jpvec = thrust::device_ptr(jpvec);
        t_resid = thrust::device_ptr(R);

        grid_size_1d = dim3( std::ceil (ntot / 1024.0) );

    }

    CG::~CG()
    {
        cudaFree(pvec);
        cudaFree(Jpvec);
    }

    __host__ void CG::solve_step() {

        /*
        0.a Compute matvec Jpvec = J * pvec
        0.b Compute r^T . r and store this
        1. Compute alpha
        2. Update deltaT and R
        3. Optionally precondition residual R
        4. Compute beta
        5. Update search dir `pvec`

        Steps 1 and 2 involves a matvec which needs to happen through a call to the physics class. This needs to happen first
        Steps 1 and 4 need to happen here through calls to thrust library.
        Steps 2 and 3 can be combined into 1 kernel - update_deltat_r
        Step 5 needs another kernel - update_searchdir

        */

        double rsqr = thrust::transform_reduce(t_resid, t_resid + ntot, square(), 0.0, thrust::plus<double>());
        double alpha_denom = thrust::inner_product(t_pvec, t_pvec + ntot, t_jpvec, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
        double alpha = rsqr / alpha_denom;
        update_deltat_r<<<grid_size_1d, 1024>>>(deltaT, pvec, R, jpvec, alpha, ntot);
        // In the future insert preconditioner here
        double rnewsqr = thrust::transform_reduce(t_resid, t_resid + ntot, square(), 0.0, thrust::plus<double>());
        double beta = rnewsqr / rsqr;
        update_searchdir<<<grid_size_1d, 1024>>>(pvec, R, beta, ntot);
        
    }

}