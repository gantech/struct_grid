#include "BiCGStab.h"
#include <cmath>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/inner_product.h>

namespace BiCGStabNS {

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

// Kernel function to update deltaT and R using alpha in Stabilized Bi-Conjugate Gradient method
__global__ void update_deltat_r(double *deltaT, double * pvec, double * R, double * svec, double *tvec, double alpha, double omegak, int ntot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ntot ) {
        deltaT[idx] += alpha * pvec[idx] + omegak * svec[idx];
        R[idx] = svec[idx] - omegak * tvec[idx];
    }
}

// Kernel function to calculate svec
__global__ void calc_svec(double * R, double * jpvec, double * svec, double alpha, int ntot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ntot )
        svec[idx] = R[idx] - alpha * jpvec[idx];
}

// Kernel function to update search direction using beta in Stabilized Bi-Conjugate Gradient method
__global__ void update_searchdir(double * pvec, double * R, double * jpvec, double beta, double omegakm1, int ntot) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ntot )
        pvec[idx] = R[idx] + beta * (pvec[idx] - omegakm1 * jpvec[idx]);
}


    BiCGStab::BiCGStab(int nx, int ny, double * J, double *deltaT, double *R):
    LinearSolver::LinearSolver(nx, ny, J, deltaT, R),
    ntot(nx * ny)
    {
        cudaMalloc(&pvec, ntot * sizeof(double));
        cudaMalloc(&jpvec, ntot * sizeof(double));
        cudaMalloc(&resid0, ntot * sizeof(double));
        cudaMalloc(&svec, ntot * sizeof(double));
        cudaMalloc(&tvec, ntot * sizeof(double));

        grid_size_1d = std::ceil (ntot / 1024.0);

        initialize_const<<<grid_size_1d, 1024>>>(deltaT, 0.0, ntot);

        t_pvec = thrust::device_pointer_cast(pvec);
        t_jpvec = thrust::device_pointer_cast(jpvec);
        t_resid = thrust::device_pointer_cast(R);
        t_resid0 = thrust::device_pointer_cast(resid0);
        t_svec = thrust::device_pointer_cast(svec);
        t_tvec = thrust::device_pointer_cast(tvec);


    }

    BiCGStab::~BiCGStab()
    {
        cudaFree(pvec);
        cudaFree(jpvec);
        cudaFree(resid0);
        cudaFree(svec);
        cudaFree(tvec);
    }

    __host__ void BiCGStab::solve_step(int nsteps) {

        initialize_const<<<grid_size_1d, 1024>>>(deltaT, 0.0, ntot);
        cudaMemcpy(resid0, R, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(pvec, R, nx * ny * sizeof(double), cudaMemcpyDeviceToDevice);
        double rhokm1 = 1.0, rhok = 1.0, alpha = 1.0, omegak = 1.0, omegakm1 = 1.0, beta = 1.0;
        matvec(pvec, jpvec);
        for (int istep = 0; istep < nsteps; istep++) {
            rhok = thrust::inner_product(t_resid0, t_resid0 + ntot, t_resid, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
            double beta = (rhok / rhokm1) * (alpha / omegakm1);
            // pvec = R + beta * (pvec - omegakm1 * jpvec)
            update_searchdir<<<grid_size_1d, 1024>>>(pvec, R, jpvec, beta, omegakm1, ntot);
            matvec(pvec, jpvec);
            alpha = rhok / thrust::inner_product(t_jpvec, t_jpvec + ntot, t_resid0, 0.0, thrust::plus<double>(), thrust::multiplies<double>());
            // Calc s = R - alpha * jpvec
            calc_svec<<<grid_size_1d, 1024>>>(R, jpvec, svec, alpha, ntot);
            matvec(svec, tvec);
            omegak = thrust::inner_product(t_tvec, t_tvec + ntot, t_svec, 0.0, thrust::plus<double>(), thrust::multiplies<double>()) /
                     thrust::transform_reduce(t_tvec, t_tvec + ntot, square(), 0.0, thrust::plus<double>());
            // Update deltaT = deltaT + alpha * pvec + omegak * s
            // Update R = s - omegak * tvec
            // Note: We combine the above two steps into one kernel to save memory bandwidth
            update_deltat_r<<<grid_size_1d, 1024>>>(deltaT, pvec, R, svec, tvec, alpha, omegak, ntot);
            rhokm1 = rhok;
            omegakm1 = omegak;
        }
        
    }

}
