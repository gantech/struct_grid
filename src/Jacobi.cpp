#include "Jacobi.h"

namespace LinearSolvers {

    Jacobi(int nx, int ny, double * J, double *T, double *deltaT, double *R):
    LinearSolvers(nx, ny, J, T, deltaT, R)
    {
        cudaMalloc(&deltaT1, nx * ny * sizeof(double));

    }

    ~Jacobi()
    {
        cudaFree(deltaT1);
    }

    __host__ void Jacobi::solve_step() {

        LinearSolvers::jacobi_kernel<<<grid_size, block_size>>>(deltaT, deltaT1, J, R, nx, ny);
        cudaDeviceSynchronize();
        LinearSolvers::jacobi_kernel<<<grid_size, block_size>>>(deltaT1, deltaT, J, R, nx, ny);
        cudaDeviceSynchronize();                
    }

}