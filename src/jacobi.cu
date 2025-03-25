#include "Jacobi.h"

namespace JacobiNS {

// Kernel function for Jacobi smoother - No tiling or shared memory
__global__ void jacobi_kernel(double *deltaT, double * deltaT1, double *J, double *R, int nx, int ny) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (( i < nx) && (j < ny)) {
        
        int idx_r = (j * nx) + i;
        int idx_j = idx_r * 5;

        double jij = J[idx_j];
        double jim1j = J[idx_j + 1];
        double jip1j = J[idx_j + 2];
        double jijm1 = J[idx_j + 3];
        double jijp1 = J[idx_j + 4];

        double tip1j = 0.0;
        double tim1j = 0.0;
        double tijp1 = 0.0;
        double tijm1 = 0.0;

        if (i == 0) {
            tip1j = deltaT[idx_r + 1];
        } else if (i == (nx - 1)) {
            tim1j = deltaT[idx_r - 1];
        } else {
            tip1j = deltaT[idx_r + 1];
            tim1j = deltaT[idx_r - 1];
        }

        if (j == 0) {
            tijp1 = deltaT[idx_r + nx];
        } else if (j == (ny - 1)) {
            tijm1 = deltaT[idx_r - nx];
        } else {
            tijm1 = deltaT[idx_r - nx];
            tijp1 = deltaT[idx_r + nx];
        }

        deltaT1[idx_r] = (R[idx_r] - jim1j * tim1j - jip1j * tip1j - jijm1 * tijm1 - jijp1 * tijp1) / jij;
        
    } 

}

    Jacobi::Jacobi(int nx, int ny, double * J, double *T, double *deltaT, double *R):
    LinearSolvers::LinearSolvers(nx, ny, J, T, deltaT, R)
    {
        cudaMalloc(&deltaT1, nx * ny * sizeof(double));

    }

    Jacobi::~Jacobi()
    {

        cudaFree(deltaT1);
    }

    __host__ void Jacobi::solve_step() {

        jacobi_kernel<<<grid_size, block_size>>>(deltaT, deltaT1, J, R, nx, ny);
        cudaDeviceSynchronize();
        jacobi_kernel<<<grid_size, block_size>>>(deltaT1, deltaT, J, R, nx, ny);
        cudaDeviceSynchronize();                
    }

}