#include "Jacobi.h"
#include <iostream>

namespace JacobiNS {

// Kernel function for Jacobi smoother - No tiling or shared memory
__global__ void jacobi_kernel(double *deltaT, double * deltaT1, double *J, double *R, int nx, int ny) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col < nx) && (row < ny)) {
        
        int idx_r = (row * nx) + col;
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

        if (col == 0) {
            tip1j = deltaT[idx_r + 1];
        } else if (col == (nx - 1)) {
            tim1j = deltaT[idx_r - 1];
        } else {
            tip1j = deltaT[idx_r + 1];
            tim1j = deltaT[idx_r - 1];
        }

        if (row == 0) {
            tijp1 = deltaT[idx_r + nx];
        } else if (row == (ny - 1)) {
            tijm1 = deltaT[idx_r - nx];
        } else {
            tijm1 = deltaT[idx_r - nx];
            tijp1 = deltaT[idx_r + nx];
        }

        double tmp = (R[idx_r] - jim1j * tim1j - jip1j * tip1j - jijm1 * tijm1 - jijp1 * tijp1) / jij;

        deltaT1[idx_r] = tmp;
        
        if (std::isnan(tmp)) {
           printf("Row, Col is %d, %d, %d, %d- deltaT =  %f, J - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f, T - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f \n", row, col, blockIdx.x, blockIdx.y, tmp, jijm1, jijp1, jim1j, jip1j, jij, tijm1, tijp1, tim1j, tip1j, deltaT[idx_r]);
        }

    } 

}

    Jacobi::Jacobi(int nx, int ny, double * J, double *T, double *deltaT, double *R):
    LinearSolver::LinearSolver(nx, ny, J, T, deltaT, R)
    {
        cudaMalloc(&deltaT1, nx * ny * sizeof(double));

    }

    Jacobi::~Jacobi()
    {

        cudaFree(deltaT1);
    }

    __host__ void Jacobi::solve_step(int nsteps) {

        for (int istep = 0; istep < nsteps; istep++) {
            jacobi_kernel<<<grid_size, block_size>>>(deltaT, deltaT1, J, R, nx, ny);
            cudaDeviceSynchronize();
            jacobi_kernel<<<grid_size, block_size>>>(deltaT1, deltaT, J, R, nx, ny);
            cudaDeviceSynchronize();
        }
    }

}
