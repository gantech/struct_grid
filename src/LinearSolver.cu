#include "LinearSolver.h"
#include <cmath>
#include <iostream>
namespace LinearSolverNS {

// Kernel to compute matrix vector product of the linear system of equations J * v . 
__global__ void compute_matvec(double * v, double * J, double * result, int nx, int ny) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx_r = (j * nx) + i;
    int idx_j = idx_r * 5;

    if ( (i < nx) && (j < ny)) {

        double jij = J[idx_j];
        double jim1j = J[idx_j + 1];
        double jip1j = J[idx_j + 2];
        double jijm1 = J[idx_j + 3];
        double jijp1 = J[idx_j + 4];

        double vip1j = 0.0;
        double vim1j = 0.0;
        double vijp1 = 0.0;
        double vijm1 = 0.0;

        if ( i == 0) {
            vip1j = v[idx_r + 1];
        } else if ( i == (nx - 1)) {
            vim1j = v[idx_r - 1];
        } else {
            vip1j = v[idx_r + 1];
            vim1j = v[idx_r - 1];
        }

        if ( j == 0) {
            vijp1 = v[idx_r + nx];
        } else if ( j == (ny - 1)) {
            vijm1 = v[idx_r - nx];
        } else {
            vijm1 = v[idx_r - nx];
            vijp1 = v[idx_r + nx];
        }

        result[idx_r] = jim1j * vim1j + jip1j * vip1j + jijm1 * vijm1 + jijp1 * vijp1 + jij * v[idx_r];
    }
}

// Kernel to compute residual of linear system of equations R - J * deltaT
__global__ void compute_linresid(double * deltaT, double * J, double * R, double * lin_resid, int nx, int ny) {



    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx_r = (j * nx) + i;
    int idx_j = idx_r * 5;

    if ( (i < nx) && (j < ny)) {

        double jij = J[idx_j];
        double jim1j = J[idx_j + 1];
        double jip1j = J[idx_j + 2];
        double jijm1 = J[idx_j + 3];
        double jijp1 = J[idx_j + 4];

        double deltaTip1j = 0.0;
        double deltaTim1j = 0.0;
        double deltaTijp1 = 0.0;
        double deltaTijm1 = 0.0;

        if ( i == 0) {
            deltaTip1j = deltaT[idx_r + 1];
        } else if ( i == (nx - 1)) {
            deltaTim1j = deltaT[idx_r - 1];
        } else {
            deltaTip1j = deltaT[idx_r + 1];
            deltaTim1j = deltaT[idx_r - 1];
        }

        if ( j == 0) {
            deltaTijp1 = deltaT[idx_r + nx];
        } else if ( j == (ny - 1)) {
            deltaTijm1 = deltaT[idx_r - nx];
        } else {
            deltaTijm1 = deltaT[idx_r - nx];
            deltaTijp1 = deltaT[idx_r + nx];
        }

        lin_resid[idx_r] = R[idx_r] - (jim1j * deltaTim1j + jip1j * deltaTip1j + jijm1 * deltaTijm1 + jijp1 * deltaTijp1 + jij * deltaT[idx_r]);
    }
}

    LinearSolver::LinearSolver(int nxinp, int nyinp, 
        double * Jinp, double *Tinp, double *deltaTinp, double *Rinp):
    nx(nxinp), ny(nyinp), J(Jinp), T(Tinp), deltaT(deltaTinp), R(Rinp) {

        grid_size = dim3(std::ceil(nx/TILE_SIZE), std::ceil(ny/TILE_SIZE));
        grid_size_1d = dim3( std::ceil (nx * ny / 1024.0) );
        block_size = dim3(TILE_SIZE, TILE_SIZE, 1);


    }

    void LinearSolver::matvec(double * v, double * result) {
        compute_matvec<<<grid_size, block_size>>>(v, J, result, nx, ny);
        cudaDeviceSynchronize();
    }

    void 
    LinearSolver::linresid(double * lin_resid) {
        compute_linresid<<<grid_size, block_size>>>(deltaT, J, R, lin_resid, nx, ny);
        cudaDeviceSynchronize();
    }
        
    
}
