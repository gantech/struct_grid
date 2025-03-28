#include "Multigrid.h"
#include "Jacobi.h"
#include "CG.h"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>


namespace MultiGridNS {

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

// Restrict residual by one level. Expected that nxc = nxf/2 and nyc = nyf/2
// Expected to run on a grid and block that represents the coarse mesh
__global__ void restrict_resid(double * rc, double * rf, int nxc, int nyc, int nxf, int nyf) {

    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    int jc = blockIdx.y * blockDim.y + threadIdx.y;

    // Each cell in the coarse mesh (ic, jc) is a sum of the 4 cells corresponding to (2 * ic, 2 * jc), (2 * ic + 1, 2 * jc), (2 * ic, 2 * jc + 1), (2 * ic + 1, 2 * jc + 1)
    int idx_rc = (jc * nxc) + ic;
    int idx_rf1 = (2 * jc * nxf) + (2 * ic);
    int idx_rf2 = (2 * jc * nxf) + (2 * ic + 1);
    int idx_rf3 = (2 * jc + 1) * nxf + (2 * ic);
    int idx_rf4 = (2 * jc + 1) * nxf + (2 * ic + 1);

    if ( (ic < nxc) && (jc < nyc) ) 
        rc[idx_rc] = (rf[idx_rf1] + rf[idx_rf2] + rf[idx_rf3] + rf[idx_rf4]);//std::sqrt(2.0);

}

// Prolongate error by one level. Expected that nxc = nxf/2 and nyc = nyf/2
// Expected to run on a grid and block that represents the coarse mesh
__global__ void prolongate_error(double * deltaTc, double * deltaTf, int nxc, int nyc, int nxf, int nyf) {

    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    int jc = blockIdx.y * blockDim.y + threadIdx.y;

    // Each cell in the coarse mesh (ic, jc) is a sum of the 4 cells corresponding to (2 * ic, 2 * jc), (2 * ic + 1, 2 * jc), (2 * ic, 2 * jc + 1), (2 * ic + 1, 2 * jc + 1)
    int idx_rc = (jc * nxc) + ic;
    int idx_rf1 = (2 * jc * nxf) + (2 * ic);
    int idx_rf2 = (2 * jc * nxf) + (2 * ic + 1);
    int idx_rf3 = (2 * jc + 1) * nxf + (2 * ic);
    int idx_rf4 = (2 * jc + 1) * nxf + (2 * ic + 1);

    if ( (ic < nxc) && (jc < nyc) ) {

        //printf("Prolongating (i,j), idx_rc = %d, %d, %d deltaT = %e, idx_rf1 = %d, idx_rf2 = %d, idx_rf3 = %d, idx_rf4 = %d \n", ic, jc, idx_rc, deltaTc[idx_rc], idx_rf1, idx_rf2, idx_rf3, idx_rf4);
        deltaTf[idx_rf1] += 0.9*deltaTc[idx_rc];//std::sqrt(2.0);
        deltaTf[idx_rf2] += 0.9*deltaTc[idx_rc];//std::sqrt(2.0);
        deltaTf[idx_rf3] += 0.9*deltaTc[idx_rc];//std::sqrt(2.0);
        deltaTf[idx_rf4] += 0.9*deltaTc[idx_rc];//std::sqrt(2.0);
        
    }    

}


// Create Jacobian matrix at the coarser level by using the finer level. Expected that nxc = nxf/2 and nyc = nyf/2
// Expected to run on a grid and block that represents the coarse mesh
__global__ void restrict_j(double * jc, double * jf, int nxc, int nyc, int nxf, int nyf)  {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Each cell in the coarse mesh (ic, jc) is a sum of the 4 cells corresponding to (2 * ic, 2 * jc), (2 * ic + 1, 2 * jc), (2 * ic, 2 * jc + 1), (2 * ic + 1, 2 * jc + 1)
    int idx_jc = ((j * nxc) + i) * 5;
    int idx_jf1 = ((2 * j * nxf) + (2 * i)) * 5;
    int idx_jf2 = ((2 * j * nxf) + (2 * i + 1)) * 5;
    int idx_jf3 = ((2 * j + 1) * nxf + (2 * i)) * 5;
    int idx_jf4 = ((2 * j + 1) * nxf + (2 * i + 1)) * 5;

    if ( (i < nxc) && (j < nyc) ) {

        // Diagonals and Interlinks of the 4 cells
        jc[idx_jc] = (jf[idx_jf1] + jf[idx_jf2] + jf[idx_jf3] + jf[idx_jf4]    + jf[idx_jf1+2] + jf[idx_jf1+4]    + jf[idx_jf2+1] + jf[idx_jf2+4]    + jf[idx_jf3+3] + jf[idx_jf3+2]    + jf[idx_jf4+1] + jf[idx_jf4+3]);
        jc[idx_jc+1] = (jf[idx_jf1+1] + jf[idx_jf3+1]);
        jc[idx_jc+2] = (jf[idx_jf2+2] + jf[idx_jf4+2]);
        jc[idx_jc+3] = (jf[idx_jf1+3] + jf[idx_jf2+3]);
        jc[idx_jc+4] = (jf[idx_jf3+4] + jf[idx_jf4+4]);
        
    }

}

MultiGrid::MultiGrid(int nx, int ny, double * J, double *T, double *deltaT, double *R, int nlevels_inp, std::string bottom_solver_inp):
LinearSolver::LinearSolver(nx, ny, J, T, deltaT, R),
nlevels(nlevels)
{

    nxl.resize(nlevels);
    nyl.resize(nlevels);
    
    smoothers.resize(nlevels);
    Jmg.resize(nlevels);
    Rmg.resize(nlevels);
    deltaTmg.resize(nlevels);
    Rlinmg.resize(nlevels);
    
    smoothers.push_back(new JacobiNS::Jacobi(nx, ny, J, T, deltaT, R));
    nxl[0] = nx;
    nyl[0] = ny;
    grid_size_mg.push_back(dim3(ceil(nx / (double)TILE_SIZE), ceil(ny / (double)TILE_SIZE), 1));

    
    Jmg[0] = J;
    Rmg[0] = R;
    deltaTmg[0] = deltaT;
    cudaMalloc(&Rlinmg[0], nx * ny * sizeof(double));
    
    for(int ilevel = 1; ilevel < nlevels; ilevel++) {
        nxl[ilevel] = nx / (1 << ilevel);
        nyl[ilevel] = ny / (1 << ilevel);
        cudaMalloc(&Jmg[ilevel], nxl[ilevel] * nyl[ilevel] * 5 * sizeof(double));
        cudaMalloc(&Rmg[ilevel], nxl[ilevel] * nyl[ilevel] * sizeof(double));
        cudaMalloc(&deltaTmg[ilevel], nxl[ilevel] * nyl[ilevel] * sizeof(double));
        cudaMalloc(&Rlinmg[ilevel], nxl[ilevel] * nyl[ilevel] * sizeof(double));

        grid_size_mg.push_back(dim3(ceil(nxl[ilevel] / (double)TILE_SIZE), ceil(nyl[ilevel] / (double)TILE_SIZE), 1));
    }

    for (int ilevel = 0; ilevel < (nlevels-1); ilevel++)
        smoothers.push_back(new JacobiNS::Jacobi(nxl[ilevel], nyl[ilevel], Jmg[ilevel], Rmg[ilevel], deltaTmg[ilevel], Rlin[ilevel]));        
    
    if (bottom_solver == "Conjugate Gradient")
        smoothers.push_back(new ConjugateGradientNS::ConjugateGradient(nxl[nlevels-1], nyl[nlevels-1], Jmg[nlevels-1], Rmg[nlevels-1], deltaTmg[nlevels-1], Rlin[nlevels-1]));
    else
        smoothers.push_back(new JacobiNS::Jacobi(nxl[nlevels-1], nyl[nlevels-1], Jmg[nlevels-1], Rmg[nlevels-1], deltaTmg[nlevels-1], Rlin[nlevels-1]));
    
    // Create restricted Jacobian matrices at each coarse level
    for (int ilevel = 1; ilevel < nlevels; ilevel++) {
        restrict_j<<<grid_size_mg[ilevel], block_size>>>(Jmg[ilevel-1], Jmg[ilevel], nxl[ilevel], nyl[ilevel], nxl[ilevel-1], nyl[ilevel-1]);
        cudaDeviceSynchronize();
    }

}

MultiGrid::~MultiGrid()
{

    delete smoothers[0];
    cudaFree(Rlinmg[0]);
    for(int ilevel = 1; ilevel < nlevels-1; ilevel++) {
        delete smoothers[ilevel];
        cudaFree(Jmg[ilevel]);
        cudaFree(Rmg[ilevel]);
        cudaFree(deltaTmg[ilevel]);
        cudaFree(Rlinmg[ilevel]);
    }

}


/*

    Sub-steps for 1 step of multigrid

    1. Initialize deltaT to zero at all levels
    2. Do some smoothing on the finest level first
    3. Compute Rlin at finest level
    4. For (ilevel = 1; ilevel < nlevels-1; ilevel++)
        Restrict residual to coarser level
        Do some smoothing at this level to get the error
        Compute linear residual
    5. Restrict residual to the coarsest level
    6. Do bottom level solve with user-specified choice of solver
    7. For (ilevel = nlevels-2; ilevel > 0; ilevel--)
        Prolongate error from ilevel+1 to ilevel
        Do some more smoothing at ilevel
    8. Prolongate error to finest level
    9. Do more smoothing at the finest level

    
*/    

void MultiGrid::solve_step() {

    initialize_const<<<grid_size_mg_1d[0], block_size_1d>>>(deltaT, 0.0, nxl[0] * nyl[0]);
    cudaDeviceSynchronize();
    for (int ilevel=1; ilevel < nlevels; ilevel++) {
        initialize_const<<<grid_size_mg_1d[ilevel], block_size_1d>>>(deltaTmg[ilevel-1], 0.0, nxl[ilevel] * nyl[ilevel]);
        cudaDeviceSynchronize();
    }

    smoothers[0].solve(10);
    linresid(Rlin);

    for (int ilevel = 1; ilevel < nlevels-1; ilevel++) {
        restrict_resid<<<grid_size_mg[ilevel], block_size>>>(Rmg[ilevel], Rlin[ilevel-1], nxl[ilevel], nyl[ilevel], nxl[ilevel-1], nyl[ilevel-1]);
        cudaDeviceSynchronize();
        smoothers[ilevel]->solve(10);
        smoothers[ilevel]->linresid(Rlinmg[ilevel]);
    }

    restrict_resid<<<grid_size_mg[nlevels-1], block_size>>>(Rmg[nlevels-1], Rlin[nlevels-2], nxl[nlevels-1], nyl[nlevels-1], nxl[nlevels-2], nyl[nlevels-2]);
    cudaDeviceSynchronize();
    smoothers[nlevels-1]->solve(10); // This might need to be a special call for the bottom solve

    for (int ilevel = nlevels-2; ilevel > -1; ilevel--) {
        prolongate_error<<<grid_size_mg[ilevel+1], block_size>>>(deltaTmg[ilevel+1] , deltaTmg[ilevel], nxl[ilevel+1], nyl[ilevel+1], nxl[ilevel], nyl[ilevel]);
        cudaDeviceSynchronize();
        smoothers[ilevel]->solve(10);
    }
    

}


}