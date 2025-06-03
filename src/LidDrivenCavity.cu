#include <iostream>
#include "LidDrivenCavity.h"

namespace LidDrivenCavityNS {


// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_const(double *T, double val, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] = val ;
    
}


// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] += deltaT[idx];

}

// Kernel function for calculation of mass flow rate phi - No tiling or shared memory
__global__ void compute_phi(double *umom, double *vmom, double *phi, int nx, int ny, double dx, double dy) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i < nx) && (j < ny)) {
        phi[j * nx + i] = 0.5 * (vmom[(j-1) * nx + i] + vmom[j * nx + i])* dx;
        phi[j * nx + i + 1] = 0.5 * (umom[j * nx + i - 1] + umom[j * nx + i ]) * dy;
    }
}



    LidDrivenCavity::LidDrivenCavity(int nx_inp, int ny_inp, double nu_inp) {

        nx = nx_inp;
        ny = ny_inp;
        dx = 1.0 / nx;
        dy = 3.0 / ny;
        nu = nu_inp;
        
        grid_size = dim3(nx, ny);
        block_size = dim3(32, 32);

        grid_size_1d = dim3( ceil ( nx * ny / 1024.0) );

        cudaMalloc(&umom, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&vmom, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&pres, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&phi, 2 * nx * ny * sizeof(double));
        cudaMalloc(&a_inv, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&deltaU, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&deltaV, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&deltaP, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&Jmom, (nx + 2) * (ny + 2) * 5 * sizeof(double));
        cudaMalloc(&Jcont, (nx + 2) * (ny + 2) * 5 * sizeof(double));
        cudaMalloc(&u_nlr, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&v_nlr, (nx + 2) * (ny + 2) * sizeof(double));
        cudaMalloc(&cont_nlr, (nx + 2) * (ny + 2) * sizeof(double));

        std::cout << "Allocated " <<  19 * (nx + 2) * (ny + 2) * sizeof(double) / double(1 << 30) << " GB of memory" << std::endl;
        t_unlr = thrust::device_ptr<double>(u_nlr);
        t_vnlr = thrust::device_ptr<double>(v_nlr);
        t_cont_nlr = thrust::device_ptr<double>(cont_nlr);

        initialize_const<<<grid_size_1d, block_size_1d>>>(umom, 0.0, nx, ny);
        initialize_const<<<grid_size_1d, block_size_1d>>>(vmom, 0.0, nx, ny);
        initialize_const<<<grid_size_1d, block_size_1d>>>(pres, 0.0, nx, ny);
        initialize_const<<<grid_size_1d, block_size_1d>>>(deltaU, 0.0, nx, ny);
        initialize_const<<<grid_size_1d, block_size_1d>>>(deltaV, 0.0, nx, ny);
        initialize_const<<<grid_size_1d, block_size_1d>>>(deltaP, 0.0, nx, ny);

    }

    LidDrivenCavity::~LidDrivenCavity() {

        cudaFree(umom);
        cudaFree(vmom);
        cudaFree(pres);
        cudaFree(phi);
        cudaFree(a_inv);
        cudaFree(deltaU);
        cudaFree(deltaV);
        cudaFree(deltaP);
        cudaFree(Jmom);
        cudaFree(Jcont);
        cudaFree(u_nlr);
        cudaFree(v_nlr);
        cudaFree(cont_nlr);

    }
    
}

int main() {

    LidDrivenCavityNS::LidDrivenCavity * lcav = new LidDrivenCavityNS::LidDrivenCavity(128, 384, 0.001);

    delete lcav;
    return 0;
}