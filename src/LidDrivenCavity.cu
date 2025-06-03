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


// Kernel function for calculation of Jacobian and Residual - No tiling or shared memory
__global__ void compute_mom_r_j(double *umom, double * vmom, double *pres, double *Jmom, double * u_nlr, double * v_nlr, int nx, int ny, double dx, double dy, double nu) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    double deltat_inv = 1.0/0.001;

    if ((i < nx) && (j < ny)) {
        
        double y = (0.5 + j) * dy;
        double x = (0.5 + i) * dx;
        int idx_r = (j * nx) + i;
        int idx_j = idx_r * 5;

        double jij = 4.0;
        double jip1j = -1.0;
        double jim1j = -1.0;
        double jijp1 = -1.0;
        double jijm1 = -1.0;

        double tip1j = 0.0;
        double tim1j = 0.0;
        double tijp1 = 0.0;
        double tijm1 = 0.0;

        double ruadd = 0.0;
        double rvadd = 0.0;

        if (i == 0) {
            jij += 2.0;
            jip1j -= 0.3333333333333333 ;
            jim1j += 1.0;
            tip1j = T[idx_r + 1];
            double t_bc_left = 300.0 + (y*y*y/27.0);
            ruadd += kc * 8.0 * t_bc_left / 3.0 ;
        } else if (i == (nx - 1)) {
            jij += 2.0;
            jim1j -= 0.3333333333333333;
            jip1j += 1.0;
            tim1j = T[idx_r - 1];
            double t_bc_right = 300.0 + 1.0 + (y*y*y/27.0);
            radd += kc * 8.0 * t_bc_right / 3.0;
        } else {
            tip1j = T[idx_r + 1];
            tim1j = T[idx_r - 1];
            radd += 0.5 * ()
        }

        if (j == 0) {
            jij += 2.0;
            jijp1 -= 0.3333333333333333;
            jijm1 += 1.0;
            tijp1 = T[idx_r + nx];
            double t_bc_bot = 300.0 + (x*x);
            radd += kc * 8.0 * t_bc_bot / 3.0;
        } else if (j == (ny - 1)) {
            jij += 2.0;
            jijm1 -= 0.3333333333333333;
            jijp1 += 1.0;
            tijm1 = T[idx_r - nx];
            double t_bc_top = 300.0 + 1.0 + (x*x);
            radd += kc * 8.0 * t_bc_top / 3.0;
        } else {
            tijm1 = T[idx_r - nx];
            tijp1 = T[idx_r + nx];
        }

        // Write to residual
        double tmp = nu * ( jijm1 * tijm1 + jijp1 * tijp1 + jim1j * tim1j + jip1j * tip1j + jij * T[idx_r] - (2.0 + 2.0 * y / 9.0) * dx * dy) + radd;

        // if (std::abs(tmp/(dx * dy * kc)) > 20.0) {
        //     printf("i, j is %d, %d - x,y = %f, %f, Residuals - %f, %f, J - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f, T - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f \n", i, j, x, y, 2.0 - 2.0 * y / 9.0, tmp / (dx * dy * kc), jijm1, jijp1, jim1j, jip1j, jij, tijm1, tijp1, tim1j, tip1j, T[idx_r]);
        // }

        R[idx_r] = -tmp;

        // Write to the Jacobian
        J[idx_j] = jij * kc; //i,j
        J[idx_j + 1] = jim1j * kc; //i-1,j
        J[idx_j + 2] = jip1j * kc; //i+1,j
        J[idx_j + 3] = jijm1 * kc; //i,j-1
        J[idx_j + 4] = jijp1 * kc; //i,j+1
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