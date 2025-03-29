#include <iostream>
#include "LidDrivenCavity.h"

namespace LidDrivenCavityNS {

    LidDrivenCavity::LidDrivenCavity(int nx_inp, int ny_inp, double nu_inp) {

        nx = nx_inp;
        ny = ny_inp;
        dx = 1.0 / nx;
        dy = 3.0 / ny;
        nu = nu_inp;
        
        grid_size = dim3(nx, ny);
        block_size = dim3(32, 32);

        grid_size_1d = dim3( ceil (nx * ny / 1024.0) );       

        cudaMalloc(&umom, nx * ny * sizeof(double));
        cudaMalloc(&vmom, nx * ny * sizeof(double));
        cudaMalloc(&pres, nx * ny * sizeof(double));
        cudaMalloc(&deltaU, nx * ny * sizeof(double));
        cudaMalloc(&deltaV, nx * ny * sizeof(double));
        cudaMalloc(&deltaP, nx * ny * sizeof(double));
        cudaMalloc(&Jmom, nx * ny * 5 * sizeof(double));
        cudaMalloc(&Jcont, nx * ny * 5 * sizeof(double));
        cudaMalloc(&u_nlr, nx * ny * sizeof(double));
        cudaMalloc(&v_nlr, nx * ny * sizeof(double));
        cudaMalloc(&cont_nlr, nx * ny * sizeof(double));

        std::cout << "Allocated " <<  19 * nx * ny * sizeof(double) / double(1 << 30) << " GB of memory" << std::endl;
        t_unlr = thrust::device_ptr<double>(u_nlr);
        t_vnlr = thrust::device_ptr<double>(v_nlr);
        t_cont_nlr = thrust::device_ptr<double>(cont_nlr);

    }

    LidDrivenCavity::~LidDrivenCavity() {

        cudaFree(umom);
        cudaFree(vmom);
        cudaFree(pres);
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