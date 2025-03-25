#include "LaplaceHeat.h"

namespace LaplaceHeat {

    void LaplaceHeat(int nx_inp, int ny_inp, double kc_inp) {

        nx = nx_inp;
        ny = ny_inp;
        kc = kc_inp;
        cudaMalloc(&T, nx * ny * sizeof(double));
        cudaMalloc(&deltaT, nx * ny * sizeof(double));
        cudaMalloc(&J, nx * ny * 5 * sizeof(double));
        cudaMalloc(&nlr, nx * ny * sizeof(double));

        double dx = 1.0 / double(nx);
        double dy = 3.0 / double(ny);

        grid_size = dim3(nx, ny);
        block_size = dim3(32, 32);

        grid_size_1d = dim3( ceil (nx * ny / 1024.0) );        

    }

    ~LaplaceHeat() {

        // Free memory
        cudaFree(T);
        cudaFree(deltaT);
        cudaFree(J);
        cudaFree(nlr);
    }


    __host__ void LaplaceHeat::initialize() {
        LaplaceHeat::initialize<<<grid_size_1d, block_size_1d>>>(T, nx, ny, dx, dy);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::initialize_ref() {
        LaplaceHeat::initialize_ref<<<grid_size_1d, block_size_1d>>>(T, nx, ny, dx, dy);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::update() {
        LaplaceHeat::update<<<grid_size_1d, block_size_1d>>>(T, deltaT, nx, ny);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::compute_r_j() {
        LaplaceHeat::compute_r_j<<<grid_size, block_size>>>(T, J, nlr, nx, ny, dx, dy);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::compute_r() {
        LaplaceHeat::compute_r<<<grid_size, block_size>>>(T, J, nlr, nx, ny, dx, dy);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::compute_matvec(double * v, double * result) {
        LaplaceHeat::compute_matvec<<<grid_size, block_size>>>(v, J, result, nx, ny);
        cudaDeviceSynchronize();
    }

}


// int main() {

//     l = LaplaceHeat(128, 384);
//     l.compute_r_j();
//     solver = Jacobi()
//     for (int i = 0; i < 80; i++) {
//         for (int j = 0; j < 10000; j++) 
//             l.jacobi();
//         l.update();
//         l.compute_r_j();
//     }
//     return 0;
    
// }