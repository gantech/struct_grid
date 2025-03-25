#include "LaplaceHeat.h"

#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

namespace LaplaceHeatNS {

    // Functor to square the elements
    struct square {
        __device__ double operator()(double a) {
            return a * a;
        }
    };

    LaplaceHeat::LaplaceHeat(int nx_inp, int ny_inp, double kc_inp) {

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

        solver = new JacobiNS::Jacobi(nx, ny, J, T, deltaT, nlr);
        
    }

    LaplaceHeat::~LaplaceHeat() {

        // Free memory
        cudaFree(T);
        cudaFree(deltaT);
        cudaFree(J);
        cudaFree(nlr);
    }


    __host__ void LaplaceHeat::initialize_const(double val) {
        LaplaceHeatNS::initialize_const<<<grid_size_1d, block_size_1d>>>(T, val, nx, ny);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::initialize_ref() {
        LaplaceHeatNS::initialize_ref<<<grid_size_1d, block_size_1d>>>(T, nx, ny, dx, dy);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::update() {
        LaplaceHeatNS::update<<<grid_size_1d, block_size_1d>>>(T, deltaT, nx, ny);
        cudaDeviceSynchronize();
    }

    __host__ double LaplaceHeat::compute_r_j() {
        LaplaceHeatNS::compute_r_j<<<grid_size, block_size>>>(T, J, nlr, nx, ny, dx, dy, kc);        
        cudaDeviceSynchronize();
        thrust::device_ptr<double> t_nlr(nlr);
        return std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx * ny, square(), 0.0, thrust::plus<double>()));        
    }

    __host__ double LaplaceHeat::compute_r() {
        LaplaceHeatNS::compute_r<<<grid_size, block_size>>>(T, J, nlr, nx, ny, dx, dy, kc);
        cudaDeviceSynchronize();
        thrust::device_ptr<double> t_nlr(nlr);
        return std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx * ny, square(), 0.0, thrust::plus<double>()));              
    }

    __host__ void LaplaceHeat::compute_matvec(double * v, double * result) {
        LaplaceHeatNS::compute_matvec<<<grid_size, block_size>>>(v, J, result, nx, ny);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::solve(int nsteps) {
        for (int j = 0; j < nsteps; j++) {
            solver->solve_step();
    }

}

}


int main() {

    LaplaceHeatNS::LaplaceHeat l(128, 384, 0.01);
    l.initialize_const(300.0);
    double * resid = new double[80];
    for (int i = 0; i < 80; i++) {
        resid[i] = l.compute_r_j();
        l.solve(1000); // Loops of Jacobi
        l.update();
    }
    return 0;
    
}

