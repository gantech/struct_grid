#include "LaplaceHeat.h"
#include <iostream>
#include <fstream>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

namespace LaplaceHeatNS {

// Functor to square the elements
struct square {
    __device__ double operator()(double a) {
        return a * a;
    }
};

// Kernel function for reference temperature solution
__device__ double ref_temp(double x, double y) {
    return 300.0 + x*x + (y*y*y)/ 27.0;
}

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_ref(double *T, int nx, int ny, double dx, double dy) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col < nx) && (row < ny)) {
        double y = (0.5 + row) * dy;
        double x = (0.5 + col) * dx;
        T[(row * (nx+2)) + col + 1] = ref_temp(x, y);
    }

}

// Kernel function for bottom boundary condition - No tiling or shared memory
__global__ void bottom_bc(double *T, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < nx ) {
        double x = (0.5 + idx) * dx;
        T[idx + 1] = ref_temp(x, 0.0);
    }
}

// Kernel function for upper boundary condition - No tiling or shared memory
__global__ void upper_bc(double *T, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < nx ) {
        double x = (0.5 + idx) * dx;
        T[(ny * (nx+2)) + idx + 1] = ref_temp(x, 3.0);
    }
}

// Kernel function for left boundary condition - No tiling or shared memory
__global__ void left_bc(double *T, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ny ) {
        double y = (0.5 + idx) * dy;
        T[( (idx+1) * (nx+2)) + 0] = ref_temp(0.0, y);
    }
}

// Kernel function for right boundary condition - No tiling or shared memory
__global__ void right_bc(double *T, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ny ) {
        double y = (0.5 + idx) * dy;
        T[( (idx+1) * (nx+2)) + nx] = ref_temp(1.0, y);
    }
}

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if ( idx < (nx * ny) ) {
        int row = idx / nx;
        int col = idx % nx;
        T[ (row+1) * (nx+2) + col + 1] += deltaT[idx];
        // printf("idx = %d, deltaT = %d, %d \n", idx, std::isnan(deltaT[idx]), (std::abs(deltaT[idx]) < 10.0));
    }

}

// Kernel function for calculation of Jacobian and Residual - No tiling or shared memory
__global__ void compute_r_j(double *T, double *J, double *R, int nx, int ny, double dx, double dy, double kc) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((col < nx) && (row < ny)) {

        double y = (0.5 + row) * dy;
        double x = (0.5 + col) * dx;
        int idx_t = ( (row+1) * (nx+2)) + col + 1;
        int idx_r = row * nx + col;
        int idx_j = idx_r * 5;

        double jij = -4.0;
        double jip1j = 1.0;
        double jim1j = 1.0;
        double jijp1 = 1.0;
        double jijm1 = 1.0;

        double tip1j = 0.0;
        double tim1j = 0.0;
        double tijp1 = 0.0;
        double tijm1 = 0.0;

        double radd = 0.0;
        if (col == 0) {
            jij -= 2.0;
            jip1j += 0.3333333333333333 ;
            jim1j -= 1.0;
            tip1j = T[idx_t + 1];
            radd += kc * 8.0 * T[idx_t - 1] / 3.0 ;
        } else if (col == (nx - 1)) {
            jij -= 2.0;
            jim1j += 0.3333333333333333;
            jip1j -= 1.0;
            tim1j = T[idx_t - 1];
            radd += kc * 8.0 * T[idx_t + 1] / 3.0;
        } else {
            tip1j = T[idx_t + 1];
            tim1j = T[idx_t - 1];
        }

        if (row == 0) {
            jij -= 2.0;
            jijp1 += 0.3333333333333333;
            jijm1 -= 1.0;
            tijp1 = T[idx_t + (nx + 2)];
            radd += kc * 8.0 * T[idx_t - (nx + 2)] / 3.0;
        } else if (row == (ny - 1)) {
            jij -= 2.0;
            jijm1 += 0.3333333333333333;
            jijp1 -= 1.0;
            tijm1 = T[idx_t - (nx + 2)];
            radd += kc * 8.0 * T[idx_t + (nx + 1)] / 3.0;
        } else {
            tijm1 = T[idx_t - (nx+2)];
            tijp1 = T[idx_t + (nx+2)];
        }

        // Write to residual
        double tmp = kc * ( jijm1 * tijm1 + jijp1 * tijp1 + jim1j * tim1j + jip1j * tip1j + jij * T[idx_t] - (2.0 + 2.0 * y / 9.0) * dx * dy) + radd;

        // if (std::abs(tmp/(dx * dy * kc)) > 20.0) {
        if (std::isnan(tmp)) {
           printf("Row, Col is %d, %d - x,y = %f, %f, Residuals - %f, %f, J - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f, T - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f \n", row, col, x, y, 2.0 - 2.0 * y / 9.0, tmp, jijm1, jijp1, jim1j, jip1j, jij, tijm1, tijp1, tim1j, tip1j, T[idx_t]);
        }

        R[idx_r] = -tmp;

        // Write to the Jacobian
        J[idx_j] = jij * kc; //i,j
        J[idx_j + 1] = jim1j * kc; //i-1,j
        J[idx_j + 2] = jip1j * kc; //i+1,j
        J[idx_j + 3] = jijm1 * kc; //i,j-1
        J[idx_j + 4] = jijp1 * kc; //i,j+1
    }
}

    LaplaceHeat::LaplaceHeat(int nx_inp, int ny_inp, double kc_inp, std::string solver_type) {

        nx = nx_inp;
        ny = ny_inp;
        kc = kc_inp;
        cudaMalloc(&T, (nx+2) * (ny+2) * sizeof(double));
        cudaMalloc(&deltaT, nx * ny * sizeof(double));
        cudaMalloc(&J, nx * ny * 5 * sizeof(double));
        cudaMalloc(&nlr, nx * ny * sizeof(double));

        t_nlr = thrust::device_ptr<double>(nlr);

        dx = 1.0 / double(nx);
        dy = 3.0 / double(ny);

        grid_size = dim3(std::ceil(ny/32), std::ceil(nx/32));
        block_size = dim3(32, 32);

        grid_size_1d = dim3( ceil (nx * ny / 1024.0) );

        if (solver_type == "Jacobi") {
            solver = new JacobiNS::Jacobi(nx, ny, J, deltaT, nlr);
        } else if (solver_type == "ADI" ) {
            solver = new ADINS::ADI(nx, ny, J, deltaT, nlr);
        } else if (solver_type == "MG" ) {
            solver = new MultiGridNS::MultiGrid(nx, ny, J, deltaT, nlr, 4, "Jacobi");
        } else if (solver_type == "CG" ) {
            solver = new CGNS::CG(nx, ny, J, deltaT, nlr);
        } else {
            std::cout << "Invalid solver type. Availabl solvers are Jacobi and ADI. " << std::endl;
            exit(1);
        }

    }


    LaplaceHeat::~LaplaceHeat() {

        // Free memory
        cudaFree(T);
        cudaFree(deltaT);
        cudaFree(J);
        cudaFree(nlr);
    }

    // Replace your custom initialize_const function
    __host__ void LaplaceHeat::initialize_const(double * arr, double val, int ntot) {
        thrust::device_ptr<double> d_ptr(arr);
        thrust::fill(d_ptr, d_ptr + ntot, val);
    }

    __host__ void LaplaceHeat::initialize_ref() {
        LaplaceHeatNS::initialize_ref<<<grid_size, block_size>>>(T, nx, ny, dx, dy);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::apply_bc() {
        cudaStream_t streams[4];
        for (int i = 0; i < 4; ++i) cudaStreamCreate(&streams[i]);

        bottom_bc<<<dim3(ceil(nx/1024.0)), dim3(1024), 0, streams[0]>>>(T, nx, ny, dx, dy);
        upper_bc<<<dim3(ceil(nx/1024.0)), dim3(1024), 0, streams[1]>>>(T, nx, ny, dx, dy);
        left_bc<<<dim3(ceil(ny/1024.0)), dim3(1024), 0, streams[2]>>>(T, nx, ny, dx, dy);
        right_bc<<<dim3(ceil(ny/1024.0)), dim3(1024), 0, streams[3]>>>(T, nx, ny, dx, dy);

        for (int i = 0; i < 4; ++i) cudaStreamSynchronize(streams[i]);
        for (int i = 0; i < 4; ++i) cudaStreamDestroy(streams[i]);
        cudaDeviceSynchronize();
    }

    __host__ void LaplaceHeat::update() {
        LaplaceHeatNS::update<<<grid_size_1d, block_size_1d>>>(T, deltaT, nx, ny);
        cudaDeviceSynchronize();
    }

    __host__ double LaplaceHeat::compute_r_j() {
        LaplaceHeatNS::compute_r_j<<<grid_size, block_size>>>(T, J, nlr, nx, ny, dx, dy, kc);
        cudaDeviceSynchronize();
        return std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx * ny, square(), 0.0, thrust::plus<double>()));
    }

    __host__ void LaplaceHeat::solve(int nsteps) {
        solver->solve_step(nsteps);
    }

}


int main() {

    double * resid = new double[80];

    int nx = 128;
    int ny = 384;

    std::ofstream resid_file_jacobi("jacobi_resid.txt");
    resid_file_jacobi << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * ljacobi = new LaplaceHeatNS::LaplaceHeat(nx, ny, 0.01, "Jacobi");
    ljacobi->initialize_const(ljacobi->T, 300.0, (nx+2)*(ny+2));
    ljacobi->apply_bc();
    for (int i = 0; i < 80; i++) {
        ljacobi->initialize_const(ljacobi->deltaT, 0.0, nx * ny);
        resid[i] = ljacobi->compute_r_j();
        resid_file_jacobi << i << ", " << resid[i] << std::endl;
        ljacobi->solve(100); // Loops of Jacobi
        ljacobi->update();
    }
    resid_file_jacobi.close();
    delete ljacobi;

    std::ofstream resid_file_adi("adi_resid.txt");
    resid_file_adi << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * ladi = new LaplaceHeatNS::LaplaceHeat(nx, ny, 0.01, "ADI");
    ladi->initialize_const(ladi->T, 300.0, (nx+2)*(ny+2));
    ladi->apply_bc();
    for (int i = 0; i < 80; i++) {
        ladi->initialize_const(ladi->deltaT, 0.0, nx * ny);
        resid[i] = ladi->compute_r_j();
        resid_file_adi << "Iter = " << i << ", " << resid[i] << std::endl;
        ladi->solve(100); // Loops of ADI
        ladi->update();
    }
    resid_file_adi.close();
    delete ladi;

    std::ofstream resid_file_mg("mg_resid.txt");
    resid_file_mg << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * lmg = new LaplaceHeatNS::LaplaceHeat(nx, ny, 0.01, "MG");
    lmg->initialize_const(lmg->T, 300.0, (nx+2)*(ny+2));
    lmg->apply_bc();
    for (int i = 0; i < 80; i++) {
        lmg->initialize_const(lmg->deltaT, 0.0, nx * ny);
        resid[i] = lmg->compute_r_j();
        resid_file_mg << "Iter = " << i << ", " << resid[i] << std::endl;
        lmg->solve(100); // Loops of MG
        lmg->update();
    }
    resid_file_mg.close();
    delete lmg;

    std::ofstream resid_file_cg("cg_resid.txt");
    resid_file_cg << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * lcg = new LaplaceHeatNS::LaplaceHeat(nx, ny, 0.01, "CG");
    lcg->initialize_const(lcg->T, 300.0, (nx+2)*(ny+2));
    lcg->apply_bc();
    for (int i = 0; i < 80; i++) {
        lcg->initialize_const(lcg->deltaT, 0.0, nx * ny);
        resid[i] = lcg->compute_r_j();
        resid_file_cg << "Iter = " << i << ", " << resid[i] << std::endl;
        lcg->solve(100); // Loops of CG
        lcg->update();
    }
    resid_file_cg.close();
    delete lcg;

    delete [] resid;

    return 0;

}
