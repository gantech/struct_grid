#include "LaplaceHeat.h"
#include <iostream>
#include <fstream>
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

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_const(double *T, double val, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny))
        T[idx] = val ;

}

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
        T[(row * nx) + col] = ref_temp(x, y);
    }

}

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) {
        T[idx] += deltaT[idx];
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
        int idx_r = (row * nx) + col;
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
            tip1j = T[idx_r + 1];
            radd += kc * 8.0 * ref_temp(0.0, y) / 3.0 ;
        } else if (col == (nx - 1)) {
            jij -= 2.0;
            jim1j += 0.3333333333333333;
            jip1j -= 1.0;
            tim1j = T[idx_r - 1];
            radd += kc * 8.0 * ref_temp(1.0, y) / 3.0;
        } else {
            tip1j = T[idx_r + 1];
            tim1j = T[idx_r - 1];
        }

        if (row == 0) {
            jij -= 2.0;
            jijp1 += 0.3333333333333333;
            jijm1 -= 1.0;
            tijp1 = T[idx_r + nx];
            radd += kc * 8.0 * ref_temp(x, 0.0) / 3.0;
        } else if (row == (ny - 1)) {
            jij -= 2.0;
            jijm1 += 0.3333333333333333;
            jijp1 -= 1.0;
            tijm1 = T[idx_r - nx];
            radd += kc * 8.0 * ref_temp(x, 1.0) / 3.0;
        } else {
            tijm1 = T[idx_r - nx];
            tijp1 = T[idx_r + nx];
        }

        // Write to residual
        double tmp = kc * ( jijm1 * tijm1 + jijp1 * tijp1 + jim1j * tim1j + jip1j * tip1j + jij * T[idx_r] - (2.0 + 2.0 * y / 9.0) * dx * dy) + radd;

        // if (std::abs(tmp/(dx * dy * kc)) > 20.0) {
        if (std::isnan(tmp)) {
           printf("Row, Col is %d, %d - x,y = %f, %f, Residuals - %f, %f, J - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f, T - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f \n", row, col, x, y, 2.0 - 2.0 * y / 9.0, tmp, jijm1, jijp1, jim1j, jip1j, jij, tijm1, tijp1, tim1j, tip1j, T[idx_r]);
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
        cudaMalloc(&T, nx * ny * sizeof(double));
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
            solver = new JacobiNS::Jacobi(nx, ny, J, T, deltaT, nlr);
        } else if (solver_type == "ADI" ) {
            solver = new ADINS::ADI(nx, ny, J, T, deltaT, nlr);
        } else if (solver_type == "MG" ) {
            solver = new MultiGridNS::MultiGrid(nx, ny, J, T, deltaT, nlr, 3, "Jacobi");
        } else if (solver_type == "CG" ) {
            solver = new CGNS::CG(nx, ny, J, T, deltaT, nlr);
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


    __host__ void LaplaceHeat::initialize_const(double * arr, double val) {
        LaplaceHeatNS::initialize_const<<<grid_size_1d, block_size_1d>>>(arr, val, nx, ny);
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
        return std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx * ny, square(), 0.0, thrust::plus<double>()));
    }

    __host__ double LaplaceHeat::compute_r() {
        LaplaceHeatNS::compute_r<<<grid_size, block_size>>>(T, J, nlr, nx, ny, dx, dy, kc);
        cudaDeviceSynchronize();
        return std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx * ny, square(), 0.0, thrust::plus<double>()));
    }

    __host__ void LaplaceHeat::solve(int nsteps) {
        solver->solve_step(nsteps);
    }

}


int main() {

    double * resid = new double[80];

    std::ofstream resid_file_jacobi("jacobi_resid.txt");
    resid_file_jacobi << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * ljacobi = new LaplaceHeatNS::LaplaceHeat(128, 384, 0.01, "Jacobi");
    ljacobi->initialize_const(ljacobi->T, 300.0);
    for (int i = 0; i < 80; i++) {
        ljacobi->initialize_const(ljacobi->deltaT, 0.0);
        resid[i] = ljacobi->compute_r_j();
        resid_file_jacobi << i << ", " << resid[i] << std::endl;
        ljacobi->solve(100); // Loops of Jacobi
        ljacobi->update();
    }
    resid_file_jacobi.close();
    delete ljacobi;

    std::ofstream resid_file_adi("adi_resid.txt");
    resid_file_adi << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * ladi = new LaplaceHeatNS::LaplaceHeat(128, 384, 0.01, "ADI");
    ladi->initialize_const(ladi->T, 300.0);
    for (int i = 0; i < 80; i++) {
        ladi->initialize_const(ladi->deltaT, 0.0);
        resid[i] = ladi->compute_r_j();
        resid_file_adi << "Iter = " << i << ", " << resid[i] << std::endl;
        ladi->solve(100); // Loops of ADI
        ladi->update();
    }
    resid_file_adi.close();
    delete ladi;

    std::ofstream resid_file_mg("mg_resid.txt");
    resid_file_mg << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * lmg = new LaplaceHeatNS::LaplaceHeat(128, 384, 0.01, "MG");
    lmg->initialize_const(lmg->T, 300.0);
    for (int i = 0; i < 80; i++) {
        lmg->initialize_const(lmg->deltaT, 0.0);
        resid[i] = lmg->compute_r_j();
        resid_file_mg << "Iter = " << i << ", " << resid[i] << std::endl;
        lmg->solve(100); // Loops of MG
        lmg->update();
    }
    resid_file_mg.close();
    delete lmg;

    std::ofstream resid_file_cg("cg_resid.txt");
    resid_file_cg << "Iter, Residual" << std::endl;
    LaplaceHeatNS::LaplaceHeat * lcg = new LaplaceHeatNS::LaplaceHeat(128, 384, 0.01, "CG");
    lcg->initialize_const(lcg->T, 300.0);
    for (int i = 0; i < 80; i++) {
        lcg->initialize_const(lcg->deltaT, 0.0);
        resid[i] = lcg->compute_r_j();
        resid_file_cg << "Iter = " << i << ", " << resid[i] << std::endl;
        lcg->solve(100); // Loops of CG
        lcg->update();
    }
    resid_file_cg.close();
    delete lcg;


    return 0;

}
