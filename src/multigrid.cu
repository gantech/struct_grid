#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

#define TILE_SIZE 32
#define TILE_SIZE_ADI 2

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize(double *T, int nx, int ny, double dx, double dy);

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_ref(double *T, int nx, int ny, double dx, double dy);

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny, double dx, double dy);

// Kernel function for calculation of Jacobian and Residual - No tiling or shared memory
__global__ void compute_r_j(double *T, double *J, double *R, int nx, int ny, double dx, double dy, double kc);

// Kernel function for calculation of Residual - No tiling or shared memory
__global__ void compute_r(double *T, double * J, double *R, int nx, int ny, double dx, double dy, double kc) ;

// Kernel function for Thomas solves in the X direction - part of ADI 
__global__ void adi_x(double *T, double *J, double *R, int nx, int ny);

// Kernel function for Thomas solves in the Y direction - part of ADI 
__global__ void adi_y(double *T, double *J, double *R, int nx, int ny);

// Functor to square the elements
struct square {
    __device__ double operator()(double a) {
        return a * a;
    }
};

// Kernel function to initialize a given field to zero
__global__ void initialize_zero(double * T, int nx, int ny) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = (j * nx) + i;

    if ( (i < nx) && (j < ny))
        T[idx] = 0.0;
}

// Kernel to compute linear residual of the linear system of equations J * deltaT = rhs. 
// Write linear residual to new array R. If you want the rhs overwritten, pass the same pointers for rhs and R
__global__ void compute_lin_resid(double * deltaT, double * J, double * rhs, double * R, int nx, int ny) {

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

        double tip1j = 0.0;
        double tim1j = 0.0;
        double tijp1 = 0.0;
        double tijm1 = 0.0;

        if ( i == 0) {
            tip1j = deltaT[idx_r + 1];
        } else if ( i == (nx - 1)) {
            tim1j = deltaT[idx_r - 1];
        } else {
            tip1j = deltaT[idx_r + 1];
            tim1j = deltaT[idx_r - 1];
        }

        if ( j == 0) {
            tijp1 = deltaT[idx_r + nx];
        } else if ( j == (ny - 1)) {
            tijm1 = deltaT[idx_r - nx];
        } else {
            tijm1 = deltaT[idx_r - nx];
            tijp1 = deltaT[idx_r + nx];
        }

        // Write to residual

        if (std::abs(jij) < 1e-5) {
            printf("nx = %d, ny = %d, i = %d, j = %d, R = %e, jim1j = %e, jip1j = %e, jijm1 = %e, jijp1 = %e, jij = %e \n", nx, ny, i, j, R[idx_r], jim1j, jip1j, jijm1, jijp1, jij);
        }
        R[idx_r] = rhs[idx_r] - jim1j * tim1j - jip1j * tip1j - jijm1 * tijm1 - jijp1 * tijp1 - jij * deltaT[idx_r];
    }
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
    int idx_rf3 = (2 * (jc + 1) * nxf) + (2 * ic);
    int idx_rf4 = (2 * (jc + 1) * nxf) + (2 * ic + 1);

    if ( (ic < nxc) && (jc < nyc) )
        rc[idx_rc] = (rf[idx_rf1] + rf[idx_rf2] + rf[idx_rf3] + rf[idx_rf4])/std::sqrt(2.0);

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
    int idx_rf3 = (2 * (jc + 1) * nxf) + (2 * ic);
    int idx_rf4 = (2 * (jc + 1) * nxf) + (2 * ic + 1);

    if ( (ic < nxc) && (jc < nyc) ) {

        // printf("Prolongating (i,j) = %d, %d, deltaT = %e \n", ic, jc, deltaTc[idx_rc]);
        deltaTf[idx_rf1] += deltaTc[idx_rc]/std::sqrt(2.0);
        deltaTf[idx_rf2] += deltaTc[idx_rc]/std::sqrt(2.0);
        deltaTf[idx_rf3] += deltaTc[idx_rc]/std::sqrt(2.0);
        deltaTf[idx_rf4] += deltaTc[idx_rc]/std::sqrt(2.0);
        
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
    int idx_jf3 = ((2 * (j + 1) * nxf) + (2 * i)) * 5;
    int idx_jf4 = ((2 * (j + 1) * nxf) + (2 * i + 1)) * 5;

    if ( (i < nxc) && (j < nyc) ) {

        // Diagonals and Interlinks of the 4 cells
        double j0 = 0.5*(jf[idx_jf1] + jf[idx_jf2] + jf[idx_jf3] + jf[idx_jf4]    + jf[idx_jf1+2] + jf[idx_jf1+4]    + jf[idx_jf2+1] + jf[idx_jf2+4]    + jf[idx_jf3+3] + jf[idx_jf3+2]    + jf[idx_jf4+1] + jf[idx_jf4+3]);
        double j1 = 0.5*(jf[idx_jf1+1] + jf[idx_jf3+1]);
        double j2 = 0.5*(jf[idx_jf2+2] + jf[idx_jf4+2]);
        double j3 = 0.5*(jf[idx_jf1+3] + jf[idx_jf2+3]);
        double j4 = 0.5*(jf[idx_jf3+4] + jf[idx_jf4+4]);

        // printf("i = %d, j = %d, j0 = %e, jc0 = %e, j1 = %e, jc1 = %e, j2 = %e, jc2 = %e, j3 = %e, jc3 = %e, j4 = %e, jc4 = %e \n", i, j, j0, jc[idx_jc], j1, jc[idx_jc+1], j2, jc[idx_jc+2], j3, jc[idx_jc+3], j4, jc[idx_jc+4]);
        
        jc[idx_jc]= j0;
        jc[idx_jc+1]= j1;
        jc[idx_jc+2]= j2;
        jc[idx_jc+3]= j3;
        jc[idx_jc+4]= j4;
        

        // if ( std::abs(j0 - 2.0 * jc[idx_jc]) > 1.0e-5) {
        //     printf("i = %d, j = %d, j0 = %e, jc0 = %e, j0f components = %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e, %e \n", i, j, j0, jc[idx_jc], jf[idx_jf1] , jf[idx_jf2] , jf[idx_jf3] , jf[idx_jf4], jf[idx_jf1+2] , jf[idx_jf1+4] , jf[idx_jf2+1] , jf[idx_jf2+4] , jf[idx_jf3+3] , jf[idx_jf3+2] , jf[idx_jf4+1] , jf[idx_jf4+3] );
        // }

        // if ( std::abs(j1 - 2.0 * jc[idx_jc+1]) > 1.0e-5) {
        //     printf("i = %d, j = %d, j1 = %e, jc1 = %e, j1f components = %e, %e \n", i, j, j1, jc[idx_jc+1], jf[idx_jf1+1] , jf[idx_jf3+1]);
        // }

        // if ( std::abs(j2 - 2.0 * jc[idx_jc+2]) > 1.0e-5) {
        //     printf("i = %d, j = %d, j2 = %e, jc2 = %e, j2f components = %e, %e \n", i, j, j2, jc[idx_jc+2], jf[idx_jf2+2] , jf[idx_jf4+2]);
        // }

        // if ( std::abs(j3 - 2.0 * jc[idx_jc+3]) > 1.0e-5) {
        //     printf("i = %d, j = %d, j3 = %e, jc3 = %e, j3f components = %e, %e \n", i, j, j3, jc[idx_jc+3], jf[idx_jf1+3] , jf[idx_jf2+3]);
        // }

        // if ( std::abs(j4 - 2.0 * jc[idx_jc+4]) > 1.0e-5) {
        //     printf("i = %d, j = %d, j4 = %e, jc4 = %e, j4f components = %e, %e \n", i, j, j4, jc[idx_jc+4], jf[idx_jf3+4] , jf[idx_jf4+4]);
        // }

        // printf("nxc = %d, nyc = %d, i = %d, j = %d, j = %e, %e, %e, %e, %e \n", nxc, nyc, i, j, jc[idx_jc], jc[idx_jc+1], jc[idx_jc+2], jc[idx_jc+3], jc[idx_jc+4]);

    }

}

// Kernel function for Gauss-Seidel smoother - No tiling or shared memory
__global__ void gauss_seidel(double *deltaT, double *J, double *R, int nx, int ny) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (( i < nx) && (j < ny)) {
        
        int idx_r = (j * nx) + i;
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

        if (i == 0) {
            tip1j = deltaT[idx_r + 1];
        } else if (i == (nx - 1)) {
            tim1j = deltaT[idx_r - 1];
        } else {
            tip1j = deltaT[idx_r + 1];
            tim1j = deltaT[idx_r - 1];
        }

        if (j == 0) {
            tijp1 = deltaT[idx_r + nx];
        } else if (j == (ny - 1)) {
            tijm1 = deltaT[idx_r - nx];
        } else {
            tijm1 = deltaT[idx_r - nx];
            tijp1 = deltaT[idx_r + nx];
        }

        deltaT[idx_r] = 0.5*(R[idx_r] - jim1j * tim1j - jip1j * tip1j - jijm1 * tijm1 - jijp1 * tijp1) / jij;

        if (std::abs(jij) < 1e-8)
            printf("nx = %d, ny = %d, i = %d, j = %d, deltaT = %e, R = %e, jim1j = %e, jip1j = %e, jijm1 = %e, jijp1 = %e, jij = %e \n", nx, ny, i, j, deltaT[idx_r], R[idx_r], jim1j, jip1j, jijm1, jijp1, jij);


        // if (std::isinf(deltaT[idx_r]) || std::isnan(deltaT[idx_r]))
    }
}


int main() {

    // Finest level problem size
    int nx_f = 128;
    int ny_f = 384;

    // Need resolution only on the finest grid to assemble the equations
    double dx = 1.0 / double(nx_f);
    double dy = 3.0 / double(ny_f);

    double kc = 0.01;

    // Number of levels in multigrid - each refined in all directions by a factor of 2
    int nlevels = 4; 
    std::vector<int> nx(nlevels);
    std::vector<int> ny(nlevels);
    for (int i = 0; i < nlevels; i++) {
        nx[i] = nx_f / (1 << i);
        ny[i] = ny_f / (1 << i);
        std::cout << "ilevel = " << i << ", nx = " << nx[i] << ", ny = " << ny[i] << std::endl;
    }

    // Fields for temperature and non-linear residual are required only at the finest level
    double * T;
    cudaMalloc(&T, nx_f * ny_f * sizeof(double));
    double * nlr;
    cudaMalloc(&nlr, nx_f * ny_f * sizeof(double));

    std::vector<double*> deltaT(nlevels), J(nlevels), R(nlevels), Rlin(nlevels);
    for (int i = 0; i < nlevels; i++) {
        cudaMalloc(&deltaT[i], nx[i] * ny[i] * sizeof(double));
        cudaMalloc(&J[i], nx[i] * ny[i] * 5 * sizeof(double));
        cudaMalloc(&R[i], nx[i] * ny[i] * sizeof(double));
        cudaMalloc(&Rlin[i], nx[i] * ny[i] * sizeof(double));
    }

    // Grid and block size
    std::vector<dim3> grid_size;
    for (int ilevel = 0; ilevel < nlevels; ilevel++) 
        grid_size.push_back(dim3(ceil(nx[ilevel] / (double)TILE_SIZE), ceil(ny[ilevel] / (double)TILE_SIZE), 1));
    // Keep block size same for all grids for now
    dim3 block_size(TILE_SIZE, TILE_SIZE, 1);

    initialize<<<grid_size[0], block_size>>>(T, nx[0], ny[0], dx, dy);
    cudaDeviceSynchronize();

    compute_r_j<<<grid_size[0], block_size>>>(T, J[0], nlr, nx[0], ny[0], dx, dy, kc);
    cudaDeviceSynchronize();
    double glob_resid = 0.0;
    thrust::device_ptr<double> t_nlr(nlr);
    glob_resid = std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx[0] * ny[0], square(), 0.0, thrust::plus<double>()));
    std::cout << "Starting residual with const 300.0 field = " << glob_resid << std::endl;

    // initialize_ref<<<grid_size[0], block_size>>>(T, nx[0], ny[0], dx, dy);
    // cudaDeviceSynchronize();

    // compute_r_j<<<grid_size[0], block_size>>>(T, J[0], nlr, nx[0], ny[0], dx, dy, kc);
    // cudaDeviceSynchronize();
    // glob_resid = 0.0;
    // glob_resid = std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx[0] * ny[0], square(), 0.0, thrust::plus<double>()));
    // std::cout << "Starting residual with correct solution field T = 300.0 + x^2 + (y/3)^3 = " << glob_resid << std::endl;    

    // // Compute Jacobian directly on second level. Won't match the restriction for the matrix. 
    // double *T2;
    // cudaMalloc(&T2, nx[1]*ny[1] * sizeof(double));
    // compute_r_j<<<grid_size[1], block_size>>>(T2, J[1], R[1], nx[1], ny[1], 2.0 * dx, 2.0 * dy, kc);
    // cudaDeviceSynchronize();
    
    // Compute the Jacobian matrix at the coarser levels 
    for (int ilevel = 1; ilevel < nlevels; ilevel++) {
        std::cout << "Restricting J from ilevel = " << ilevel - 1 << " to ilevel = " << ilevel << std::endl;
        restrict_j<<<grid_size[ilevel], block_size>>>(J[ilevel], J[ilevel-1], nx[ilevel], ny[ilevel], nx[ilevel-1], ny[ilevel-1]);
        cudaDeviceSynchronize();
    }


    // double * h_Jc = new double(nx[nlevels-1] * ny[nlevels-1] * 5);
    // cudaMemcpy(h_Jc, J[nlevels-1], nx[nlevels-1] * ny[nlevels-1] * 5 * sizeof(double), cudaMemcpyDeviceToHost);

    // std::ofstream myfile;
    // myfile.open("Jcoarse.txt");
    // for (int j = 0; j < ny[nlevels-1]; j++) {
    //     for (int i = 0; i < nx[nlevels-1]; i++) {
    //         int idx_j = (j * nx[nlevels-1]) + i;
    //         myfile << "i,j = " << i <<  " " << j << " " << h_Jc[idx_j] << " " << h_Jc[idx_j + 1] << " " << h_Jc[idx_j + 2] << " " << h_Jc[idx_j + 3] << " " << h_Jc[idx_j + 4] << std::endl;
    //     }
    // }
    // myfile.close();

    // Write 1 V-cycle of multigrid

    for (int iloop = 0; iloop < 80; iloop++) {
    std::cout << "Loop = " << iloop << std::endl;
    
    // Downstroke of V-cycle

    // Initialize deltaT at all levels to zero
    for (int ilevel = 0; ilevel < nlevels; ilevel++) {
        initialize_zero<<<grid_size[ilevel], block_size>>>(deltaT[ilevel], nx[ilevel], ny[ilevel]);
        cudaDeviceSynchronize();
    }
        
    
    // Do some smoothing on the finest level first
    for (int ismooth = 0; ismooth < 10; ismooth++) {
        gauss_seidel<<<grid_size[0], block_size>>>(deltaT[0], J[0], nlr, nx[0], ny[0]);
        cudaDeviceSynchronize();
    }

    // // Compute the residual of the linear system of equations at this level
    compute_lin_resid<<<grid_size[0], block_size>>>(deltaT[0], J[0], nlr, Rlin[0], nx[0], ny[0]);
    cudaDeviceSynchronize();

    thrust::device_ptr<double> t_r0(Rlin[0]);
    glob_resid = std::sqrt(thrust::transform_reduce(t_r0, t_r0 + nx[0] * ny[0], square(), 0.0, thrust::plus<double>()));
    std::cout << "Finest level linear residual after smoothing = " << glob_resid << std::endl;

    for (int ilevel = 1; ilevel < nlevels-1; ilevel++) {
        // Restrict the residual of the linear system
        restrict_resid<<<grid_size[ilevel], block_size>>>(R[ilevel], Rlin[ilevel-1], nx[ilevel], ny[ilevel], nx[ilevel-1], ny[ilevel-1]);
        cudaDeviceSynchronize();
        thrust::device_ptr<double> t_r(R[ilevel]);
        double tmp_resid = std::sqrt(thrust::transform_reduce(t_r, t_r + nx[ilevel] * ny[ilevel], square(), 0.0, thrust::plus<double>()));
        std::cout << "At level ilev = " << ilevel << ", restricted residual = " << tmp_resid << std::endl;
        
        // Perform some smoothing at this level to get the error
        for (int ismooth = 0; ismooth < 10; ismooth++) {
            gauss_seidel<<<grid_size[ilevel], block_size>>>(deltaT[ilevel], J[ilevel], R[ilevel], nx[ilevel], ny[ilevel]);
            cudaDeviceSynchronize();
        }

        // Compute the residual of the linear system of equations at this level.
        compute_lin_resid<<<grid_size[ilevel], block_size>>>(deltaT[ilevel], J[ilevel], R[ilevel], Rlin[ilevel], nx[ilevel], ny[ilevel]);
        cudaDeviceSynchronize();
        thrust::device_ptr<double> t_linr(Rlin[ilevel]);
        tmp_resid = std::sqrt(thrust::transform_reduce(t_linr, t_linr + nx[ilevel] * ny[ilevel], square(), 0.0, thrust::plus<double>()));
        std::cout << "At level ilev = " << ilevel << ", residual after smoothing = " << tmp_resid << std::endl;

    }
     // Restrict the residual of the linear system to coarsest level
    restrict_resid<<<grid_size[nlevels-1], block_size>>>(R[nlevels-1], Rlin[nlevels-2], nx[nlevels-1], ny[nlevels-1], nx[nlevels-2], ny[nlevels-2]);
    cudaDeviceSynchronize();

    // Do bottom level solve with ADI 
    dim3 grid_size_adix(ceil(ny[nlevels-1] / (double)TILE_SIZE_ADI), 1, 1);
    dim3 block_size_adi(TILE_SIZE_ADI, 1,1);
    dim3 grid_size_adiy(ceil(nx[nlevels-1] / (double)TILE_SIZE_ADI), 1, 1);

    for (int ismooth = 0; ismooth < 10; ismooth++) {
        gauss_seidel<<<grid_size[nlevels-1], block_size>>>(deltaT[nlevels-1], J[nlevels-1], R[nlevels-1], nx[nlevels-1], ny[nlevels-1]);
        cudaDeviceSynchronize();
        // adi_x<<<grid_size_adix, block_size_adi>>>(deltaT[nlevels-1], J[nlevels-1], R[nlevels-1], nx[nlevels-1], ny[nlevels-1]);
        // cudaDeviceSynchronize();
        // adi_y<<<grid_size_adiy, block_size_adi>>>(deltaT[nlevels-1], J[nlevels-1], R[nlevels-1], nx[nlevels-1], ny[nlevels-1]);
        // cudaDeviceSynchronize();
        compute_lin_resid<<<grid_size[nlevels-1], block_size>>>(deltaT[nlevels-1], J[nlevels-1], R[nlevels-1], Rlin[nlevels-1], nx[nlevels-1], ny[nlevels-1]);        
        cudaDeviceSynchronize();
        // thrust::device_ptr<double> t_linr(Rlin[nlevels-1]);
        // double tmp_resid = std::sqrt(thrust::transform_reduce(t_linr, t_linr + nx[nlevels-1] * ny[nlevels-1], square(), 0.0, thrust::plus<double>()));
        // std::cout << "At coarsest level ismooth = " << ismooth << ", residual after smoothing = " << tmp_resid << std::endl;
    }

    // Upstroke of V-cycle - This should end on the finest level (ilevel = 0)
    for (int ilevel = nlevels - 2; ilevel > 0; ilevel--) {
        // Prolongate the error
        std::cout << "Prolongating error at ilevel = " << ilevel << ", ilevel + 1 = " << ilevel + 1 << std::endl;
        std::cout << "Grid Size = " << grid_size[ilevel+1].x << ", " << grid_size[ilevel+1].y << std::endl;
        prolongate_error<<<grid_size[ilevel+1], block_size>>>(deltaT[ilevel+1], deltaT[ilevel], nx[ilevel+1], ny[ilevel+1], nx[ilevel], ny[ilevel]);
        cudaDeviceSynchronize();

        // Do some more smoothing at this level to reduce the error
        for (int ismooth = 0; ismooth < 10; ismooth++) {
            gauss_seidel<<<grid_size[ilevel], block_size>>>(deltaT[ilevel], J[ilevel], R[ilevel], nx[ilevel], ny[ilevel]);
            cudaDeviceSynchronize();
        }

        compute_lin_resid<<<grid_size[ilevel], block_size>>>(deltaT[ilevel], J[ilevel], R[ilevel], Rlin[ilevel], nx[ilevel], ny[ilevel]);
        cudaDeviceSynchronize();
        thrust::device_ptr<double> t_linr(Rlin[ilevel]);
        double tmp_resid = std::sqrt(thrust::transform_reduce(t_linr, t_linr + nx[ilevel] * ny[ilevel], square(), 0.0, thrust::plus<double>()));
        std::cout << "At level ilev = " << ilevel << ", residual after smoothing in upstroke = " << tmp_resid << std::endl;

    }

    prolongate_error<<<grid_size[1], block_size>>>(deltaT[1], deltaT[0], nx[1], ny[1], nx[0], ny[0]);
    cudaDeviceSynchronize();
    for (int ismooth=0; ismooth < 10; ismooth++) {
        gauss_seidel<<<grid_size[0], block_size>>>(deltaT[0], J[0], nlr, nx[0], ny[0]);
        cudaDeviceSynchronize();
    }

    update<<<grid_size[0], block_size>>>(T, deltaT[0], nx[0], ny[0], dx, dy);
    cudaDeviceSynchronize();

    compute_r_j<<<grid_size[0], block_size>>>(T, J[0], nlr, nx[0], ny[0], dx, dy, kc);
    cudaDeviceSynchronize();
    glob_resid = std::sqrt(thrust::transform_reduce(t_nlr, t_nlr + nx[0] * ny[0], square(), 0.0, thrust::plus<double>()));
    std::cout << "Ending residual = " << glob_resid << std::endl;    

    }


    double *h_R = new double[nx_f * ny_f];
    cudaMemcpy(h_R, nlr, nx_f * ny_f * sizeof(double), cudaMemcpyDeviceToHost);

    // Write h_R to a file 
    std::ofstream outfile("residual.txt");
    for (int j = 0; j < ny_f; ++j) {
        for (int i = 0; i < nx_f; ++i) {
            outfile << h_R[j * nx_f + i] << " ";
        }
        outfile << std::endl;
    }
    outfile.close();
    delete[] h_R;    

    double *h_T = new double[nx_f * ny_f];
    cudaMemcpy(h_T, T, nx_f * ny_f * sizeof(double), cudaMemcpyDeviceToHost);

    // Write h_T to a file
    std::ofstream tfile("temperature_output.txt");
    for (int j = 0; j < ny_f; ++j) {
        for (int i = 0; i < nx_f; ++i) {
            tfile << h_T[j * nx_f + i] << " ";
        }
        tfile << std::endl;
    }
    tfile.close();
    delete[] h_T;    

    return 0;
}