#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/transform_reduce.h>
#include <thrust/device_vector.h>

#define TILE_SIZE 32
#define TILE_SIZE_ADI 2
#define NNX 128
#define NNY 384

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize(double *T, int nx, int ny, double dx, double dy);

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny, double dx, double dy);

// Kernel function for calculation of Jacobian and Residual - No tiling or shared memory
__global__ void compute_r_j(double *T, double *J, double *R, int nx, int ny, double dx, double dy, double kc);

// Kernel function for calculation of Residual - No tiling or shared memory
__global__ void compute_r(double *T, double * J, double *R, int nx, int ny, double dx, double dy, double kc) ;

// Functor to square the elements
struct square {
    __device__ double operator()(double a) {
        return a * a;
    }
};

// Kernel function for Thomas solves in the X direction - part of ADI 
__global__ void adi_x(double *T, double *J, double *R, int nx, int ny);

// Kernel function for Thomas solves in the Y direction - part of ADI 
__global__ void adi_y(double *T, double *J, double *R, int nx, int ny);


// Kernel function for calculation of Residual - No tiling or shared memory
__global__ void jacobi_iter(double *T, double *deltaT, double *J, double *R, int nx, int ny, double dx, double dy, double kc) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < nx) && (col < ny)) {
        
        int idx_r = (col * nx) + row;
        int idx_j = idx_r * 5;

        double jij = J[idx_j];
        // double jim1j = J[idx_j + 1];
        // double jip1j = J[idx_j + 2];
        // double jijm1 = J[idx_j + 3];
        // double jijp1 = J[idx_j + 4];

        // double tip1j = 0.0;
        // double tim1j = 0.0;
        // double tijp1 = 0.0;
        // double tijm1 = 0.0;

        // if (row == 0) {
        //     tip1j = deltaT[idx_r + 1];
        // } else if (row == (nx - 1)) {
        //     tim1j = deltaT[idx_r - 1];
        // } else {
        //     tip1j = deltaT[idx_r + 1];
        //     tim1j = deltaT[idx_r - 1];
        // }

        // if (col == 0) {
        //     tijp1 = deltaT[idx_r + nx];
        // } else if (col == (ny - 1)) {
        //     tijm1 = deltaT[idx_r - nx];
        // } else {
        //     tijm1 = deltaT[idx_r - nx];
        //     tijp1 = deltaT[idx_r + nx];
        // }

        // T[idx_r] += 0.9*(R[idx_r] - jim1j * tim1j - jip1j * tip1j - jijm1 * tijm1 - jijp1 * tijp1) / jij;
        T[idx_r] += 0.9 * (R[idx_r])/jij;
    }
}




int main() {

    // Problem size
    int nx = 128;
    int ny = 384;
    double dx = 1.0 / double(nx);
    double dy = 3.0 / double(ny);
    std::cout << "dx = " << dx << ", dy = " << dy << std::endl;
    double kc = 0.001;

    // Allocate memory for input and output matrices
    double *T, *deltaT, *J, *R;
    cudaMalloc(&T, nx * ny * sizeof(double));
    cudaMalloc(&deltaT, nx * ny * sizeof(double));
    cudaMalloc(&J, nx * ny * 5 * sizeof(double));
    cudaMalloc(&R, nx * ny * sizeof(double));

    thrust::device_ptr<double> t_res(R);

    // Grid and block size
    dim3 grid_size(ceil(nx / (double)TILE_SIZE), ceil(ny / (double)TILE_SIZE), 1);
    dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
    std::cout << "Grid size: " << grid_size.x << ", " << grid_size.y << std::endl;
    std::cout << "Block size: " << block_size.x << ", " << block_size.y << std::endl;

    initialize<<<grid_size, block_size>>>(T, nx, ny, dx, dy);
    
    compute_r_j<<<grid_size, block_size>>>(T, J, R, nx, ny, dx, dy, kc);

    // CUDA Events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start,0);


    // // Call the Jacobi iteration 1000 times
    // for (int i = 0; i < 100000; ++i) {
    // // std::cout << "Iteration: " << i << std::endl;
    //     jacobi_iter<<<grid_size, block_size>>>(T, deltaT, J, R, nx, ny, dx, dy, kc);
    //     // update<<<grid_size, block_size>>>(T, deltaT, nx, ny, dx, dy);
    //     compute_r<<<grid_size, block_size>>>(T, J, R, nx, ny, dx, dy, kc);

    //     double glob_resid = std::sqrt(thrust::transform_reduce(t_res, t_res + nx * ny, square(), 0.0, thrust::plus<double>()));
    //     std::cout << "Iter = " << i << ", Residual = " << glob_resid << std::endl;        
    // }
    // update<<<grid_size, block_size>>>(T, deltaT, nx, ny, dx, dy);

    dim3 grid_size_adix(ceil(ny / (double)TILE_SIZE_ADI), 1, 1);
    dim3 block_size_adi(TILE_SIZE_ADI, 1,1);
    dim3 grid_size_adiy(ceil(nx / (double)TILE_SIZE_ADI), 1, 1);

    for (int i = 0; i < 10000; i++) {
        //adi_x<<<grid_size_adix, block_size_adi, (5*nx*TILE_SIZE_ADI*sizeof(double))>>>(T, J, R, nx, ny);
        adi_x<<<grid_size_adix, block_size_adi>>>(T, J, R, nx, ny);
        update<<<grid_size, block_size>>>(T, deltaT, nx, ny, dx, dy);
        compute_r_j<<<grid_size, block_size>>>(T, J, R, nx, ny, dx, dy, kc);

        //adi_y<<<grid_size_adiy, block_size_adi, (5*ny*TILE_SIZE_ADI*sizeof(double))>>>(T, J, R, nx, ny);
        adi_y<<<grid_size_adiy, block_size_adi>>>(T, J, R, nx, ny);
        update<<<grid_size, block_size>>>(T, deltaT, nx, ny, dx, dy);
        compute_r_j<<<grid_size, block_size>>>(T, J, R, nx, ny, dx, dy, kc);

        double glob_resid = std::sqrt(thrust::transform_reduce(t_res, t_res + nx * ny, square(), 0.0, thrust::plus<double>()));
        std::cout << "Iter = " << i << ", Residual = " << glob_resid << std::endl;
    }

    cudaDeviceSynchronize();

    // Record stop event
    cudaEventRecord(stop);
    // Synchronize the events to ensure accurate time measurement
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the elapsed time
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    double *h_R = new double[nx * ny];
    cudaMemcpy(h_R, R, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);

    double inv_kcdxdy = 1.0 / (kc * dx * dy);
    // Write h_R to a file 
    std::ofstream outfile("output.txt");
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            outfile << h_R[j * nx + i] * inv_kcdxdy << " ";
        }
        outfile << std::endl;
    }
    outfile.close();
    delete[] h_R;

    double *h_T = new double[nx * ny];
    cudaMemcpy(h_T, T, nx * ny * sizeof(double), cudaMemcpyDeviceToHost);

    // Write h_T to a file
    std::ofstream tfile("temperature_output.txt");
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            tfile << h_T[j * nx + i] << " ";
        }
        tfile << std::endl;
    }
    tfile.close();
    delete[] h_T;

    // Free memory
    cudaFree(T);
    cudaFree(J);
    cudaFree(R);

    return 0;
}