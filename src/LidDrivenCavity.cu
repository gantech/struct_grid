#include <iostream>
#include "LidDrivenCavity.h"

namespace LidDrivenCavityNS {


// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_const(double *T, double val, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] = val ;
    
}

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_rand(double *T, double val, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] = val * (double)rand() / (double)RAND_MAX;
    
}

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] += deltaT[idx];

}

__device__ double face_flux_mom(double ul, double ur,
                                double pl, double pr, 
                                double pll, double prr,
                                double a_invl, double a_invr, 
                                double deltaface, double deltaperp) {
    double df_inv = 1.0 / deltaface;
    return flux = 0.5 * (  (ul + ur) 
                + 0.5 * df_inv * (a_invl * (prr - pl) + a_invr * (pr - pll)) 
                - (a_invl + a_invr) * (pr - pl) ) * deltaperp;
}


// Create an aligned shared memory using a union
union SharedMemory {
  char raw[1];
  __align__(128) double aligned;
};

template <const int BM, const int BN>
__global__ void compute_rmom_j(double * u, double * v, double * p, double * a_inv,
                               double * u_nlr, double * v_nlr, double * Jmom, 
                               int nx, int ny, double dx, double dy, 
                               double nu, double dt) {

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN; 


    extern __shared__ SharedMemory shared_mem[];    
    double * us = reinterpret_cast<double*>(shared_mem);
    double * vs = us + ((BM+4) * (BN+4));
    double * ps = vs + ((BM+4) * (BN+4));
    double * a_invs = ps + ((BM+4) * (BN+4));
    // Continue to keep memory allocation for u_nlr, v_nlr and 
    // Jmom as (BM+4)*(BN+4). However, I will only compute 
    // these for BM * BN cells.
    double * u_nlrs = a_invs + ((BM+4) * (BN+4)); 
    double * v_nlrs = u_nlrs + ((BM+4) * (BN+4));
    double * Jmoms = v_nlrs + ((BM+4) * (BN+4));

    double dx_inv = 1.0 / dx;
    double dy_inv = 1.0 / dy;
    double dt_inv = 1.0 / dt;
    /* TODO:
    1. Transfer data from global memory to shared memory for us, vs, ps, a_inv_s
    2. Compute Jmom, u_nlr and v_nlr
    3. Transfer data from shared memory to global memory for Jmom, u_nlr and v_nlr
    */
    
    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;
    int sidx = (threadRow + 2) * (BN + 4) + threadCol + 2;

    // Step 2 - Compute Jmom, u_nlr and v_nlr
    phi_w = face_flux_mom(us[sidx - 1], us[sidx],     ps[sidx - 1], ps[sidx],     ps[sidx - 2], ps[sidx + 1], a_invs[sidx - 1], a_invs[sidx],     dx, dy);
    phi_e = face_flux_mom(us[sidx],     us[sidx + 1], ps[sidx],     ps[sidx + 1], ps[sidx - 1], ps[sidx + 2], a_invs[sidx],     a_invs[sidx + 1], dx, dy);
    phi_s = face_flux_mom(vs[sidx - (BN + 4)], vs[sidx],            ps[sidx - (BN + 4)], ps[sidx],            ps[sidx - 2*(BN + 4)], ps[sidx + (BN + 4)],   a_invs[sidx - (BN + 4)], a_invs[sidx],            dy, dx);
    phi_n = face_flux_mom(vs[sidx],            vs[sidx + (BN + 4)], ps[sidx],            ps[sidx + (BN + 4)], ps[sidx - (BN + 4)],   ps[sidx + 2*(BN + 4)], a_invs[sidx],            a_invs[sidx + (BN + 4)], dy, dx);
    
    u_nlrs[sidx] += dx * dy * dt_inv + 0.5 * (ps[sidx + 1] - ps[sidx - 1]) 
                    - (phi_w > 0.0 ? us[sidx-1] : us[sidx]) * phi_w
                    + (phi_e > 0.0 ? us[sidx]   : us[sidx+1]) * phi_e
                    - (phi_s > 0.0 ? us[sidx - (BN + 4)] : us[sidx]           ) * phi_s
                    + (phi_n > 0.0 ? us[sidx]            : us[sidx + (BN + 4)]) * phi_n
                    - nu * ( 4.0 * us[sidx] - us[sidx-1] - us[sidx+1] - us[sidx - (BN + 4)] - us[sidx + (BN + 4)]);

    v_nlrs[sidx] += dx * dy * dt_inv + 0.5 * (ps[sidx + (BN + 4)] - ps[sidx - (BN + 4)])
                    - (phi_w > 0.0 ? vs[sidx-1] : vs[sidx]) * phi_w
                    + (phi_e > 0.0 ? vs[sidx]   : vs[sidx+1]) * phi_e
                    - (phi_s > 0.0 ? vs[sidx - (BN + 4)] : vs[sidx]           ) * phi_s
                    + (phi_n > 0.0 ? vs[sidx]            : vs[sidx + (BN + 4)]) * phi_n
                    - nu * ( 4.0 * vs[sidx] - vs[sidx-1] - vs[sidx+1] - vs[sidx - (BN + 4)] - vs[sidx + (BN + 4)]);

}

template <const int BM, const int BN>
__global__ void compute_rcont_j(double * u, double * v, double * p, double * a_inv,
                               double * cont_nlr, double * Jcont, 
                               int nx, int ny, double dx, double dy) {

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN; 
    
    extern __shared__ SharedMemory shared_mem[];    
    double * us = reinterpret_cast<double*>(shared_mem);
    double * vs = us + ((BM+4) * (BN+4));
    double * ps = vs + ((BM+4) * (BN+4));
    double * a_invs = ps + ((BM+4) * (BN+4));
    // Continue to keep memory allocation for u_nlr, v_nlr and 
    // Jmom as (BM+4)*(BN+4). However, I will only compute 
    // these for BM * BN cells.
    double * cont_nlrs = a_invs + ((BM+4) * (BN+4)); 
    double * Jconts = cont_nlrs + ((BM+4) * (BN+4));

    double dx_inv = 1.0 / dx;
    double dy_inv = 1.0 / dy;
    /* TODO:
    1. Transfer data from global memory to shared memory for us, vs, ps, a_inv_s
    2. Compute Jcont, cont_nlr
    3. Transfer data from shared memory to global memory for Jcont, cont_nlr
    */
    
    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;
    int sidx = (threadRow + 2) * (BN + 4) + threadCol + 2;

    // Step 2 - Compute Jcont, cont_nlr
    phi_w = face_flux_mom(us[sidx - 1], us[sidx],     ps[sidx - 1], ps[sidx],     ps[sidx - 2], ps[sidx + 1], a_invs[sidx - 1], a_invs[sidx],     dx, dy);
    phi_e = face_flux_mom(us[sidx],     us[sidx + 1], ps[sidx],     ps[sidx + 1], ps[sidx - 1], ps[sidx + 2], a_invs[sidx],     a_invs[sidx + 1], dx, dy);
    phi_s = face_flux_mom(vs[sidx - (BN + 4)], vs[sidx],            ps[sidx - (BN + 4)], ps[sidx],            ps[sidx - 2*(BN + 4)], ps[sidx + (BN + 4)],   a_invs[sidx - (BN + 4)], a_invs[sidx],            dy, dx);
    phi_n = face_flux_mom(vs[sidx],            vs[sidx + (BN + 4)], ps[sidx],            ps[sidx + (BN + 4)], ps[sidx - (BN + 4)],   ps[sidx + 2*(BN + 4)], a_invs[sidx],            a_invs[sidx + (BN + 4)], dy, dx);
    
    cont_nlrs[sidx] += phi_e - phi_w + phi_n - phi_s;
}


__host__ double LidDrivenCavity::compute_mom_r_j() {
  // Launch the kernel for computing the residuals
  const uint BM = 32;
  const uint BN = 32;
  int shared_memory_size = (BM + 4) * (BN + 4) * sizeof(double) * 11;  
  compute_rmom_j<<<grid_size, block_size, shared_memory_size>>>(umom, vmom, pres, a_inv, u_nlr, v_nlr, Jmom, nx, ny, dx, dy);
  return 1.0;
}

__host__ double LidDrivenCavity::compute_cont_r_j() {
  // Launch the kernel for computing the residuals
  const uint BM = 32;
  const uint BN = 32;
  int shared_memory_size = (BM + 4) * (BN + 4) * sizeof(double) * 10;
  compute_rcont_j<<<grid_size, block_size, shared_memory_size>>>(umom, vmom, pres, a_inv, cont_nlr, Jcont, nx, ny, dx, dy);
  return 1.0;
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

    cudaMalloc(&umom, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&vmom, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&pres, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&phi, 2 * nx * ny * sizeof(double));
    cudaMalloc(&a_inv, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&deltaU, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&deltaV, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&deltaP, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&Jmom, (nx + 4) * (ny + 4) * 5 * sizeof(double));
    cudaMalloc(&Jcont, (nx + 4) * (ny + 4) * 5 * sizeof(double));
    cudaMalloc(&u_nlr, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&v_nlr, (nx + 4) * (ny + 4) * sizeof(double));
    cudaMalloc(&cont_nlr, (nx + 4) * (ny + 4) * sizeof(double));

    std::cout << "Allocated " <<  19 * (nx + 4) * (ny + 4) * sizeof(double) / double(1 << 30) << " GB of memory" << std::endl;
    t_unlr = thrust::device_ptr<double>(u_nlr);
    t_vnlr = thrust::device_ptr<double>(v_nlr);
    t_cont_nlr = thrust::device_ptr<double>(cont_nlr);

    initialize_rand<<<grid_size_1d, block_size_1d>>>(umom, nx+4, ny+4);
    initialize_rand<<<grid_size_1d, block_size_1d>>>(vmom, nx+4, ny+4);
    initialize_rand<<<grid_size_1d, block_size_1d>>>(pres, nx+4, ny+4);
    initialize_rand<<<grid_size_1d, block_size_1d>>>(a_inv, nx+4, ny+4);
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

    for (int i = 0; i < 100; i++) {
        lcav->compute_rmom_j();
        lcav->compute_rcont_j();
    }

    delete lcav;
    return 0;
}