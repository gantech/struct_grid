#include <iostream>
#include "LidDrivenCavity.h"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

namespace LidDrivenCavityNS {

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_const(double *T, double val, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] = val ;
    
}

// // Kernel function for initialization - No tiling or shared memory
// __global__ void initialize_rand(double *T, int nx, int ny) {

//     int idx = blockIdx.x * blockDim.x + threadIdx.x;

//     if (idx < (nx * ny))  {

//         auto seed =  curand_init(seed, idx, 0, &states[idx]);

//         T[idx] = (double )curand_uniform();
//     }
    
// }

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
    return 0.5 * (  (ul + ur) 
                + 0.5 * df_inv * (a_invl * (prr - pl) + a_invr * (pr - pll)) 
                - (a_invl + a_invr) * (pr - pl) ) * deltaperp;
}


// Create an aligned shared memory using a union
union SharedMemory {
  char raw[1];
  __align__(128) double aligned;
};

template <const int BM, const int BN>
__global__ void compute_rmom_j(const __grid_constant__ CUtensorMap tensor_map_umom,
                               const __grid_constant__ CUtensorMap tensor_map_vmom,
                               const __grid_constant__ CUtensorMap tensor_map_p,
                               const __grid_constant__ CUtensorMap tensor_map_a_inv,
                               const __grid_constant__ CUtensorMap tensor_map_u_nlr,
                               const __grid_constant__ CUtensorMap tensor_map_v_nlr,
                               double * u, double * v, double * p, double * a_inv,
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

    // Initialize shared memory barrier with the number of threads participating in the barrier.
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar_a[6];
    if (threadIdx.x == 0) {
        for (int i = 0; i < 6; i++) {
          init(&bar_a[i], blockDim.x);
        }
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    barrier::arrival_token token[6];
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(us, &tensor_map_umom, cCol * BN, cRow * BM, bar_a[0]);
        token[0] = cuda::device::barrier_arrive_tx(bar_a[0], 1, (BM+4) * (BN+4) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(vs, &tensor_map_vmom, cCol * BN, cRow * BM, bar_a[1]);
        token[1] = cuda::device::barrier_arrive_tx(bar_a[1], 1, (BM+4) * (BN+4) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(ps, &tensor_map_p, cCol * BN, cRow * BM, bar_a[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar_a[2], 1, (BM+4) * (BN+4) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(a_invs, &tensor_map_a_inv, cCol * BN, cRow * BM, bar_a[3]);
        token[3] = cuda::device::barrier_arrive_tx(bar_a[3], 1, (BM+4) * (BN+4) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(u_nlrs, &tensor_map_u_nlr, cCol * BN, cRow * BM, bar_a[4]);
        token[4] = cuda::device::barrier_arrive_tx(bar_a[4], 1, (BM+4) * (BN+4) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(v_nlrs, &tensor_map_v_nlr, cCol * BN, cRow * BM, bar_a[5]);
        token[5] = cuda::device::barrier_arrive_tx(bar_a[5], 1, (BM+4) * (BN+4) * sizeof(double));
    } else {
        for (int i = 0; i < 6; i++) {
            token[i] = bar_a[i].arrive();
        }
    }

    for (int i=0; i < 6; i++) {
      bar_a[i].wait(std::move(token[i]));
    }
    

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
    double phi_w = face_flux_mom(us[sidx - 1], us[sidx],     ps[sidx - 1], ps[sidx],     ps[sidx - 2], ps[sidx + 1], a_invs[sidx - 1], a_invs[sidx],     dx, dy);
    double phi_e = face_flux_mom(us[sidx],     us[sidx + 1], ps[sidx],     ps[sidx + 1], ps[sidx - 1], ps[sidx + 2], a_invs[sidx],     a_invs[sidx + 1], dx, dy);
    double phi_s = face_flux_mom(vs[sidx - (BN + 4)], vs[sidx],            ps[sidx - (BN + 4)], ps[sidx],            ps[sidx - 2*(BN + 4)], ps[sidx + (BN + 4)],   a_invs[sidx - (BN + 4)], a_invs[sidx],            dy, dx);
    double phi_n = face_flux_mom(vs[sidx],            vs[sidx + (BN + 4)], ps[sidx],            ps[sidx + (BN + 4)], ps[sidx - (BN + 4)],   ps[sidx + 2*(BN + 4)], a_invs[sidx],            a_invs[sidx + (BN + 4)], dy, dx);
    
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


  if (threadIdx.x == 0) {
    for (int i = 0; i < 6; i++)
      (&bar_a[i])->~barrier();
  }                    

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
    double phi_w = face_flux_mom(us[sidx - 1], us[sidx],     ps[sidx - 1], ps[sidx],     ps[sidx - 2], ps[sidx + 1], a_invs[sidx - 1], a_invs[sidx],     dx, dy);
    double phi_e = face_flux_mom(us[sidx],     us[sidx + 1], ps[sidx],     ps[sidx + 1], ps[sidx - 1], ps[sidx + 2], a_invs[sidx],     a_invs[sidx + 1], dx, dy);
    double phi_s = face_flux_mom(vs[sidx - (BN + 4)], vs[sidx],            ps[sidx - (BN + 4)], ps[sidx],            ps[sidx - 2*(BN + 4)], ps[sidx + (BN + 4)],   a_invs[sidx - (BN + 4)], a_invs[sidx],            dy, dx);
    double phi_n = face_flux_mom(vs[sidx],            vs[sidx + (BN + 4)], ps[sidx],            ps[sidx + (BN + 4)], ps[sidx - (BN + 4)],   ps[sidx + 2*(BN + 4)], a_invs[sidx],            a_invs[sidx + (BN + 4)], dy, dx);
    
    cont_nlrs[sidx] += phi_e - phi_w + phi_n - phi_s;
}


PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
  // Get pointer to cuTensorMapEncodeTiled
  cudaDriverEntryPointQueryResult driver_status;
  void* cuTensorMapEncodeTiled_ptr = nullptr;
  cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status);
  assert(driver_status == cudaDriverEntryPointSuccess);
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
}
 
CUtensorMap get_tensor_map(double *A, const int M, const int N,
                           const int BM, const int BN) {

  CUtensorMap tensor_map_a{};
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;
  uint64_t size[rank] = {M, N};
  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {(N) * sizeof(double)};
  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {BN, BM};
  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res_a = cuTensorMapEncodeTiled(
    &tensor_map_a,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT64,
    rank,                       // cuuint32_t tensorRank,
    A,                 // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,
    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    // Don't set out-of-bounds elements to anything during TMA transfers
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  return tensor_map_a;

}

__host__ double LidDrivenCavity::compute_mom_r_j() {
  // Launch the kernel for computing the residuals
  const uint BM = 32;
  const uint BN = 32;
  int shared_memory_size = (BM + 4) * (BN + 4) * sizeof(double) * 11;  
  compute_rmom_j<BM,BN>
        <<<grid_size, block_size, shared_memory_size>>>( get_tensor_map(umom, nx+4, ny+4, BM+4, BN+4),
                                                       get_tensor_map(vmom, nx+4, ny+4, BM+4, BN+4),
                                                       get_tensor_map(pres, nx+4, ny+4, BM+4, BN+4),
                                                       get_tensor_map(a_inv, nx+4, ny+4, BM+4, BN+4),
                                                       get_tensor_map(u_nlr, nx+4, ny+4, BM+4, BN+4),
                                                       get_tensor_map(v_nlr, nx+4, ny+4, BM+4, BN+4),
                                                       umom, vmom, pres, a_inv, u_nlr, v_nlr, Jmom, 
                                                       nx, ny, dx, dy, nu, dt);
  return 1.0;
}

__host__ double LidDrivenCavity::compute_cont_r_j() {
  // Launch the kernel for computing the residuals
  const uint BM = 32;
  const uint BN = 32;
  int shared_memory_size = (BM + 4) * (BN + 4) * sizeof(double) * 10;
  compute_rcont_j<BM, BN>
        <<<grid_size, block_size, shared_memory_size>>>(umom, vmom, pres, a_inv, cont_nlr, Jcont, nx, ny, dx, dy);
  return 1.0;
}

LidDrivenCavity::LidDrivenCavity(int nx_inp, int ny_inp, double nu_inp) {

    nx = nx_inp;
    ny = ny_inp;
    dx = 1.0 / nx;
    dy = 3.0 / ny;
    nu = nu_inp;
    dt = 0.001;
    
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

    // initialize_rand<<<grid_size_1d, block_size_1d>>>(umom, nx+4, ny+4);
    // initialize_rand<<<grid_size_1d, block_size_1d>>>(vmom, nx+4, ny+4);
    // initialize_rand<<<grid_size_1d, block_size_1d>>>(pres, nx+4, ny+4);
    // initialize_rand<<<grid_size_1d, block_size_1d>>>(a_inv, nx+4, ny+4);
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

    for (int i = 0; i < 1e6; i++) {
        lcav->compute_mom_r_j();
        lcav->compute_cont_r_j();
    }

    delete lcav;
    return 0;
}
