#include <iostream>
#include "LidDrivenCavity.h"
#include <cuda.h>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

#define cudaCheck2(err) (cudaCheck(err, __FILE__, __LINE__))


void cudaCheck(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
           cudaGetErrorString(error));
    exit(EXIT_FAILURE);
  }
};

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

// Kernel function for bottom boundary condition - No tiling or shared memory
__global__ void bottom_bc(double *u, double * v, double * p, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < nx ) {
        double x = (0.5 + idx) * dx;
        u[idx + 1] = 0.0; // Wall velocity
        v[idx + 1] = 0.0; // No penetration
        p[idx + 1] = p[(nx+2) + idx + 1]; // Reference pressure (could be zero or Neumann)
    }
}

// Kernel function for upper boundary condition - No tiling or shared memory
__global__ void upper_bc(double *u, double * v, double * p, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < nx ) {
        double x = (0.5 + idx) * dx;
        u[(ny * (nx+2)) + idx + 1] = 1.0; // Lid velocity
        v[(ny * (nx+2)) + idx + 1] = 0.0; // No penetration
        p[(ny * (nx+2)) + idx + 1] = p[((ny-1) * (nx+2)) + idx + 1]; // Reference pressure (could be zero or Neumann)
    }
}

// Kernel function for left boundary condition - No tiling or shared memory
__global__ void left_bc(double *u, double * v, double * p, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ny ) {
        double y = (0.5 + idx) * dy;
        u[( (idx+1) * (nx+2)) + 0] = 0.0; // Wall velocity
        v[( (idx+1) * (nx+2)) + 0] = 0.0; // No penetration
        p[( (idx+1) * (nx+2)) + 0] = p[( (idx+1) * (nx+2)) + 1]; // Reference pressure (could be zero or Neumann)
    }
}

// Kernel function for right boundary condition - No tiling or shared memory
__global__ void right_bc(double *u, double * v, double * p, int nx, int ny, double dx, double dy) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < ny ) {
        double y = (0.5 + idx) * dy;
        u[( (idx+1) * (nx+2)) + nx] = 0.0; // Wall velocity
        v[( (idx+1) * (nx+2)) + nx] = 0.0; // No penetration
        p[( (idx+1) * (nx+2)) + nx] = p[( (idx+1) * (nx+2)) + nx - 1]; // Reference pressure (could be zero or Neumann)
    }
}

// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny))
        T[idx] += deltaT[idx];

}

__global__ void compute_gradp_kernel(double * p, double * gpx, double * gpy, int nx, int ny, double dx, double dy) {

    int j = blockIdx.x * blockDim.x + threadIdx.x; // row index
    int i = blockIdx.y * blockDim.y + threadIdx.y; // col index

    if ((i < nx) && (j < ny)) {
        int idx = (j + 1) * (nx + 2) + (i + 1);
        gpx[idx]     = (p[idx + 1] - p[idx - 1]) / (2.0 * dx); // dp/dx
        gpy[idx] = (p[idx + (nx + 2)] - p[idx - (nx + 2)]) / (2.0 * dy); // dp/dy
        //printf("i: %d, j: %d, idx: %d, dp/dx: %f, dp/dy: %f\n", i, j, idx, gpx[idx], gpy[idx]);
    }
}

__device__ double face_flux_mom(double ul, double ur,
                                double pl, double pr,
                                double gpl, double gpr,
                                double a_invl, double a_invr,
                                double deltaface, double deltaperp) {
    double df_inv = 1.0 / deltaface;
    return 0.5 * (  (ul + ur)
                + 0.5 * df_inv * (a_invl * gpl + a_invr * gpr)
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
                               const __grid_constant__ CUtensorMap tensor_map_gpx,
                               const __grid_constant__ CUtensorMap tensor_map_gpy,
                               const __grid_constant__ CUtensorMap tensor_map_a_inv,
                               const __grid_constant__ CUtensorMap tensor_map_u_nlr,
                               const __grid_constant__ CUtensorMap tensor_map_v_nlr,
                               double * u, double * v, double * p, double * gpx, double * gpy, double * a_inv,
                               double * u_nlr, double * v_nlr, double * Jmom,
                               int nx, int ny, double dx, double dy,
                               double nu, double dt) {

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;

    extern __shared__ SharedMemory shared_mem[];
    double * us = reinterpret_cast<double*>(shared_mem);
    double * vs = us + 76 * 16;
    double * ps = vs + 76 * 16;
    double * gpxs = ps + 76 * 16;
    double * gpys = gpxs + 76 * 16;
    double * a_invs = gpys + 76 * 16;
    // Continue to keep memory allocation for u_nlr, v_nlr and
    // Jmom as (BM+2)*(BN+2). However, I will only compute
    // these for BM * BN cells.
    double * u_nlrs = a_invs + 76 * 16;
    double * v_nlrs = u_nlrs + 76 * 16;
    double * Jmoms = v_nlrs + 76 * 16;

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
        token[0] = cuda::device::barrier_arrive_tx(bar_a[0], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(vs, &tensor_map_vmom, cCol * BN, cRow * BM, bar_a[1]);
        token[1] = cuda::device::barrier_arrive_tx(bar_a[1], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(ps, &tensor_map_p, cCol * BN, cRow * BM, bar_a[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar_a[2], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(gpxs, &tensor_map_gpx, cCol * BN, cRow * BM, bar_a[3]);
        token[3] = cuda::device::barrier_arrive_tx(bar_a[3], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(gpys, &tensor_map_gpy, cCol * BN, cRow * BM, bar_a[4]);
        token[4] = cuda::device::barrier_arrive_tx(bar_a[4], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(a_invs, &tensor_map_a_inv, cCol * BN, cRow * BM, bar_a[5]);
        token[5] = cuda::device::barrier_arrive_tx(bar_a[5], 1, (BM+2) * (BN+2) * sizeof(double));
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
    int sidx = (threadRow + 1) * (BN + 2) + threadCol + 1;

    // Step 2 - Compute Jmom, u_nlr and v_nlr
    double phi_w = face_flux_mom(us[sidx - 1], us[sidx],     ps[sidx - 1], ps[sidx],     gpxs[sidx - 1], gpxs[sidx], a_invs[sidx - 1], a_invs[sidx],     dx, dy);
    double phi_e = face_flux_mom(us[sidx],     us[sidx + 1], ps[sidx],     ps[sidx + 1], gpxs[sidx], gpxs[sidx + 1], a_invs[sidx],     a_invs[sidx + 1], dx, dy);
    double phi_s = face_flux_mom(vs[sidx - (BN + 2)], vs[sidx],            ps[sidx - (BN + 2)], ps[sidx],            gpys[sidx - (BN + 2)], gpys[sidx],   a_invs[sidx - (BN + 2)], a_invs[sidx],            dy, dx);
    double phi_n = face_flux_mom(vs[sidx],            vs[sidx + (BN + 2)], ps[sidx],            ps[sidx + (BN + 2)], gpys[sidx],   gpys[sidx + (BN + 2)], a_invs[sidx],            a_invs[sidx + (BN + 2)], dy, dx);

    u_nlrs[sidx] += dx * dy * dt_inv + 0.5 * (ps[sidx + 1] - ps[sidx - 1])
                    - (phi_w > 0.0 ? us[sidx-1] : us[sidx]) * phi_w
                    + (phi_e > 0.0 ? us[sidx]   : us[sidx+1]) * phi_e
                    - (phi_s > 0.0 ? us[sidx - (BN + 2)] : us[sidx]           ) * phi_s
                    + (phi_n > 0.0 ? us[sidx]            : us[sidx + (BN + 2)]) * phi_n
                    - nu * ( 4.0 * us[sidx] - us[sidx-1] - us[sidx+1] - us[sidx - (BN + 2)] - us[sidx + (BN + 2)]);

    v_nlrs[sidx] += dx * dy * dt_inv + 0.5 * (ps[sidx + (BN + 2)] - ps[sidx - (BN + 2)])
                    - (phi_w > 0.0 ? vs[sidx-1] : vs[sidx]) * phi_w
                    + (phi_e > 0.0 ? vs[sidx]   : vs[sidx+1]) * phi_e
                    - (phi_s > 0.0 ? vs[sidx - (BN + 2)] : vs[sidx]           ) * phi_s
                    + (phi_n > 0.0 ? vs[sidx]            : vs[sidx + (BN + 2)]) * phi_n
                    - nu * ( 4.0 * vs[sidx] - vs[sidx-1] - vs[sidx+1] - vs[sidx - (BN + 2)] - vs[sidx + (BN + 2)]);

    // printf("Thread (%d,%d) sidx: %d, u_nlr: %f, v_nlr: %f\n", threadRow, threadCol, sidx, u_nlrs[sidx], v_nlrs[sidx]);

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_u_nlr, cCol * BN, cRow * BM,
                                                    u_nlrs);
      // Wait for TMA transfer to have finished reading shared memory.
      // Create a "bulk async-group" out of the previous bulk copy operation.
      cde::cp_async_bulk_commit_group();
      // Wait for the group to have completed reading from shared memory.
      cde::cp_async_bulk_wait_group_read<0>();

      cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_v_nlr, cCol * BN, cRow * BM,
                                                    v_nlrs);
      // Wait for TMA transfer to have finished reading shared memory.
      // Create a "bulk async-group" out of the previous bulk copy operation.
      cde::cp_async_bulk_commit_group();
      // Wait for the group to have completed reading from shared memory.
      cde::cp_async_bulk_wait_group_read<0>();

      for (int i = 0; i < 6; i++)
        (&bar_a[i])->~barrier();
    }

}

template <const int BM, const int BN>
__global__ void compute_rcont_j(const __grid_constant__ CUtensorMap tensor_map_umom,
                               const __grid_constant__ CUtensorMap tensor_map_vmom,
                               const __grid_constant__ CUtensorMap tensor_map_p,
                               const __grid_constant__ CUtensorMap tensor_map_gpx,
                               const __grid_constant__ CUtensorMap tensor_map_gpy,
                               const __grid_constant__ CUtensorMap tensor_map_a_inv,
                               const __grid_constant__ CUtensorMap tensor_map_cont_nlr,
                               double * u, double * v, double * p, double * gpx, double * gpy, double * a_inv,
                               double * cont_nlr, double * Jcont,
                               int nx, int ny, double dx, double dy) {

    const int cRow = blockIdx.y * BM;
    const int cCol = blockIdx.x * BN;

    extern __shared__ SharedMemory shared_mem[];
    double * us = reinterpret_cast<double*>(shared_mem);
    double * vs = us + 76 * 16;
    double * ps = vs + 76 * 16;
    double * gpxs = ps + 76 * 16;
    double * gpys = gpxs + 76 * 16;
    double * a_invs = gpys + 76 * 16;
    // Continue to keep memory allocation for cont_nlr and
    // Jcont as (BM+2)*(BN+2). However, I will only compute
    // these for BM * BN cells.
    double * cont_nlrs = a_invs + 76 * 16;
    double * Jconts = cont_nlrs + 76 * 16;

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
        token[0] = cuda::device::barrier_arrive_tx(bar_a[0], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(vs, &tensor_map_vmom, cCol * BN, cRow * BM, bar_a[1]);
        token[1] = cuda::device::barrier_arrive_tx(bar_a[1], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(ps, &tensor_map_p, cCol * BN, cRow * BM, bar_a[2]);
        token[2] = cuda::device::barrier_arrive_tx(bar_a[2], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(gpxs, &tensor_map_gpx, cCol * BN, cRow * BM, bar_a[3]);
        token[2] = cuda::device::barrier_arrive_tx(bar_a[3], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(gpys, &tensor_map_gpy, cCol * BN, cRow * BM, bar_a[4]);
        token[2] = cuda::device::barrier_arrive_tx(bar_a[4], 1, (BM+2) * (BN+2) * sizeof(double));
        cde::cp_async_bulk_tensor_2d_global_to_shared(a_invs, &tensor_map_a_inv, cCol * BN, cRow * BM, bar_a[5]);
        token[3] = cuda::device::barrier_arrive_tx(bar_a[5], 1, (BM+2) * (BN+2) * sizeof(double));
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
    /* TODO:
    1. Transfer data from global memory to shared memory for us, vs, ps, a_inv_s
    2. Compute Jcont, cont_nlr
    3. Transfer data from shared memory to global memory for Jcont, cont_nlr
    */

    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;
    int sidx = (threadRow + 1) * (BN + 2) + threadCol + 1;

    // Step 2 - Compute Jcont, cont_nlr
    double phi_w = face_flux_mom(us[sidx - 1], us[sidx],     ps[sidx - 1], ps[sidx],     gpxs[sidx - 1], gpxs[sidx], a_invs[sidx - 1], a_invs[sidx],     dx, dy);
    double phi_e = face_flux_mom(us[sidx],     us[sidx + 1], ps[sidx],     ps[sidx + 1], gpxs[sidx], gpxs[sidx + 1], a_invs[sidx],     a_invs[sidx + 1], dx, dy);
    double phi_s = face_flux_mom(vs[sidx - (BN + 2)], vs[sidx],            ps[sidx - (BN + 2)], ps[sidx],            gpys[sidx - (BN + 2)], gpys[sidx],   a_invs[sidx - (BN + 2)], a_invs[sidx],            dy, dx);
    double phi_n = face_flux_mom(vs[sidx],            vs[sidx + (BN + 2)], ps[sidx],            ps[sidx + (BN + 2)], gpys[sidx],   gpys[sidx + (BN + 2)], a_invs[sidx],            a_invs[sidx + (BN + 2)], dy, dx);

    cont_nlrs[sidx] += phi_e - phi_w + phi_n - phi_s;

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
      cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_cont_nlr, cCol * BN, cRow * BM,
                                                    cont_nlrs);
      // Wait for TMA transfer to have finished reading shared memory.
      // Create a "bulk async-group" out of the previous bulk copy operation.
      cde::cp_async_bulk_commit_group();
      // Wait for the group to have completed reading from shared memory.
      cde::cp_async_bulk_wait_group_read<0>();

      for (int i = 0; i < 6; i++)
        (&bar_a[i])->~barrier();
    }

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
  //int shared_memory_size = (BM + 2) * (BN + 2) * sizeof(double) * 13;
  int shared_memory_size = 76 * 128 * 10;

  cudaFuncSetAttribute(compute_rmom_j<BM, BN>, 
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      shared_memory_size);

  compute_rmom_j<BM,BN>
        <<<grid_size, block_size, shared_memory_size>>>(get_tensor_map(umom, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(vmom, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(pres, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(gpx, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(gpy, nx+2, ny+2, BM+2, BN+2),                                               
                                                        get_tensor_map(a_inv, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(u_nlr, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(v_nlr, nx+2, ny+2, BM+2, BN+2),
                                                        umom, vmom, pres, gpx, gpy, a_inv, u_nlr, v_nlr, Jmom,
                                                        nx, ny, dx, dy, nu, dt);
  cudaCheck2(cudaDeviceSynchronize());
  return 1.0;
}

__host__ double LidDrivenCavity::compute_cont_r_j() {
  // Launch the kernel for computing the residuals
  const uint BM = 32;
  const uint BN = 32;
  int shared_memory_size = (BM + 2) * (BN + 2) * sizeof(double) * 12;
  cudaFuncSetAttribute(compute_rcont_j<BM, BN>, 
      cudaFuncAttributeMaxDynamicSharedMemorySize, 
      shared_memory_size);
  
  compute_rcont_j<BM, BN>
        <<<grid_size, block_size, shared_memory_size>>>(get_tensor_map(umom, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(vmom, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(pres, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(gpx, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(gpy, nx+2, ny+2, BM+2, BN+2),                                                        
                                                        get_tensor_map(a_inv, nx+2, ny+2, BM+2, BN+2),
                                                        get_tensor_map(cont_nlr, nx+2, ny+2, BM+2, BN+2),
                                                        umom, vmom, pres, gpx, gpy, a_inv, cont_nlr, Jcont, nx, ny, dx, dy);
  cudaCheck2(cudaDeviceSynchronize());
  return 1.0;
}


__host__ void LidDrivenCavity::compute_gradp() {
    // Compute the pressure gradient components using central differences
    
    compute_gradp_kernel<<<grid_size, block_size>>>(pres, gpx, gpy, nx, ny, dx, dy);
    cudaDeviceSynchronize();
}

__host__ void LidDrivenCavity::apply_bc() {
    cudaStream_t streams[4];
    for (int i = 0; i < 4; ++i) cudaStreamCreate(&streams[i]);

    bottom_bc<<<dim3(ceil(nx/1024.0)), dim3(1024), 0, streams[0]>>>(umom, vmom, pres, nx, ny, dx, dy);
    upper_bc<<<dim3(ceil(nx/1024.0)), dim3(1024), 0, streams[1]>>>(umom, vmom, pres, nx, ny, dx, dy);
    left_bc<<<dim3(ceil(ny/1024.0)), dim3(1024), 0, streams[2]>>>(umom, vmom, pres, nx, ny, dx, dy);
    right_bc<<<dim3(ceil(ny/1024.0)), dim3(1024), 0, streams[3]>>>(umom, vmom, pres, nx, ny, dx, dy);

    for (int i = 0; i < 4; ++i) cudaStreamSynchronize(streams[i]);
    for (int i = 0; i < 4; ++i) cudaStreamDestroy(streams[i]);
    cudaDeviceSynchronize();
}

LidDrivenCavity::LidDrivenCavity(int nx_inp, int ny_inp, double nu_inp) {

    nx = nx_inp;
    ny = ny_inp;
    dx = 1.0 / nx;
    dy = 3.0 / ny;
    nu = nu_inp;
    dt = 0.001;

    grid_size = dim3(nx/32, ny/32);
    block_size = dim3(1024);

    grid_size_1d = dim3( ceil ( nx * ny / 1024.0) );

    cudaMalloc(&umom, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&vmom, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&pres, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&gpx, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&gpy, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&a_inv, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&deltaU, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&deltaV, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&deltaP, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&Jmom, (nx + 2) * (ny + 2) * 5 * sizeof(double));
    cudaMalloc(&Jcont, (nx + 2) * (ny + 2) * 5 * sizeof(double));
    cudaMalloc(&u_nlr, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&v_nlr, (nx + 2) * (ny + 2) * sizeof(double));
    cudaMalloc(&cont_nlr, (nx + 2) * (ny + 2) * sizeof(double));

    std::cout << "Allocated " <<  21 * (nx + 2) * (ny + 2) * sizeof(double) / double(1 << 30) << " GB of memory" << std::endl;
    t_unlr = thrust::device_ptr<double>(u_nlr);
    t_vnlr = thrust::device_ptr<double>(v_nlr);
    t_cont_nlr = thrust::device_ptr<double>(cont_nlr);

    // initialize_rand<<<grid_size_1d, block_size_1d>>>(umom, nx+2, ny+2);
    // initialize_rand<<<grid_size_1d, block_size_1d>>>(vmom, nx+2, ny+2);
    // initialize_rand<<<grid_size_1d, block_size_1d>>>(pres, nx+2, ny+2);
    // initialize_rand<<<grid_size_1d, block_size_1d>>>(a_inv, nx+2, ny+2);
    initialize_const<<<grid_size_1d, block_size_1d>>>(deltaU, 0.0, nx, ny);
    initialize_const<<<grid_size_1d, block_size_1d>>>(deltaV, 0.0, nx, ny);
    initialize_const<<<grid_size_1d, block_size_1d>>>(deltaP, 0.0, nx, ny);

}

LidDrivenCavity::~LidDrivenCavity() {

    cudaFree(umom);
    cudaFree(vmom);
    cudaFree(pres);
    cudaFree(gpx);
    cudaFree(gpy);    
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

    for (int i = 0; i < 1e5; i++) {
        lcav->compute_mom_r_j();
        lcav->compute_cont_r_j();
        lcav->apply_bc();
        lcav->compute_gradp();
    }

    delete lcav;
    return 0;
}
