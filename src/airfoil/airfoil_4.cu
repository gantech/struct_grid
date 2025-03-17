#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>

#define TILE_SIZE 16
#define TILE_SIZE_ADI 2
#define NDIM 2

__global__ void compute_area(double * pts, double * area, int nxp, int nyp);
__global__ void compute_if(double * pts, double * cell_center, double * area, int nx, int ny, int nxp, int nyp);
__global__ void compute_cellcenter(double * pts, double * cell_center, int nx, int ny, int nxp, int nyp) ;



// Kernel to initialize a field phi as x^2 + y^3
__global__ void initialize_phi(double * pts, double * phi, double * phi_bc_bot, double * phi_bc_top, int nx, int ny, int nxp, int nyp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int idx_p = ( (j * nxp) + i ) * NDIM;
    int idx_phi = ( (j * nx) + i ) ;

    if ( (i < nx) && (j < ny)) { // Make sure you're within the grid

        double xij = pts[idx_p];
        double yij = pts[idx_p + 1];
        double xip1j = pts[idx_p + NDIM];
        double yip1j = pts[idx_p + NDIM + 1];
        double xijp1 = pts[idx_p + nxp * NDIM];
        double yijp1 = pts[idx_p + nxp * NDIM + 1];
        double xip1jp1 = pts[idx_p + nxp * NDIM + NDIM];
        double yip1jp1 = pts[idx_p + nxp * NDIM + NDIM + 1];

        double x = 0.25 * (xij + xip1j + xijp1 + xip1jp1);
        double y = 0.25 * (yij + yip1j + yijp1 + yip1jp1);

        phi[idx_phi] = x * x + y * y * y;

        if (j == 0) {
            x = 0.5 * (xij + xip1j);
            y = 0.5 * (yij + yip1j);
            phi_bc_bot[i] = x * x + y * y * y;
        } else if ( j == (ny - 1)) {
            x = 0.5 * (xijp1 + xip1jp1);
            y = 0.5 * (yijp1 + yip1jp1);
            phi_bc_top[i] = x * x + y * y * y;
        }

    }

}

// Kernel to compute gradient of the reference field phi as 2x iHat + 3y^2 jHat and laplacian of the reference field as 2.0 + 6.0 * y
__global__ void reference_grad_lapl_phi(double * pts, double * grad_phi_ref, double * lapl_phi_ref, int nx, int ny, int nxp, int nyp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int idx_p = ( (j * nxp) + i ) * NDIM;
    int idx_phi = ( j * nx + i);
    int idx_grad_phi = ( (j * nx) + i ) * NDIM;

    if ( (i < nx) && (j < ny)) { // Make sure you're within the grid

        double xij = pts[idx_p];
        double yij = pts[idx_p + 1];
        double xip1j = pts[idx_p + NDIM];
        double yip1j = pts[idx_p + NDIM + 1];
        double xijp1 = pts[idx_p + nxp * NDIM];
        double yijp1 = pts[idx_p + nxp * NDIM + 1];
        double xip1jp1 = pts[idx_p + nxp * NDIM + NDIM];
        double yip1jp1 = pts[idx_p + nxp * NDIM + NDIM + 1];

        double x = 0.25 * (xij + xip1j + xijp1 + xip1jp1);
        double y = 0.25 * (yij + yip1j + yijp1 + yip1jp1);

        grad_phi_ref[idx_grad_phi] = 2.0 * x;
        grad_phi_ref[idx_grad_phi + 1] = 3.0 * y * y;
        lapl_phi_ref[idx_phi] = 2.0 + 6.0 * y;

    }

}

// Kernel to compute vector gradient of phi
/*

    There are 5 quantities for every point.
    1. Jacobian ( x_psi * y_eta - y_psi * x_eta)
    2. alpha_psi_x = y_eta
    3. alpha_psi_y = -x_eta
    4. alpha_eta_x = y_psi
    5. alpha_eta_y = x_psi

    Once these quantities are computed, the x- and y-derivative of a variable phi are
    
        phi_x = 1/J ( (phi * alpha_psi_x)_psi + (phi * alpha_eta_x)_eta )
        phi_y = 1/J ( (phi * alpha_psi_y)_psi + (phi * alpha_eta_y)_eta )
   
*/
__global__ void vector_grad_gauss(double * phi, double * grad_phi, double * grad_phi_ref,  double * area, double * phi_bc_bot, double * phi_bc_top, int nx, int ny, int nxp, int nyp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx_phi = (j * nx + i);
    int idx_phi_ip1 = (j * nx + (i+1)%nx );
    int idx_phi_im1 = (j * nx + (i-1) );
    if (i == 0)
        idx_phi_im1 = (j * nx + nx-1 );
    int idx_gp = (j * nx + i) * NDIM;

    int idx_a = (j * nxp + i) * 7;



    if ( (i < nx) && (j < ny)) { // Make sure you're within the grid
        // printf("BlockIdx.x %d, BlockIdx.y %d, threadIdx.x %d, threadIdx.y %d, i = %d, j = %d - idx_gp %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j, idx_gp);
        double phi_xix_e = 0.0;
        double phi_xix_w = 0.0;
        double phi_xiy_e = 0.0;
        double phi_xiy_w = 0.0;

        double phi_etax_n = 0.0;
        double phi_etay_n = 0.0;
        double phi_etax_s = 0.0;
        double phi_etay_s = 0.0;

        double phiij = phi[idx_phi];
        double phiijp1 = 0.0;
        double phiijm1 = 0.0;

        if (j < (ny -1 ))
            phiijp1 = phi[idx_phi + nx];
        if (j > 0)
            phiijm1 = phi[idx_phi - nx];

        if ( j == 0) {

            double phiijp1 = phi[idx_phi + nx];
            phi_etax_s = phi_bc_bot[i]  * area[idx_a];
            phi_etax_n = (area[idx_a + nxp  * 7 + 5] * phiijp1 + (1.0 - area[idx_a + nxp * 7 + 5]) * phiij ) * area[idx_a + nxp * 7];
            phi_etay_s = phi_bc_bot[i]  * area[idx_a + 1];
            phi_etay_n = (area[idx_a + nxp * 7 + 5] * phiijp1 + (1.0 - area[idx_a + nxp * 7 + 5]) * phiij ) * area[idx_a + nxp * 7 + 1];         

        } else if (j == (ny - 1)) {

            double phiijm1 = phi[idx_phi - nx];
            phi_etax_s =  ( area[idx_a + 5] * phiij + (1.0 - area[idx_a + 5]) * phiijm1 ) * area[idx_a];
            phi_etax_n = phi_bc_top[i]  * area[idx_a + nxp * 7];
            phi_etay_s = ( area[idx_a + 5] * phiij + (1.0 - area[idx_a + 5]) * phiijm1 ) * area[idx_a + 1];
            phi_etay_n = phi_bc_top[i]  * area[idx_a + nxp * 7 + 1];         

        } else {

            double phiijm1 = phi[idx_phi - nx];
            double phiijp1 = phi[idx_phi + nx];
            phi_etax_s = (area[idx_a + 5] * phiij + (1.0 - area[idx_a + 5]) * phiijm1 ) * area[idx_a];
            phi_etax_n = (area[idx_a + nxp * 7 + 5] * phiijp1 + (1.0 - area[idx_a + nxp * 7 + 5]) * phiij ) * area[idx_a + nxp * 7];
            phi_etay_s = (area[idx_a + 5] * phiij + (1.0 - area[idx_a + 5]) * phiijm1 ) * area[idx_a + 1];
            phi_etay_n = (area[idx_a + nxp * 7 + 5] * phiijp1 + (1.0 - area[idx_a + nxp * 7 + 5]) * phiij ) * area[idx_a + nxp * 7 + 1];         

        }
    
        double phiip1j = phi[idx_phi_ip1];
        double phiim1j = phi[idx_phi_im1];
        phi_xix_w = ( area[idx_a + 6] * phiij + (1.0 - area[idx_a + 6]) * phiim1j ) * area[idx_a + 2];
        phi_xix_e = ( area[idx_a + 7 + 6] * phiip1j + (1.0 - area[idx_a + 7 + 6]) * phiij ) * area[idx_a + 7 + 2];
        phi_xiy_w = ( area[idx_a + 6] * phiij + (1.0 - area[idx_a + 6]) * phiim1j ) * area[idx_a + 3];
        phi_xiy_e = ( area[idx_a + 7 + 6] * phiip1j + (1.0 - area[idx_a + 7 + 6]) * phiij ) * area[idx_a + 7 + 3];
        
        // printf("i %d j %d - %d \n", i, j, idx_gp);
        double tmp = (phi_xix_e - phi_xix_w + phi_etax_n - phi_etax_s)/area[idx_a + 4];
        double tmp1 = (phi_xiy_e - phi_xiy_w + phi_etay_n - phi_etay_s)/area[idx_a + 4];
        grad_phi[idx_gp] = tmp;
        grad_phi[idx_gp + 1] = tmp1;

        // if ( (tmp * grad_phi_ref[idx_gp]) < 0.0 )
        //    printf("i %d, j %d, grad_phi_x = %e, grad_phi_ref_x %e, phi_x_ew = %e, phi_x_ns = %e, IF: (S) %e, (W) %e, (N) %e, (E) %e, Total %e, Phi (I-1) %e, (I), %e, (I+1) %e, (J-1) %e, (J+1) %e \n", i, j, tmp, grad_phi_ref[idx_gp], (phi_xix_e - phi_xix_w)/area[idx_a+4], (phi_etax_n - phi_etax_s)/area[idx_a+4], area[idx_a+5], area[idx_a+6], area[idx_a+nxp*7+5], area[idx_a+7+5], area[idx_a+4], phiim1j, phiij, phiip1j, phiijm1, phiijp1);
        // if (i == 0)
        //     printf("i %d, j %d, grad_phi_x = %e, phi_xiy_e = %e, phi_xiy_w = %e, phi_etay_n = %e, phi_etay_s = %e, Areas: (S) %e, %e, (N) %e, %e, (E) %e, %e, (W) %e, %e, Phi (I-1) %e, (I), %e, (I+1) %e, (J-1) %e, (J+1) %e\n", i, j, tmp1, phi_xiy_e, phi_xiy_w, phi_etay_n, phi_etay_s, area[idx_a], area[idx_a+1], area[idx_a+nxp*7], area[idx_a+nxp*7+1], area[idx_a+7+2], area[idx_a+7+3], area[idx_a+2], area[idx_a+3], phiim1j, phiij, phiip1j, phiijm1, phiijp1);
    }
}

// Inline device function (can be called from kernels)
__device__ __inline__ double mag(double x, double y) {
    return std::sqrt(x * x + y * y);
}

// Inline device function (can be called from kernels)
__device__ __inline__ double lin_interp(double x1, double x2, double t) {
    return t * x1 + (1.0 - t) * x2;
}

__global__ void compute_r_j(double * phi, double * grad_phi, double *jac, double * res, double * area, double * phi_bc_bot, double * phi_bc_top, double * cell_center, double * pts, int nx, int ny, int nxp, int nyp) {
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if ( (i < nx) && ( j < ny)) {


        int idx_jac = (j * nx + i) * 5;

        int idx_gp = (j * nx + i) * NDIM;
        int idx_gp_im1 = (j * nx + (i-1) ) * NDIM;
        if (i == 0)
            idx_gp_im1 = (j * nx + nx-1 ) * NDIM;
        int idx_gp_ip1 = (j * nx + (i+1)%nx ) * NDIM;

        int idx_phi = (j * nx + i);
        int idx_phi_im1 = (j * nx + (i-1) );
        if (i == 0)
            idx_phi_im1 = (j * nx + nx-1 );
        int idx_phi_ip1 = (j * nx + (i+1)%nx );

        int idx_a = (j * nxp + i) * 7;

        double rCx = cell_center[idx_gp];
        double rCy = cell_center[idx_gp + 1];
        double jac_c = 0.0;
        double loc_res = 0.0;

        //West face
        double rFx = cell_center[idx_gp_im1];
        double rFy = cell_center[idx_gp_im1 + 1];
        double dCF = mag(rFx-rCx, rFy-rCy);
        double dCFx = (rFx - rCx)/dCF;
        double dCFy = (rFy - rCy)/dCF;

        double Ef = - (area[idx_a+2]*area[idx_a+2]+area[idx_a+3]*area[idx_a+3])/(dCFx * area[idx_a+2] + dCFy * area[idx_a+3]);
        double gphifx = lin_interp(grad_phi[idx_gp], grad_phi[idx_gp_im1], area[idx_a+6]);
        double gphify = lin_interp(grad_phi[idx_gp+1], grad_phi[idx_gp_im1+1], area[idx_a+6]);
        loc_res += (phi[idx_phi_im1]-phi[idx_phi])*Ef/dCF + gphifx * (-area[idx_a+2] - Ef * dCFx) + gphify * (-area[idx_a+3] - Ef * dCFy);
        jac_c = -Ef/dCF;
        jac[idx_jac+1] = Ef/dCF;

        //East face
        rFx = cell_center[idx_gp_ip1];
        rFy = cell_center[idx_gp_ip1 + 1];
        dCF = mag(rFx-rCx, rFy-rCy);
        dCFx = (rFx - rCx)/dCF;
        dCFy = (rFy - rCy)/dCF;

        Ef = (area[idx_a+7+2]*area[idx_a+7+2]+area[idx_a+7+3]*area[idx_a+7+3])/(dCFx * area[idx_a+7+2] + dCFy * area[idx_a+7+3]);
        gphifx = lin_interp(grad_phi[idx_gp_ip1], grad_phi[idx_gp], area[idx_a+7+6]);
        gphify = lin_interp(grad_phi[idx_gp_ip1+1], grad_phi[idx_gp+1], area[idx_a+7+6]);
        loc_res += (phi[idx_phi_ip1]-phi[idx_phi])*Ef/dCF + gphifx * (area[idx_a+7+2] - Ef * dCFx) + gphify * (area[idx_a+7+3] - Ef * dCFy);
        jac_c -= Ef/dCF;
        jac[idx_jac+2] = Ef/dCF;

        //South face
        if (j > 0) {
            rFx = cell_center[idx_gp-nx*NDIM];
            rFy = cell_center[idx_gp-nx*NDIM + 1];
            dCF = mag(rFx-rCx, rFy-rCy);
            dCFx = (rFx - rCx)/dCF;
            dCFy = (rFy - rCy)/dCF;

            Ef = - (area[idx_a] * area[idx_a] + area[idx_a+1] * area[idx_a+1])/(dCFx * area[idx_a] + dCFy * area[idx_a+1]);
            gphifx = lin_interp(grad_phi[idx_gp], grad_phi[idx_gp-nx*NDIM], area[idx_a+5]);
            gphify = lin_interp(grad_phi[idx_gp+1], grad_phi[idx_gp-nx*NDIM+1], area[idx_a+5]);
            loc_res += (phi[idx_phi-nx]-phi[idx_phi])*Ef/dCF + gphifx * (-area[idx_a] - Ef * dCFx) + gphify * (-area[idx_a+1] - Ef * dCFy);
            jac_c -= Ef/dCF;
            jac[idx_jac+3] = Ef/dCF;
        } else {
            rFx = pts[i * NDIM];
            rFy = pts[i * NDIM + 1];
            dCF = mag(rFx-rCx, rFy-rCy);

            loc_res += (phi_bc_bot[i]-phi[idx_phi])/dCF;
            jac_c -= 1.0/dCF;
        }

        //North face
        if (j < (ny - 1)) {   
            rFx = cell_center[idx_gp+nx*NDIM];
            rFy = cell_center[idx_gp+nx*NDIM + 1];
            dCF = mag(rFx-rCx, rFy-rCy);
            dCFx = (rFx - rCx)/dCF;
            dCFy = (rFy - rCy)/dCF;

            Ef = (area[idx_a+nxp*7]*area[idx_a+nxp*7]+area[idx_a+nxp*7+1]*area[idx_a+nxp*7+1])/(dCFx * area[idx_a+nxp*7] + dCFy * area[idx_a+nxp*7+1]);
            gphifx = lin_interp(grad_phi[idx_gp+nx*NDIM], grad_phi[idx_gp], area[idx_a+nxp*7+5]);
            gphify = lin_interp(grad_phi[idx_gp+nx*NDIM+1], grad_phi[idx_gp+1], area[idx_a+nxp*7+5]);
            loc_res += (phi[idx_phi+nx]-phi[idx_phi])*Ef/dCF + gphifx * (area[idx_a+nxp*7] - Ef * dCFx) + gphify * (area[idx_a+nxp*7+1] - Ef * dCFy);
            jac_c -= Ef/dCF;
            jac[idx_jac+4] = Ef/dCF;
        } else if (j == (ny - 1)) {
            rFx = pts[ (j * nxp + i) * NDIM];
            rFy = pts[ (j * nxp + i) * NDIM + 1];
            dCF = mag(rFx-rCx, rFy-rCy);
            loc_res += (phi_bc_top[i]-phi[idx_phi])/dCF;
            jac_c -= 1.0/dCF;
        }

        res[idx_phi] = loc_res;
        jac[idx_jac] = jac_c;

    }
    
}


int main() {

    // Read the airfoil data

    std::ifstream plot3dfile("du00w212.x");

    if (!plot3dfile.is_open()) {
        std::cerr << "Error: could not open file" << std::endl;
        return 1;
    }

    int nxp, nyp, ntotp;
    plot3dfile >> nxp >> nyp;
    std::cout << "nxp: " << nxp << " nyp: " << nyp << std::endl;
    ntotp = nxp * nyp ;
    double * h_pts = new double[ntotp * NDIM];
    
    for (int idim = 0; idim < NDIM; idim++) {
        for (int i = 0; i < ntotp; i++) {
            plot3dfile >> h_pts[NDIM * i + idim];
        }
    }

    plot3dfile.close();

    // Allocate memory on the device
    double * pts;
    cudaMalloc(&pts, ntotp * NDIM * sizeof(double));
    cudaMemcpy(pts, h_pts, ntotp * NDIM * sizeof(double), cudaMemcpyHostToDevice);

    double * area;
    cudaMalloc(&area, ntotp * 7 * sizeof(double));


    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_p((nxp + block.x - 1) / block.x, (nyp + block.y - 1) / block.y, 1);

    printf("Grid: %d %d Block: %d %d\n", grid_p.x, grid_p.y, block.x, block.y);
    compute_area<<<grid_p, block>>>(pts, area, nxp, nyp);

    // Quantities defined at cell centers
    int nx = nxp - 1;
    int ny = nyp - 1;
    int ntot = nx * ny;
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

    double * cell_center;
    cudaMalloc(&cell_center, ntot * NDIM * sizeof(double));
    compute_cellcenter<<<grid, block>>>(pts, cell_center, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();
    compute_if<<<grid, block>>>(pts, cell_center, area, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();    
    
    // Create a field phi and initialize it
    double * phi;
    cudaMalloc(&phi, ntot * sizeof(double));
    double * phi_bc_bot;
    double * phi_bc_top;
    cudaMalloc(&phi_bc_bot, nx * sizeof(double));
    cudaMalloc(&phi_bc_top, nx * sizeof(double));

    initialize_phi<<<grid, block>>>(pts, phi, phi_bc_bot, phi_bc_top, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();

    double * h_phi = new double [ntot];
    cudaMemcpy(h_phi, phi, ntot * sizeof(double), cudaMemcpyDeviceToHost);
    
    double * grad_phi_ref;
    cudaMalloc(&grad_phi_ref, ntot * NDIM * sizeof(double));
    double * lapl_phi_ref;
    cudaMalloc(&lapl_phi_ref, ntot * sizeof(double));
    reference_grad_phi<<<grid, block>>>(pts, grad_phi_ref, lapl_phi, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();
    double * h_grad_phi_ref = new double[ntot * NDIM];
    cudaMemcpy(h_grad_phi_ref, grad_phi_ref, ntot * NDIM * sizeof(double), cudaMemcpyDeviceToHost);
    double * h_lapl_phi_ref = new double[ntot];
    cudaMemcpy(h_lapl_phi_ref, lapl_phi_ref, ntot * sizeof(double), cudaMemcpyDeviceToHost);
    

    double * grad_phi;
    cudaMalloc(&grad_phi, ntot * NDIM * sizeof(double));
    // Calculate the gradient of phi
    vector_grad_gauss<<<grid, block>>>(phi, grad_phi, grad_phi_ref, area, phi_bc_bot, phi_bc_top, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();
 
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    double * h_grad_phi = new double[ntot * NDIM];
    cudaMemcpy(h_grad_phi, grad_phi, ntot * NDIM * sizeof(double), cudaMemcpyDeviceToHost);


    // Compute residual and Jacobian
    double * res;
    cudaMalloc(&res, ntot * sizeof(double));
    double * jac;
    cudaMalloc(&jac, ntot * 5 * sizeof(double));
    compute_r_j<<<grid, block>>>(phi, grad_phi, jac, res, area, phi_bc_bot, phi_bc_top, cell_center, pts, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();

    
    double * h_res = new double[ntot];
    cudaMemcpy(h_res, res, ntot * sizeof(double), cudaMemcpyDeviceToHost);
    

    // Write h_grad_phi to a file
    double zero = 0.0;
    std::ofstream grad_file("grad_phi.vtk");
    if (grad_file.is_open()) {
        grad_file << "# vtk DataFile Version 3.0" << std::endl;
        grad_file << "DU00W2121 Airfoil " << std::endl;
        grad_file << "ASCII" << std::endl;
        grad_file << "DATASET STRUCTURED_GRID " << std::endl;
        grad_file << "DIMENSIONS " << nxp << " " << nyp << " 1" << std::endl;
        grad_file << "POINTS " << nxp * nyp * 1 << " double" << std::endl;
        for (int i = 0; i < ntotp; ++i) 
            grad_file << std::fixed << std::setprecision(6) <<h_pts[i * NDIM] << " " << h_pts[i * NDIM + 1] << " " << zero << std::endl;
        grad_file << "CELL_DATA " << nx * ny << std::endl;
        grad_file << "SCALARS phi double 1" << std::endl;
        grad_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntot; ++i) 
            grad_file << h_phi[i] << std::endl;
        grad_file << "SCALARS grad_phi_x double 1" << std::endl;
        grad_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntot; ++i) 
            grad_file << h_grad_phi[i * NDIM] << std::endl;
        grad_file << "SCALARS grad_phi_y double 1" << std::endl;
        grad_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntot; ++i) 
            grad_file << h_grad_phi[i * NDIM + 1] << std::endl;
        grad_file << "SCALARS grad_phi_x_ref double 1" << std::endl;
        grad_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntot; ++i) 
            grad_file << h_grad_phi_ref[i * NDIM] << std::endl;
        grad_file << "SCALARS grad_phi_y_ref double 1" << std::endl;
        grad_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntot; ++i) 
            grad_file << h_grad_phi_ref[i * NDIM + 1] << std::endl;
        grad_file << "SCALARS residual double 1" << std::endl;
        grad_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntot; ++i)
            grad_file << h_res[i] << std::endl;
        grad_file << "SCALARS residual double 1" << std::endl;
        grad_file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntot; ++i)
            grad_file << h_lapl_phi_ref[i] << std::endl;
        grad_file.close();

        std::cout << "Gradient of phi written to grad_phi.vtk" << std::endl;
    } else {
        std::cerr << "Error: could not open grad_phi.vtk for writing" << std::endl;
    }

    delete [] h_pts;
    cudaFree(pts);
    cudaFree(area);
    cudaFree(phi);
    delete [] h_phi;
    
    // cudaFree(grad_phi);
    delete [] h_grad_phi;
    // cudaFree(grad_phi_ref);
    delete [] h_grad_phi_ref;
    
    return 0;
}