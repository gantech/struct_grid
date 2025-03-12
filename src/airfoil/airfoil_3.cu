#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>

#define TILE_SIZE 16
#define TILE_SIZE_ADI 2
#define NDIM 2


// Kernel function for area calculation - Not cached yet
__global__ void compute_area(double * pts, double * cell_center, double * area, int nxp, int nyp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int nx = nxp - 1;
    int ny = nyp - 1;

    //printf("BlockIdx.x: %d BlockIdx.y: %d ThreadIdx.x: %d ThreadIdx.y: %d, i: %d j: %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j);
     int idx_p = (j * nxp + i) * NDIM;
     int idx_a = (j * nxp + i) * 9;
     int idx_cc = (j * nx + i % nx) * NDIM;

     if ( (i < nxp) && (j < nyp)) { // Make sure you are within the grid
         double x = pts[idx_p];
         double y = pts[idx_p + 1];        

         if ( i < (nxp-1) ) { // Assuming periodic in i direction
            double xr = pts[idx_p + NDIM];
            double yr = pts[idx_p + NDIM + 1];

            // Upward normal for (i to i+1 face)
            area[idx_a] = - (yr - y); // = alpha_{eta_x}
            area[idx_a + 1] = xr - x; // = alpha_{eta_y}
            //  printf(" Area - i-i+1 = %e, %e", -(yr -y), (xr - x) );
            
            if ( (j > 0) && ( i < (nxp - 2)) ) {
                double xf = 0.5 * (x + xr);
                double yf = 0.5 * (y + yr);
                double x_midp_cc = 0.5 * (cell_center[idx_cc]+cell_center[idx_cc-nx*NDIM]) ;
                double y_midp_cc = 0.5 * (cell_center[idx_cc+1]+cell_center[idx_cc-nx*NDIM+1]) ;

                area[idx_a+5] = xf - x_midp_cc;
                area[idx_a+6] = yf - y_midp_cc;
            } else {
                area[idx_a+5] = 0.0;
                area[idx_a+6] = 0.0;
            }

         } else {

             area[idx_a] = 0.0;
             area[idx_a + 1] = 0.0;
             area[idx_a + 5] = 0.0;
             area[idx_a + 6] = 0.0;         

         }

         if ( j < (nyp -1) ) {
            double xu = pts[idx_p + nxp * NDIM];
            double yu = pts[idx_p + nxp * NDIM + 1];

            // Rightward normal for (j to j+1 face)
            area[idx_a + 2] = (yu - y); // = alpha_{xi_x}
            area[idx_a + 3] = -(xu - x); // = alpha_{xi_y}
            //  printf(" Area - i-i+1 = %e, %e", -(yu -y), (xu - x) );

            int idx_cc_im1 = idx_cc - NDIM;
            if (i == 0)
                idx_cc_im1 = idx_cc + (nx-1) * NDIM;

            double xf = 0.5 * (x + xu);
            double yf = 0.5 * (y + yu);

            double x_midp_cc = 0.5 * (cell_center[idx_cc]+cell_center[idx_cc_im1]) ;
            double y_midp_cc = 0.5 * (cell_center[idx_cc+1]+cell_center[idx_cc_im1+1]) ;

            area[idx_a+7] = xf - x_midp_cc;
            area[idx_a+8] = yf - y_midp_cc;

         } else {

            area[idx_a + 2] = 0.0;
            area[idx_a + 3] = 0.0;
            area[idx_a + 7] = 0.0;
            area[idx_a + 8] = 0.0;

        }

         if ( (i < (nxp-1)) && (j < (nyp -1)) ) {
            
            double xr = pts[idx_p + NDIM];
            double yr = pts[idx_p + NDIM + 1];
            double xu = pts[idx_p + nxp * NDIM];
            double yu = pts[idx_p + nxp * NDIM + 1];
            double xur = pts[idx_p + nxp * NDIM + NDIM];
            double yur = pts[idx_p + nxp * NDIM + NDIM + 1];
    
            // ad = (xur - x), (yur - y)
            // bc = (xu - xr), (yu - yr)
            // Cross product - ad x bc
            area[idx_a + 4] = 0.5 * ( (xur - x) * (yu - yr) -  (xu - xr) * (yur - y) );
         }
     }
}


// Kernel to calculate cell centers
__global__ void compute_cellcenter(double * pts, double * cell_center, int nx, int ny, int nxp, int nyp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int idx_p = ( (j * nxp) + i ) * NDIM;
    int idx_c = ( (j * nx) + i ) * NDIM ;

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

        cell_center[idx_c] = x;
        cell_center[idx_c + 1] = y;
    }

}

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

// Kernel to initialize gradphi to zero
__global__ void initialize_gradphi(double * grad_phi, int nx, int ny) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int idx_gp = ( (j * nx) + i ) * NDIM ;

    if ( (i < nx) && (j < ny)) { // Make sure you're within the grid

        grad_phi[idx_gp] = 0.0;
        grad_phi[idx_gp+1] = 0.0;

    }

}

// Kernel to compute gradient of the reference field phi as 2x iHat + 3y^2 jHat
__global__ void reference_grad_phi(double * pts, double * grad_phi_ref, int nx, int ny, int nxp, int nyp) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    int idx_p = ( (j * nxp) + i ) * NDIM;
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
    int idx_gp_ip1 = (j * nx + (i+1)%nx ) * NDIM;
    int idx_gp_im1 = (j * nx + (i-1) ) * NDIM;
    if (i == 0)
        idx_gp_im1 = (j * nx + nx-1 ) * NDIM;

    int idx_a = (j * nxp + i) * 9;



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
        double gpij_x = grad_phi[idx_gp];
        double gpij_y = grad_phi[idx_gp + 1];

        double phiijp1 = 0.0;
        double phiijm1 = 0.0;

        if (j < (ny -1 ))
            phiijp1 = phi[idx_phi + nx];
        if (j > 0)
            phiijm1 = phi[idx_phi - nx];

        if ( j == 0) {

            double phiijp1 = phi[idx_phi + nx];
            double gpijp1_x = grad_phi[idx_gp + nx * NDIM];
            double gpijp1_y = grad_phi[idx_gp + nx * NDIM + 1];
            phi_etax_s = phi_bc_bot[i]  * area[idx_a];
            phi_etax_n = (0.5 * (phiijp1 + phiij) + 0.5 * ( (gpijp1_x + gpij_x) * area[idx_a + 5] + (gpijp1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a + nxp * 9];
            phi_etay_s = phi_bc_bot[i]  * area[idx_a + 1];
            phi_etay_n = (0.5 * (phiijp1 + phiij) + 0.5 * ( (gpijp1_x + gpij_x) * area[idx_a + 5] + (gpijp1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a + nxp * 9 + 1];         

        } else if (j == (ny - 1)) {

            double phiijm1 = phi[idx_phi - nx];
            double gpijm1_x = grad_phi[idx_gp - nx * NDIM];
            double gpijm1_y = grad_phi[idx_gp - nx * NDIM + 1];
            phi_etax_s = ( 0.5 * (phiijm1 + phiij) + 0.5 * ( (gpijm1_x + gpij_x) * area[idx_a + 5] + (gpijm1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a];
            phi_etax_n = phi_bc_top[i]  * area[idx_a + nxp * 9];
            phi_etay_s = ( 0.5 * (phiijm1 + phiij) + 0.5 * ( (gpijm1_x + gpij_x) * area[idx_a + 5] + (gpijm1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a + 1];
            phi_etay_n = phi_bc_top[i]  * area[idx_a + nxp * 9 + 1];         

        } else {

            double phiijm1 = phi[idx_phi - nx];
            double phiijp1 = phi[idx_phi + nx];
            double gpijm1_x = grad_phi[idx_gp - nx * NDIM];
            double gpijm1_y = grad_phi[idx_gp - nx * NDIM + 1];
            double gpijp1_x = grad_phi[idx_gp + nx * NDIM];
            double gpijp1_y = grad_phi[idx_gp + nx * NDIM + 1];

            phi_etax_s = ( 0.5 * (phiijm1 + phiij) + 0.5 * ( (gpijm1_x + gpij_x) * area[idx_a + 5] + (gpijm1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a];
            phi_etax_n = (0.5 * (phiijp1 + phiij) + 0.5 * ( (gpijp1_x + gpij_x) * area[idx_a + 5] + (gpijp1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a + nxp * 9];
            phi_etay_s = ( 0.5 * (phiijm1 + phiij) + 0.5 * ( (gpijm1_x + gpij_x) * area[idx_a + 5] + (gpijm1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a + 1];
            phi_etay_n = (0.5 * (phiijp1 + phiij) + 0.5 * ( (gpijp1_x + gpij_x) * area[idx_a + 5] + (gpijp1_y + gpij_y) * area[idx_a + 6] ) ) * area[idx_a + nxp * 9 + 1];         

        }
    
        double phiip1j = phi[idx_phi_ip1];
        double phiim1j = phi[idx_phi_im1];
        double gpip1j_x = grad_phi[idx_gp_ip1];
        double gpip1j_y = grad_phi[idx_gp_ip1 + 1];
        double gpim1j_x = grad_phi[idx_gp_im1];
        double gpim1j_y = grad_phi[idx_gp_im1 + 1];

        phi_xix_w = ( 0.5 * ( phiim1j + phiij) + 0.5 * ( (gpim1j_x + gpij_x) * area[idx_a + 7] + (gpim1j_y + gpij_y) * area[idx_a + 8] ) ) * area[idx_a + 2];
        phi_xix_e = ( 0.5 * ( phiip1j + phiij) + 0.5 * ( (gpip1j_x + gpij_x) * area[idx_a + 7] + (gpip1j_y + gpij_y) * area[idx_a + 8] ) ) * area[idx_a + 9 + 2];
        phi_xiy_w = ( 0.5 * ( phiim1j + phiij) + 0.5 * ( (gpim1j_x + gpij_x) * area[idx_a + 7] + (gpim1j_y + gpij_y) * area[idx_a + 8] ) ) * area[idx_a + 3];
        phi_xiy_e = ( 0.5 * ( phiip1j + phiij) + 0.5 * ( (gpip1j_x + gpij_x) * area[idx_a + 7] + (gpip1j_y + gpij_y) * area[idx_a + 8] ) ) * area[idx_a + 9 + 3];
        
        // printf("i %d j %d - %d \n", i, j, idx_gp);
        double tmp = (phi_xix_e - phi_xix_w + phi_etax_n - phi_etax_s)/area[idx_a + 4];
        double tmp1 = (phi_xiy_e - phi_xiy_w + phi_etay_n - phi_etay_s)/area[idx_a + 4];
        grad_phi[idx_gp] = tmp;
        grad_phi[idx_gp + 1] = tmp1;

        if ( (tmp * grad_phi_ref[idx_gp]) < 0.0 )
            printf("i %d, j %d, grad_phi_x = %e, grad_phi_ref_x %e, phi_x_ew = %e, phi_x_ns = %e, Area Total %e, Phi (I-1) %e, (I), %e, (I+1) %e, (J-1) %e, (J+1) %e \n", i, j, tmp, grad_phi_ref[idx_gp], (phi_xix_e - phi_xix_w)/area[idx_a+4], (phi_etax_n - phi_etax_s)/area[idx_a+4], area[idx_a+4], phiim1j, phiij, phiip1j, phiijm1, phiijp1);
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
    cudaMalloc(&area, ntotp * 9 * sizeof(double));


    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid_p((nxp + block.x - 1) / block.x, (nyp + block.y - 1) / block.y, 1);

    printf("Grid: %d %d Block: %d %d\n", grid_p.x, grid_p.y, block.x, block.y);

    
    // Quantities defined at cell centers
    int nx = nxp - 1;
    int ny = nyp - 1;
    int ntot = nx * ny;
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);
    
    // Create a field phi and initialize it
    double * phi;
    cudaMalloc(&phi, ntot * sizeof(double));
    double * phi_bc_bot;
    double * phi_bc_top;
    cudaMalloc(&phi_bc_bot, nx * sizeof(double));
    cudaMalloc(&phi_bc_top, nx * sizeof(double));

    double * cell_center;
    cudaMalloc(&cell_center, ntot * NDIM * sizeof(double));
    compute_cellcenter<<<grid, block>>>(pts, cell_center, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();

    compute_area<<<grid_p, block>>>(pts, cell_center, area, nxp, nyp);
    cudaDeviceSynchronize();

    initialize_phi<<<grid, block>>>(pts, phi, phi_bc_bot, phi_bc_top, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();


    double * h_phi = new double [ntot];
    cudaMemcpy(h_phi, phi, ntot * sizeof(double), cudaMemcpyDeviceToHost);
    
    double * grad_phi_ref;
    cudaMalloc(&grad_phi_ref, ntot * NDIM * sizeof(double));
    reference_grad_phi<<<grid, block>>>(pts, grad_phi_ref, nx, ny, nxp, nyp);
    cudaDeviceSynchronize();
    double * h_grad_phi_ref = new double[ntot * NDIM];
    cudaMemcpy(h_grad_phi_ref, grad_phi_ref, ntot * NDIM * sizeof(double), cudaMemcpyDeviceToHost);

    double * grad_phi;
    cudaMalloc(&grad_phi, ntot * NDIM * sizeof(double));
    initialize_gradphi<<<grid, block>>>(grad_phi, nx, ny);
    cudaDeviceSynchronize();
    // Calculate the gradient of phi
    for (int igrad=0; igrad < 10; igrad++) {
        std::cout << "Calling gradphi - " << igrad << std::endl;
        vector_grad_gauss<<<grid, block>>>(phi, grad_phi, grad_phi_ref, area, phi_bc_bot, phi_bc_top, nx, ny, nxp, nyp);
        cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();
 
    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;

    double * h_grad_phi = new double[ntot * NDIM];
    cudaMemcpy(h_grad_phi, grad_phi, ntot * NDIM * sizeof(double), cudaMemcpyDeviceToHost);

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