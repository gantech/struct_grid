#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>

#define TILE_SIZE 16
#define TILE_SIZE_ADI 2
#define NDIM 2


// Kernel function for area calculation - Not cached yet
__global__ void area_kernel(double * pts, double * area, int nxp, int nyp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    //printf("BlockIdx.x: %d BlockIdx.y: %d ThreadIdx.x: %d ThreadIdx.y: %d, i: %d j: %d \n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, i, j);
     int idx_p = (j * nxp + i) * NDIM;
     int idx_a = (j * nxp + i) * 7;

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

         } else {
             area[idx_a] = 0.0;
             area[idx_a + 1] = 0.0;
         }

         if ( j < (nyp -1) ) {
             double xu = pts[idx_p + nxp * NDIM];
             double yu = pts[idx_p + nxp * NDIM + 1];

             // Rightward normal for (j to j+1 face)
             area[idx_a + 2] = (yu - y); // = alpha_{xi_x}
             area[idx_a + 3] = -(xu - x); // = alpha_{xi_y}
            //  printf(" Area - i-i+1 = %e, %e", -(yu -y), (xu - x) );

         } else {
             area[idx_a + 2] = 0.0;
             area[idx_a + 3] = 0.0;
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


// Kernel function for interpolating factor calculation - Not cached yet
/*

    Method to find intersection point between two lines. 

    Line 1 from p1 to p2. Line 2 from p3 to p4
    Any point from p1 to p2 = p1 + t1 * (p2 - p1)
    Any point from p3 to p4 = p3 + t2 * (p4 - p3)
    We want these points to be the same. Hence, we solve equation system formed by equating these two coordinates in x and y.
    p1 + t1 * (p2 - p1) = p3 + t2 * (p4 - p3)
    
    (p2.x - p1.x) * t1 + (p3.x - p4.x) * t2 = p3.x - p1.x
    (p2.y - p1.y) * t1 + (p3.y - p4.y) * t2 = p3.y - p1.y

    detA = (p2.x - p1.x) * (p3.y - p4.y) - (p3.x - p4.x) * (p2.y - p1.y)

    t1 = ((p3.y - p4.y) * (p3.x - p1.x) - (p3.x - p4.x) * (p3.y - p1.y)) / detA
    t2 = (-(p2.y - p1.y) * (p3.x - p1.x) + (p2.x - p1.x) * (p3.y - p1.y)) / detA

)
*/
__global__ void compute_if(double * pts, double * cell_center, double * area, int nx, int ny, int nxp, int nyp) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx_p = (j * nxp + i) * NDIM;
    int idx_a = (j * nxp + i) * 7;
    int idx_cc = (j * nx + i%nx) * NDIM;

    if ( (i < nxp) && (j < ny)) { // Make sure you're within the grid
     
    
        //p3
        double x = pts[idx_p];
        double y = pts[idx_p + 1];        
        //p4
        double xu = pts[idx_p + nxp * NDIM] ;
        double yu = pts[idx_p + nxp * NDIM + 1];

        //p1
        double xim1 = 0.0;
        double yim1 = 0.0;
        //p2
        double xi = 0.0;
        double yi = 0.0;
        //p1
        if (i == 0) {
            xim1 = cell_center[idx_cc + (nx-1)*NDIM];
            yim1 = cell_center[idx_cc + (nx-1)*NDIM + 1];
            xi = cell_center[idx_cc];
            yi = cell_center[idx_cc+1];           
        } else if (i == (nxp-1)) {
            xim1 = cell_center[j * nx + (nx-1)*NDIM];
            yim1 = cell_center[j * nx + (nx-1)*NDIM+1];
            xi = cell_center[j * nx];
            yi = cell_center[j * nx + 1];
        } else {
            xim1 = cell_center[idx_cc-NDIM];
            yim1 = cell_center[idx_cc-NDIM+1];
            xi = cell_center[idx_cc];
            yi = cell_center[idx_cc+1];
        }

        double detA = (xi-xim1) * (y-yu) - (x-xu) * (yi-yim1);
        double t1 = ( (y-yu) * (x-xim1) - (x-xu) * (y-yim1) ) / detA ;

        if ( std::isinf(t1) || std::isnan(t1)) {
            printf("i %d, j %d - t1 = %e, x, y = (%e, %e), xu, yu = (%e, %e), xim1, yim1 = (%e, %e), xi, yi = (%e, %e) \n", i , j, t1, x, y, xu, yu, xim1, yim1, xi, yi );
        }
        area[idx_a + 5] = t1;

    if ( (j > 0) && (j < ny) ) {
        //p3
        double x = pts[idx_p];
        double y = pts[idx_p + 1];        
        //p4
        double xr = pts[idx_p + NDIM];
        double yr = pts[idx_p + NDIM + 1];

        //p1
        double xjm1 = cell_center[idx_cc-nx*NDIM];
        double yjm1 = cell_center[idx_cc-nx*NDIM+1];        
        //p2
        double xj = cell_center[idx_cc];
        double yj = cell_center[idx_cc+1];


    /*
    detA = (p2.x - p1.x) * (p3.y - p4.y) - (p3.x - p4.x) * (p2.y - p1.y)

    t1 = ((p3.y - p4.y) * (p3.x - p1.x) - (p3.x - p4.x) * (p3.y - p1.y)) / detA
    */        
        double detA = (xj - xjm1) * (y - yr) - (x - xr) * (yj - yjm1) ;
        double t1 = ( (y - yr) * (x - xjm1) - (x - xr) * (y - yjm1) ) / detA ;

        if (std::isinf(t1) || std::isnan(t1)) {
            printf( "i %d, j %d - t1 = %e, x, y = (%e, %e), xr, yr = (%e, %e), xjm1, yjm1 = (%e, %e), xj, yj = (%e, %e) \n", i , j, t1, x, y, xr, yr, xjm1, yjm1, xj, yj);
        }

        area[idx_a + 6] = t1;
    } else {
        area[idx_a + 6] = 0.0;
    }

    // if (j > 90) 
    //     printf("i: %d j: %d, if_y: %e, if_x: %e \n", i, j, area[idx_a + 5], area[idx_a + 6]);
    
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

        if ( (tmp * grad_phi_ref[idx_gp]) < 0.0 )
            printf("i %d, j %d, grad_phi_x = %e, grad_phi_ref_x %e, phi_x_ew = %e, phi_x_ns = %e, IF: (S) %e, (W) %e, (N) %e, (E) %e, Total %e, Phi (I-1) %e, (I), %e, (I+1) %e, (J-1) %e, (J+1) %e \n", i, j, tmp, grad_phi_ref[idx_gp], (phi_xix_e - phi_xix_w)/area[idx_a+4], (phi_etax_n - phi_etax_s)/area[idx_a+4], area[idx_a+5], area[idx_a+6], area[idx_a+nxp*7+5], area[idx_a+7+5], area[idx_a+4], phiim1j, phiij, phiip1j, phiijm1, phiijp1);
        // if (i == 0)
        //     printf("i %d, j %d, grad_phi_x = %e, phi_xiy_e = %e, phi_xiy_w = %e, phi_etay_n = %e, phi_etay_s = %e, Areas: (S) %e, %e, (N) %e, %e, (E) %e, %e, (W) %e, %e, Phi (I-1) %e, (I), %e, (I+1) %e, (J-1) %e, (J+1) %e\n", i, j, tmp1, phi_xiy_e, phi_xiy_w, phi_etay_n, phi_etay_s, area[idx_a], area[idx_a+1], area[idx_a+nxp*7], area[idx_a+nxp*7+1], area[idx_a+7+2], area[idx_a+7+3], area[idx_a+2], area[idx_a+3], phiim1j, phiij, phiip1j, phiijm1, phiijp1);
    }
}
