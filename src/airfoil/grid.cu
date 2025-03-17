#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <iomanip>

#define TILE_SIZE 16
#define TILE_SIZE_ADI 2
#define NDIM 2


// Kernel function for area calculation - Not cached yet
__global__ void compute_area(double * pts, double * area, int nxp, int nyp) {
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
