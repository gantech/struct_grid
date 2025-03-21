#define TILE_SIZE_ADI 2
#define NNX 8
#define NNY 24

// Kernel function for Thomas solves in the X direction - part of ADI
__global__ void adi_x(double *T, double *J, double *R, int nx, int ny) {

    //extern __shared__ double sharedMemory[];

    __shared__ double sharedMemory[5 * TILE_SIZE_ADI * NNX];
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("BlockIdx - %d, ThreadIdx - %d, Col is %d\n", blockIdx.x, threadIdx.x, col);

    double * a = sharedMemory + threadIdx.x * 5 * nx;
    double * b = a + nx;
    double * c = b + nx;
    double * d = c + nx;
    double * x = d + nx;

    if (col == 0) {
        for (int i=0; i < nx; i++) {
            int idx_r = (col * nx) + i;
            int idx_j = idx_r * 5;
    
            a[i] = J[idx_j + 1];
            b[i] = J[idx_j];
            c[i] = J[idx_j + 2];
            d[i] = R[idx_r] - T[idx_r+nx] * J[idx_j + 4];
        }
    } else if (col == (ny-1)) {
        for (int i=0; i < nx; i++) {
            int idx_r = (col * nx) + i;
            int idx_j = idx_r * 5;
    
            a[i] = J[idx_j + 1];
            b[i] = J[idx_j];
            c[i] = J[idx_j + 2];
            d[i] = R[idx_r] - T[idx_r-nx] * J[idx_j + 3];
        }
    } else if (col < ny) {
        for (int i=0; i < nx; i++) {
            int idx_r = (col * nx) + i;
            int idx_j = idx_r * 5;
    
            a[i] = J[idx_j + 1];
            b[i] = J[idx_j];
            c[i] = J[idx_j + 2];
            d[i] = R[idx_r] - T[idx_r-nx] * J[idx_j + 3] - T[idx_r+nx] * J[idx_j + 4];
        }        
    }

    __syncthreads();

    if (col < ny) {

        // Forward substitution
        for (int i=1; i < nx; i++) {
            double w = a[i] / b[i-1];
            b[i] = b[i] - w * c[i-1];
            d[i] = d[i] - w * d[i-1];
        }



        // Backward substitution
        x[nx-1] = d[nx-1] / b[nx-1];
        
        for (int i = nx-2; i > -1; i--) {
            x[i] = (d[i] - c[i] * x[i+1]) / b[i];
        }

        // if (col == 0) {
        //     for (int i=0; i < nx; i++) {
        //         printf("Row, Col is %d, %d - a %e, b %e, c %e, d %e, x %e\n", i, col, a[i], b[i], c[i], d[i], x[i]);
        //     }
        // }        

        // Update solution back T
        for (int i=0; i < nx; i++) {
            int idx_r = (col * nx) + i;
            T[idx_r] = x[i];
        }

    }

}


// Kernel function for Thomas solves in the Y direction - part of ADI
__global__ void adi_y(double *T, double *J, double *R, int nx, int ny) {

    //extern __shared__ double sharedMemory[];

    __shared__ double sharedMemory[5 * TILE_SIZE_ADI * NNY];
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("BlockIdx - %d, ThreadIdx - %d, Row is %d\n", blockIdx.x, threadIdx.x, row);

    double * a = sharedMemory + threadIdx.x * 5 * ny;
    double * b = a + ny;
    double * c = b + ny;
    double * d = c + ny;
    double * x = d + ny;

    if (row == 0) {
        for (int j=0; j < ny; j++) {
            int idx_r = (j * nx) + row;
            int idx_j = idx_r * 5;
    
            a[j] = J[idx_j + 3];
            b[j] = J[idx_j];
            c[j] = J[idx_j + 4];
            d[j] = R[idx_r] - T[idx_r+1] * J[idx_j + 2];
        }
    } else if (row == (nx-1)) {
        for (int j=0; j < ny; j++) {
            int idx_r = (j * nx) + row;
            int idx_j = idx_r * 5;
    
            a[j] = J[idx_j + 3];
            b[j] = J[idx_j];
            c[j] = J[idx_j + 4];
            d[j] = R[idx_r] - T[idx_r-1] * J[idx_j + 1];
        }
    } else if (row < nx) {
        for (int j=0; j < ny; j++) {
            int idx_r = (j * nx) + row;
            int idx_j = idx_r * 5;
    
            a[j] = J[idx_j + 3];
            b[j] = J[idx_j];
            c[j] = J[idx_j + 4];
            d[j] = R[idx_r] - T[idx_r-1] * J[idx_j + 1] - T[idx_r+1] * J[idx_j + 2];
        }
    }

    __syncthreads();

    if (row < nx) {

        // Forward substitution
        for (int j=1; j < ny; j++) {
            double w = a[j] / b[j-1];
            b[j] = b[j] - w * c[j-1];
            d[j] = d[j] - w * d[j-1];
        }

        // Backward substitution
        x[ny-1] = d[ny-1] / b[ny-1];
        
        for (int j = ny-2; j > -1; j--) {
            x[j] = (d[j] - c[j] * x[j+1]) / b[j];
        }

        // Update solution back T
        for (int j=0; j < ny; j++) {
            int idx_r = (j * nx) + row;
            T[idx_r] = x[j];
        }

    }

}
