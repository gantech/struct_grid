namespace LaplaceHeat {


// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_const(double *T, double val, int nx, int ny, double dx, double dy) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] = val ;
    
}

// Kernel function for initialization - No tiling or shared memory
__global__ void initialize_ref(double *T, int nx, int ny, double dx, double dy) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) {
        int row = idx % nx;
        int col = idx / nx;
        double y = (0.5 + col) * dy;
        double x = (0.5 + row) * dx;
        T[(col * nx) + row] = 300.0 + x*x + (y*y*y)/ 27.0;
    }
    
}
// Kernel function for update - No tiling or shared memory
__global__ void update(double *T, double *deltaT, int nx, int ny) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < (nx * ny)) 
        T[idx] += deltaT[idx];

}

// Kernel function for calculation of Jacobian and Residual - No tiling or shared memory
__global__ void compute_r_j(double *T, double *J, double *R, int nx, int ny, double dx, double dy, double kc) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < nx) && (col < ny)) {
        
        double y = (0.5 + col) * dy;
        double x = (0.5 + row) * dx;
        int idx_r = (col * nx) + row;
        int idx_j = idx_r * 5;

        double jij = -4.0;
        double jip1j = 1.0;
        double jim1j = 1.0;
        double jijp1 = 1.0;
        double jijm1 = 1.0;

        double tip1j = 0.0;
        double tim1j = 0.0;
        double tijp1 = 0.0;
        double tijm1 = 0.0;

        double radd = 0.0;
        if (row == 0) {
            jij -= 2.0;
            jip1j += 0.3333333333333333 ;
            jim1j -= 1.0;
            tip1j = T[idx_r + 1];
            double t_bc_left = 300.0 + (y*y*y/27.0);
            radd += kc * 8.0 * t_bc_left / 3.0 ;
        } else if (row == (nx - 1)) {
            jij -= 2.0;
            jim1j += 0.3333333333333333;
            jip1j -= 1.0;
            tim1j = T[idx_r - 1];
            double t_bc_right = 300.0 + 1.0 + (y*y*y/27.0);
            radd += kc * 8.0 * t_bc_right / 3.0;
        } else {
            tip1j = T[idx_r + 1];
            tim1j = T[idx_r - 1];
        }

        if (col == 0) {
            jij -= 2.0;
            jijp1 += 0.3333333333333333;
            jijm1 -= 1.0;
            tijp1 = T[idx_r + nx];
            double t_bc_bot = 300.0 + (x*x);
            radd += kc * 8.0 * t_bc_bot / 3.0;
        } else if (col == (ny - 1)) {
            jij -= 2.0;
            jijm1 += 0.3333333333333333;
            jijp1 -= 1.0;
            tijm1 = T[idx_r - nx];
            double t_bc_top = 300.0 + 1.0 + (x*x);
            radd += kc * 8.0 * t_bc_top / 3.0;
        } else {
            tijm1 = T[idx_r - nx];
            tijp1 = T[idx_r + nx];
        }

        // Write to residual
        double tmp = kc * ( jijm1 * tijm1 + jijp1 * tijp1 + jim1j * tim1j + jip1j * tip1j + jij * T[idx_r] - (2.0 + 2.0 * y / 9.0) * dx * dy) + radd;

        // if (std::abs(tmp/(dx * dy * kc)) > 20.0) {
        //     printf("Row, Col is %d, %d - x,y = %f, %f, Residuals - %f, %f, J - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f, T - (j-1) %f, (j+1) %f, (i-1) %f, (i+1) %f, (ij) %f \n", row, col, x, y, 2.0 - 2.0 * y / 9.0, tmp / (dx * dy * kc), jijm1, jijp1, jim1j, jip1j, jij, tijm1, tijp1, tim1j, tip1j, T[idx_r]);
        // }

        R[idx_r] = -tmp;

        // Write to the Jacobian
        J[idx_j] = jij * kc; //i,j
        J[idx_j + 1] = jim1j * kc; //i-1,j
        J[idx_j + 2] = jip1j * kc; //i+1,j
        J[idx_j + 3] = jijm1 * kc; //i,j-1
        J[idx_j + 4] = jijp1 * kc; //i,j+1
    }
}

// Kernel function for calculation of Residual - No tiling or shared memory
__global__ void compute_r(double *T, double * J, double *R, int nx, int ny, double dx, double dy, double kc) {

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if ((row < nx) && (col < ny)) {
        
        double y = (0.5 + col) * dy;
        double x = (0.5 + row) * dx;
        int idx_r = (col * nx) + row;
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

        double radd = 0.0;
        if (row == 0) {
            tip1j = T[idx_r + 1];
            double t_bc_left = 300.0 + (y*y*y/27.0);
            radd += kc * 8.0 * t_bc_left / 3.0 ;
        } else if (row == (nx - 1)) {
            tim1j = T[idx_r - 1];
            double t_bc_right = 300.0 + 1.0 + (y*y*y/27.0);
            radd += kc * 8.0 * t_bc_right / 3.0;
        } else {
            tip1j = T[idx_r + 1];
            tim1j = T[idx_r - 1];
        }

        if (col == 0) {
            tijp1 = T[idx_r + nx];
            double t_bc_bot = 300.0 + (x*x);
            radd += kc * 8.0 * t_bc_bot / 3.0;
        } else if (col == (ny - 1)) {            
            tijm1 = T[idx_r - nx];
            double t_bc_top = 300.0 + 1.0 + (x*x);
            radd += kc * 8.0 * t_bc_top / 3.0;
        } else {
            tijm1 = T[idx_r - nx];
            tijp1 = T[idx_r + nx];
        }

        // Write to residual
        R[idx_r] = -kc * ( jijm1 * tijm1 + jijp1 * tijp1 + jim1j * tim1j + jip1j * tip1j + jij * T[idx_r] - (2.0 + 2.0 * y / 9.0) * dx * dy) - radd;
    }
}


// Kernel to compute matrix vector product of the linear system of equations J * v . 
__global__ void compute_matvec(double * v, double * J, double * result, int nx, int ny) {

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

        double vip1j = 0.0;
        double vim1j = 0.0;
        double vijp1 = 0.0;
        double vijm1 = 0.0;

        if ( i == 0) {
            vip1j = v[idx_r + 1];
        } else if ( i == (nx - 1)) {
            vim1j = v[idx_r - 1];
        } else {
            vip1j = v[idx_r + 1];
            vim1j = v[idx_r - 1];
        }

        if ( j == 0) {
            vijp1 = v[idx_r + nx];
        } else if ( j == (ny - 1)) {
            vijm1 = v[idx_r - nx];
        } else {
            vijm1 = v[idx_r - nx];
            vijp1 = v[idx_r + nx];
        }

        result[idx_r] = jim1j * tim1j + jip1j * tip1j + jijm1 * tijm1 + jijp1 * tijp1 + jij * v[idx_r];
    }
}




}