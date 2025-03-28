#ifndef LINEAR_SOLVER
#define LINEAR_SOLVER
#include <cuda_runtime.h>
#define TILE_SIZE 32

namespace LinearSolverNS {

    class LinearSolver {
        public:
            // Constructor
            LinearSolver(int nxinp, int nyinp, 
                double * Jinp, double *Tinp, double *deltaTinp, double *Rinp);

            // Destructor
            ~LinearSolver() {}

            // Take 1 solver step
            __host__ virtual void solve_step() = 0;

            // Compute Matrix vector product J * v and store it into result
            __host__ virtual void matvec(double * v, double * result) final;

            // Compute the residual of the linear system of equations based on latest solution R - J * deltaT and store it into lin_resid. 
            // If lin_resid is same as R, it will be overwritten.
            __host__ virtual void linresid(double * lin_resid) final;
        
        protected:
            int nx;
            int ny;
            double * J;
            double * T;
            double * deltaT;
            double * R;

            dim3 grid_size;
            dim3 block_size(TILE_SIZE, TILE_SIZE, 1);
        
            dim3 grid_size_1d;
            dim3 block_size_1d = 1024;
        
            
    };

}


#endif // LINEAR_SOLVERS