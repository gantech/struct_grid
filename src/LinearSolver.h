#ifndef LINEAR_SOLVER
#define LINEAR_SOLVER
#include <cuda_runtime.h>

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
        
        protected:
            int nx;
            int ny;
            double * J;
            double * T;
            double * deltaT;
            double * R;

            dim3 grid_size;
            dim3 block_size;
        
            dim3 grid_size_1d;
            dim3 block_size_1d = 1024;
        
            
    };

}


#endif // LINEAR_SOLVERS