#ifndef LINEAR_SOLVER
#define LINEAR_SOLVER
#include <cuda_runtime.h>

namespace LinearSolver {

    class LinearSolver {
        public:
            // Constructor
            LinearSolver(int nxinp, int nyinp, 
                double * Jinp, double *Tinp, double *deltaTinp, double *Rinp);

            // Destructor
            ~LinearSolver();

            // Take 1 solver step
            virtual void solve_step() __host__ = 0;
        
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