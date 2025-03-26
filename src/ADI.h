#ifndef ADI
#define ADI
#include "LinearSolver.h"
#include <cuda_runtime.h>

namespace ADINS {

    __global__ void adi_x(double *T, double )
    class ADI : public LinearSolvers {
        public:
            // Constructor
            ADI(int nx, int ny, double *J, double * T, double * deltaT, double *R);

            // Destructor
            ~ADI() {}

            // Solver
            __host__ void solve_step() override;

    };
}

#endif // ADI