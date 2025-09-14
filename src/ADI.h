#ifndef ADIXY
#define ADIXY
#include "LinearSolver.h"
#include <cuda_runtime.h>

namespace ADINS {

    __global__ void adi_x(double *deltaT, double *J, double *R, int nx, int ny);

    __global__ void adi_y(double *deltaT, double *J, double *R, int nx, int ny);

    class ADI : public LinearSolverNS::LinearSolver {

        public:

            // Constructor
            ADI(int nx, int ny, double * J, double *deltaT, double *R);

            // Destructor
            ~ADI() {}

            // Solver
            __host__ void solve_step(int nsteps) override;

    };
}

#endif // ADI
