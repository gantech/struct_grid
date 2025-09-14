#ifndef JACOBI
#define JACOBI
#include "LinearSolver.h"
#include <cuda_runtime.h>

namespace JacobiNS {

    __global__ void jacobi_kernel(double * deltaT, double * deltaT1, double * J, double * R, int nx, int ny);

    class Jacobi: public LinearSolverNS::LinearSolver {

        public:

            // Constructor
            Jacobi(int nx, int ny, double * J, double *deltaT, double *R);

            // Destructor
            ~Jacobi();

            // Solver
            __host__ void solve_step(int nsteps) override;


        private:
            double * deltaT1;

    };


}


#endif // JACOBI
