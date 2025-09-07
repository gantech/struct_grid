#ifndef CG_H
#define CG_H
#include "LinearSolver.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>


namespace CGNS {

    __global__ void _kernel(double * deltaT, double * deltaT1, double * J, double * R, int nx, int ny);

    class CG: public LinearSolverNS::LinearSolver {

        public:

            // Constructor
            CG(int nx, int ny, double * J, double *T, double *deltaT, double *R);

            // Destructor
            ~CG();

            // Solver
            __host__ void solve_step(int nsteps) override;


        private:
            int ntot;
            double * pvec; // Search direction
            double * jpvec; // Matvec of J with search direction pvec

            thrust::device_ptr<double> t_pvec;
            thrust::device_ptr<double> t_jpvec;
            thrust::device_ptr<double> t_resid;

            dim3 grid_size_1d;
            dim3 block_size_1d = 1024;

    };


}


#endif // JACOBI
