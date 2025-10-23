#ifndef BICGSTAB_H
#define BICGSTAB_H
#include "LinearSolver.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>


namespace BiCGStabNS {

    class BiCGStab: public LinearSolverNS::LinearSolver {

        public:

            // Constructor
            BiCGStab(int nx, int ny, double * J, double *deltaT, double *R);

            // Destructor
            ~BiCGStab();

            // Solver
            __host__ void solve_step(int nsteps) override;


        private:
            int ntot;
            double * pvec; // Search direction
            double * jpvec; // Matvec of J with search direction pvec
            double * resid0; // Initial residual
            double * svec; // Temporary vector for computations
            double * tvec; // Temporary vector for computations

            thrust::device_ptr<double> t_pvec;
            thrust::device_ptr<double> t_jpvec;
            thrust::device_ptr<double> t_resid;
            thrust::device_ptr<double> t_resid0;
            thrust::device_ptr<double> t_svec;
            thrust::device_ptr<double> t_tvec;

            int grid_size_1d;
            int block_size_1d = 1024;

    };


}


#endif // BICGSTAB_H