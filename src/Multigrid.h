#ifndef MULTIGRID_H
#define MULTIGRID_H

#include "LinearSolver.h"
#include <cuda_runtime.h>
#include <vector>
#include <thrust/device_vector.h>


namespace MultiGridNS {

    class MultiGrid: public LinearSolverNS::LinearSolver {

        public:

            // Constructor
            MultiGrid(int nx, int ny, double * J, double *T, double *deltaT, double *R, int nlevels_inp, std::string bottom_solver_inp);

            // Destructor
            ~MultiGrid();

            // Solver
            __host__ void solve_step() override;


        private:

            int nlevels;
            // The number of cells in each direction at each level
            std::vector<int> nxl;
            std::vector<int> nyl;
            
            std::vector<LinearSolverNS::LinearSolver *> smoothers;
            std::vector<double *> Jmg;
            std::vector<double *> deltaTmg;
            std::vector<double *> Rmg;
            std::vector<double *> Rlinmg;

            std::vector<dim3> grid_size_mg;
            std::vector<dim3> grid_size_1d_mg;
            
    };


}


#endif // MULTIGRID_H

