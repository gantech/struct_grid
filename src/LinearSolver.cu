#include "LinearSolver.h"
#include <cmath>
namespace LinearSolverNS {

    LinearSolver::LinearSolver(int nxinp, int nyinp, 
        double * Jinp, double *Tinp, double *deltaTinp, double *Rinp):
    nx(nxinp), ny(nyinp), J(Jinp), T(Tinp), deltaT(deltaTinp), R(Rinp) {

        grid_size = dim3(nx, ny);
        block_size = dim3(32, 32);

        grid_size_1d = dim3( std::ceil (nx * ny / 1024.0) );

    }

}