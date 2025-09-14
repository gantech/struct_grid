#ifndef LAPLACE_HEAT
#define LAPLACE_HEAT

#include <string>
#include "LinearSolver.h"
#include "Jacobi.h"
#include "ADI.h"
#include "Multigrid.h"
#include "CG.h"
#include <thrust/device_vector.h>

namespace LaplaceHeatNS {

class LaplaceHeat {
public:

    // Constructor
    LaplaceHeat(int nx, int ny, double kc, std::string solver_type);

    // Destructor
    ~LaplaceHeat();

    __host__ void initialize_const(double * arr, double val);

    __host__ void initialize_ref();

    __host__ void update();

    __host__ double compute_r_j();

    __host__ void solve(int nsteps);

    int nx;
    int ny;
    double dx;
    double dy;
    double kc;

    double * T;
    double * nlr;
    double * deltaT;
    double * J;
    thrust::device_ptr<double> t_nlr;

    LinearSolverNS::LinearSolver * solver;

private:

    dim3 grid_size;
    dim3 block_size;

    dim3 grid_size_1d;
    dim3 block_size_1d = 1024;

};

}

#endif // LAPLACE_HEAT
