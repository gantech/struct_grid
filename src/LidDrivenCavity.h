#ifndef LID_DRIVEN_CAVITY
#define LID_DRIVEN_CAVITY

#include <string>
#include "LinearSolver.h"
#include "CG.h"
#include "BiCGStab.h"
#include <thrust/device_vector.h>

namespace LidDrivenCavityNS {

class LidDrivenCavity {
public:

    // Constructor
    LidDrivenCavity(int nx, int ny, double nu_inp);

    // Destructor
    ~LidDrivenCavity();

    __host__ void compute_gradp();

    __host__ void apply_bc();

    __host__ void update_mom(double alpha_mom = 1.0);

    __host__ void update_pres(double alpha_p = 1.0);

    __host__ double compute_mom_r_j();

    __host__ double compute_cont_r_j();

    __host__ void solve_mom(int niters);

    __host__ void solve_cont(int niters);

    __host__ void set_ainv();

    int nx;
    int ny;
    double dx;
    double dy;
    double nu;
    double dt;

    double * umom;
    double * vmom;
    double * pres;
    double * gpx;
    double * gpy;
    double * a_inv;
    double * u_nlr;
    double * v_nlr;
    double * cont_nlr;
    double * deltaU;
    double * deltaV;
    double * deltaP;
    double * Jmom;
    double * Jcont;

    thrust::device_ptr<double> t_unlr;
    thrust::device_ptr<double> t_vnlr;
    thrust::device_ptr<double> t_cont_nlr;

    std::unique_ptr<LinearSolverNS::LinearSolver> solver_u;
    std::unique_ptr<LinearSolverNS::LinearSolver> solver_v;
    std::unique_ptr<LinearSolverNS::LinearSolver> solver_p;

private:

    dim3 grid_size;
    dim3 block_size;

    dim3 grid_size_1d;
    dim3 block_size_1d = 1024;

};

}

#endif // LID_DRIVEN_CAVITY
