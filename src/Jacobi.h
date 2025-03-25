#ifndef JACOBI
#define JACOBI

namespace Jacobi {

    __global__ void jacobi_kernel(double * deltaT, double * deltaT1, double * J, double * R, int nx, int ny);

    class Jacobi: public LinearSolvers {

        public:

            // Constructor
            void Jacobi(int nx, int ny, double * J, double *T, double *deltaT, double *R);

            // Destructor
            ~Jacobi();

            // Solver
            __host__ void solve_step();


        private:
            double * deltaT1;

    };


}


#endif // JACOBI

