#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <cublas_v2.h>

void gpuBlasMmulBK(const double *A, const double *B, double *C, const int m, const int k, const int n);
void gpuBlasMmul(const double *A, const double *B, double *C, const int m, const int k, const int n);
void solver(double *h_U0, double *h_V1, double *h_V2, double *h_U, 
  int Nx, int Ny, int T, double dx, double dy, double dt, double kappa);
void RHS(double *Unew, double *Uold, double *V1, double *V2, double *Dx, double *Dy, double *Dxx, 
  double *Dyy, double *tmp, double kappa, double dt, int Nx, int Ny);

#endif