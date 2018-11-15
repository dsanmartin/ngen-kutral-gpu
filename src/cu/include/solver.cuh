#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <cublas_v2.h>

void gpuBlasMmulBK(const double *A, const double *B, double *C, const int m, const int k, const int n);
void gpuBlasMmul(const double *A, const double *B, double *C, const int m, const int k, const int n);
/*
void solver(double *h_U0, double *h_V1, double *h_V2, double *h_U, 
  int Nx, int Ny, int T, double dx, double dy, double dt, double kappa);
*/
void solver(double *h_U0, double *h_B0, double *h_V1, double *h_V2, double *h_U, double *h_B, int Nx, int Ny, int T, 
  double dx, double dy, double dt, double kappa, double epsilon, double upc, double q, double alpha);
void RHS(double *Unew, double *Uold, double *Bnew, double *Bold, double *V1, double *V2, double *Dx, double *Dy, 
  double *Dxx, double *Dyy, double kappa, double epsilon, double upc, double q, double alpha, 
  double dt, int Nx, int Ny);

void eulerMethod(double *d_U, double *d_B, double *d_V1, double *d_V2, 
  double *d_Dx, double *d_Dy, double *d_Dxx, double *d_Dyy, 
  double kappa, double epsilon, double upc, double q, double alpha, double dt, int Nx, int Ny, int T);

#endif