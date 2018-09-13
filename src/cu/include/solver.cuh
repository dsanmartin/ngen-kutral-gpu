#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <cublas_v2.h>

void gpuBlasMmulBK(const double *A, const double *B, double *C, const int m, const int k, const int n);
void gpuBlasMmul(const double *A, const double *B, double *C, const int m, const int k, const int n);
//void solver(double *U, double *V, double *h_C, int rows_U, int cols_U, int rows_V, int cols_V);
void solver(double *h_U0, double *h_V1, double *h_V2, double *h_U, int Nx, int Ny, int T);
void RHS(double *U, double *V1, double *V2, int Nx, int Ny);

#endif