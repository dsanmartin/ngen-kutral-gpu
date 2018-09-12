#ifndef SOLVER_CUH
#define SOLVER_CUH

#include <cublas_v2.h>

void gpuBlasMmul(const float *A, const float *B, float *C, const int m, const int k, const int n);
void solver(float *U, float *V, float *h_C, int rows_U, int cols_U, int rows_V, int cols_V);

#endif