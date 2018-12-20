#include <stdio.h>
#include "include/solver.cuh"
#include "include/diffmat.cuh"
#include "../c/include/files.h"
#include "../c/include/utils.h"

__device__ double gaussian(double A, double sigma_x, double sigma_y, double x, double y) {
  return A * exp((x * x) / sigma_x + (y * y) / sigma_y);
}
__device__ double u0(double x, double y) {
  return gaussian(6, -5e2, -5e2, x, y);
  //return 6 * exp(-5e-2 * ((x - 20) ** 2 + (y - 20) ** 2));
}

__device__ double b0(double x, double y) {
  return (double) rand() / (double)RAND_MAX ;
}

__device__ double v1(double x, double y) {
  return 3.0;
}

__device__ double v2(double x, double y) {
  return 3.0;
}

__global__ void U0(double *U, double *x, double *y, int N, int M) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < N * M) {
    int i = tId % M; // Row index
    int j = tId / M; // Col index
    U[j * M + i] = u0(x[j], y[i]);
  }
}

//void initialCondition(double *U, double )

void solver(Parameters parameters) {
  /* Domain differentials */
	double dx = (parameters.x_max - parameters.x_min) / parameters.N;
	double dy = (parameters.y_max - parameters.y_min) / parameters.M;
  double dt = parameters.t_max / parameters.L;
  
  double *x = (double *) malloc(parameters.N * sizeof(double));
  fillVector(x, dx, parameters.N);
  printMatrix(x, 1, parameters.N);
}

