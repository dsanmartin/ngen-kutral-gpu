#include <stdlib.h>
#include "include/diffmat.cuh"

__global__ void FD1Kernel(double *D1N, int M, double h) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < M * M) {
    int i = tId % M; // Row index
    int j = tId / M; // Col index
    if (i - j == -1) 
      D1N[j * M + i] = 1 / (2 * h);
    else if (i - j == 1)
      D1N[j * M + i] = -1 / (2 * h);

    if (i == 0)
      D1N[M * (M - 1)] = -1 / (2 * h);
    if (i == M - 1)
      D1N[M - 1] = 1 / (2 * h);
  }
}

__global__ void FD2Kernel(double *D2N, int M, double h) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < M * M) {
    int i = tId % M; // Row index
    int j = tId / M; // Col index
    if (i == j) {
      D2N[j * M + i] = -2 / (h * h);
    } else if (abs(i - j) == 1) {
      D2N[j * M + i] = 1 / (h * h);
    }

    if (i == 0)
      D2N[M * (M - 1)] = 1 / (h * h);
    if (i == M - 1)
      D2N[M - 1] = 1 / (h * h);
  }
}

__global__ void ChebyshevNodes(double *x_c, int N) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < N + 1) {
    x_c[tId] = cos(tId * M_PI / N);
  }
}

__global__ void ChebyshevMatrix(double *CDM, double *x_c, int N) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < (N + 1) * (N + 1)) {
    int i = tId % (N + 1); // Row index
    int j = tId / (N + 1); // Col index
    if (i == 0 && j == 0) {
      CDM[j * (N + 1) + i] = (2.0 * N * N + 1.0) / 6.0;
    } else if (i == N && j == N) {
      CDM[j * (N + 1) + i] = - (2.0 * N * N + 1.0) / 6.0;
    } else if (i == j) {
      CDM[j * (N + 1) + i] = - x_c[j] / (2.0 * (1.0 - x_c[j] * x_c[j]));
    } else {
      double c_i = (i == 0 || i == N) ? 2.0 : 1.0;
      double c_j = (j == 0 || j == N) ? 2.0 : 1.0;
      CDM[j * (N + 1) + i] = c_i * pow(-1.0, i + j) / (c_j * (x_c[i] - x_c[j]));
    }
  }

}

/* First derivative finite difference matrix */
void FD1(double *D1N, int M, double h, int block_size, int grid_size) {
  FD1Kernel<<<grid_size, block_size>>>(D1N, M, h);
}

/* Second derivative finite difference matrix */
void FD2(double *D2N, int M, double h, int block_size, int grid_size) {
  FD2Kernel<<<grid_size, block_size>>>(D2N, M, h);
}

/* Chebyshev nodes and differentiation matrix */
void Chebyshev(double *CDM, double *x_c, int N, int grid_size, int block_size) {
  /* Nodes */
  ChebyshevNodes<<<grid_size, block_size>>>(x_c, N);
  /* Matrix */
  ChebyshevMatrix<<<grid_size, block_size>>>(CDM, x_c, N);
}