#include <stdlib.h>
#include "include/utils.cuh"

__global__ void fillVectorKernel(double *v, double h, int N) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < N)
    v[tId] = tId * h;
}