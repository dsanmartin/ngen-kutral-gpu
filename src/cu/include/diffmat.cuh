#ifndef DIFFMAT_CUH
#define DIFFMAT_CUH

__global__ void FD1Kernel(double *D1N, int M, double h);
__global__ void FD2Kernel(double *D2N, int M, double h);
__global__ void ChebyshevKernel(double *CDM, double *x_c, int N);

#endif
