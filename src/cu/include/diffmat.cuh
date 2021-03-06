/**
 * @file diffmat.cuh
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Differentiation matrices header
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */
 
#ifndef DIFFMAT_CUH
#define DIFFMAT_CUH

__global__ void FD1Kernel(double *D1N, int M, double h);
__global__ void FD2Kernel(double *D2N, int M, double h);
__global__ void ChebyshevNodes(double *x_c, int N);
__global__ void ChebyshevMatrix(double *CDM, double *x_c, int N);
__global__ void Chebyshev2Matrix(double *CDM2, double *CDM, int N);

#endif
