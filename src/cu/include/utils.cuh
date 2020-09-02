/**
 * @file utils.cuh
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Extra CUDA functions header
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */
 
#ifndef UTILS_CUH
#define UTILS_CUH

__global__ void fillVectorKernel(double *vector, double h, int size);

#endif