/**
 * @file utils.cu
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Extra CUDA functions
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <stdlib.h>
#include "include/utils.cuh"

/**
 * @brief Fill array
 * 
 * @param v Pointer to fill, space discrete domain \f$ x \f$ or \f$ y \f$
 * @param h \f$ \Delta x \f$ or \f$ \Delta y \f$
 * @param N Number of nodes
 */
__global__ void fillVectorKernel(double *v, double h, int N) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    if (tId < N)
        v[tId] = tId * h;
}