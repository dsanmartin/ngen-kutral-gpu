/**
 * @file diffmat.h
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Differentiation matrices header
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef DIFFMAT_H
#define DIFFMAT_H

void FD1(double *D1N, int M, double h);
void FD2(double *D2N, int M, double h);
void Chebyshev(double *CDM, double *x_c, int N);

#endif