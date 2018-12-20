#ifndef DIFFMAT_H
#define DIFFMAT_H

void FD1(double *D1N, int M, double h);
void FD2(double *D2N, int M, double h);
void Chebyshev(double *CDM, double *x_c, int N);

#endif