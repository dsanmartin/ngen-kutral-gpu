#ifndef DIFFMAT_CUH
#define DIFFMAT_CUH

void FD1(double *D1N, int M, double h, int grid_size, int block_size);
void FD2(double *D2N, int M, double h, int grid_size, int block_size);
void Chebyshev(double *CDM, double *x_c, int N, int grid_size, int block_size);

#endif
