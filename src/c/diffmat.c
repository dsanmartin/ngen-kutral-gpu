#include <stdlib.h>
#include <math.h>
#include "include/diffmat.h"

#define M_PI 3.14159265358979323846

void FD1(double *D1N, int M, double h) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (i - j == -1) {
                D1N[j * M + i] = 1 / (2 * h);
            } else if (i - j == 1) {
                D1N[j * M + i] = -1 / (2 * h);
            }
        }    
    }
    D1N[M * (M - 2)] = -1 / (2 * h);
    D1N[2 * M - 1] = 1 / (2 * h);
}

void FD2(double *D2N, int M, double h) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            if (i == j) {
                D2N[j * M + i] = -2 / (h * h);
            } else if (abs(i - j) == 1) {
                D2N[j * M + i] = 1 / (h * h);
            }
        }
    }
    D2N[M * (M - 2)] = 1 / (h * h);
    D2N[2 * M - 1] = 1 / (h * h);
}


void Chebyshev(double *CDM, double *x_c, int N) {
    double c_i, c_j;
    /* Compute Chebyshev nodes */
    for (int j=0; j <= N; j++) {
        x_c[j] = cos(j * M_PI / N);
    }

    /* Chebyshev differentiation matrix */
    for (int i=0; i <= N; i++) {
        for (int j=0; j <= N; j++) {
            if (i == 0 && j == 0) {
                CDM[j * (N + 1) + i] = (2.0 * N * N + 1.0) / 6.0;
            } else if (i == N && j == N) {
                CDM[j * (N + 1) + i] = - (2.0 * N * N + 1.0) / 6.0;
            } else if (i == j) {
                CDM[j * (N + 1) + i] = - x_c[j] / (2.0 * (1.0 - x_c[j] * x_c[j]));
            } else {
                c_i = (i == 0 || i == N) ? 2.0 : 1.0;
                c_j = (j == 0 || j == N) ? 2.0 : 1.0;
                CDM[j * (N + 1) + i] = c_i * pow(-1.0, i + j) / (c_j * (x_c[i] - x_c[j]));
            }
        }
    }
}