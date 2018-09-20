#include <stdio.h>
#include <stdlib.h>
#include "include/diffmat.cuh"


/* First derivative finite difference matrix */
void FD1(double *D1N, int N, double h) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i - j == -1) {
        D1N[j * N + i] = 1 / (2 * h);
      } else if (i - j == 1) {
        D1N[j * N + i] = -1 / (2 * h);
      }
    }
  }
}

/* Second derivative finite difference matrix */
void FD2(double *D2N, int N, double h) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      if (i == j) {
        D2N[j * N + i] = -2 / (h * h);
      } else if (abs(i - j) == 1) {
        D2N[j * N + i] = 1 / (h * h);
      }
    }
  }
}