#include <stdlib.h>
#include "include/utils.h"

void fillVector(double *v, double h, int N) {
  for (int j=0; j < N; j++) {
    v[j] = j * h;
  }
}

void randomArray(double *a, int size) {
  for (int j=0; j < size; j++) {
    a[j] = (double)rand() / (double)RAND_MAX;
  }
}