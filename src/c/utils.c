#include <stdlib.h>
#include "include/utils.h"

void fillVector(double *v, double h, int N) {
  for (int j=0; j < N; j++) {
    v[j] = j * h;
  }
}