#include <stdio.h>
#include <stdlib.h>
#include<time.h>
#include "include/utils.h"

void fillVector(double *v, double h, int N) {
    for (int j=0; j < N; j++) {
        v[j] = j * h;
    }
}

void randomArray(double *a, int rows, int cols) {
    int i, j;
    for (int k = 0; k < rows * cols; k++) {
        i = k % rows;
        j = k / rows;
        if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1)
            a[k] = 0;
        else
            a[k] = (double)rand() / (double)RAND_MAX;
    }
}