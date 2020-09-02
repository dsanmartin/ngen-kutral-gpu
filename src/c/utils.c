/**
 * @file utils.c
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Some extra functions
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "include/utils.h"

/**
 * @brief Fill array
 * 
 * @param v Pointer to fill, space discrete domain \f$ x \f$ or \f$ y \f$
 * @param h \f$ \Delta x \f$ or \f$ \Delta y \f$
 * @param N Number of nodes
 */
void fillVector(double *v, double h, int N) {
    for (int j = 0; j < N; j++) {
        v[j] = j * h;
    }
}

/**
 * @brief Create a random array
 * 
 * @param a Pointer to fill
 * @param rows Number of rows
 * @param cols Number of columns
 */
void randomArray(double *a, int rows, int cols) {
    int i, j;
    for (int k = 0; k < rows * cols; k++) {
        i = k % rows;
        j = k / rows;
        if (i == 0 || i == rows - 1 || j == 0 || j == cols - 1) // Hardcoded initial conditions...
            a[k] = 0;
        else
            a[k] = (double)rand() / (double)RAND_MAX;
    }
}