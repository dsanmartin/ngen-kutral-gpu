// Low level matrix multiplication on GPU using CUDA with CURAND and CUBLAS
// C(m,n) = A(m,k) * B(k,n)
#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>
#include <time.h>
//#include <cublas_v2.h>
#include "c/include/files.h"
#include "cu/include/solver.cuh"

int main() {
	// Allocate 3 arrays on CPU
	int rows_U, cols_U, rows_V, cols_V;

	// for simplicity we are going to use square arrays
	int N = 10000;
	rows_U = N;
	cols_U = N;
	rows_V = N;
	cols_V = N;

	float *h_U = (float *)malloc(rows_U * cols_U * sizeof(float));
	float *h_V = (float *)malloc(rows_V * cols_V * sizeof(float));
	float *h_C = (float *)malloc(rows_U * cols_V * sizeof(float));

	readInput("Inputs/U.txt", h_U, rows_U, cols_U);
	readInput("Inputs/V.txt", h_V, rows_V, cols_V);

  //printf("U=\n");
	//printMatrix(h_U, rows_U, cols_U);

  //printf("V=\n");
	//printMatrix(h_V, rows_V, cols_V);

	clock_t begin = clock();

	solver(h_U, h_V, h_C, rows_U, cols_U, rows_V, cols_V);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	//printf("C=\n");
	//printMatrix(h_C, rows_U, cols_V);

	printf("Time: %lf\n", time_spent);

	// Free CPU memory
	free(h_U);
	free(h_V);
	free(h_C);

	return 0;
}