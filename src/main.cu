#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "c/include/files.h"
#include "cu/include/solver.cuh"

int main() {
	// Parameters
	int *Nxx, *Ny, *Tmax;
	float *xmin, *xmax, *ymin, *ymax, *dt, *kap, *eps, *upc, *q, *alp;
	int *Nx;// = (int*) malloc(sizeof(int));

	readConfig("test/config.txt", Nx, Ny, xmin, xmax, ymin, ymax, Tmax, 
		dt, kap, eps, upc, q, alp);

	printf("Nx: %d", Nx);

	// Allocate 3 arrays on CPU
	int rows_U, cols_U, rows_V, cols_V;

	// for simplicity we are going to use square arrays
	int N = 100;
	rows_U = N;
	cols_U = N;
	rows_V = N;
	cols_V = N;

	float *h_U = (float *)malloc(rows_U * cols_U * sizeof(float));
	float *h_V = (float *)malloc(rows_V * cols_V * sizeof(float));
	float *h_C = (float *)malloc(rows_U * cols_V * sizeof(float));

	//readInput("test/U.txt", h_U, rows_U, cols_U);
	//readInput("test/V.txt", h_V, rows_V, cols_V);

	//char *line;
	//readConf("test/config.txt"); 

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