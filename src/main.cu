#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "c/include/files.h"
#include "cu/include/solver.cuh"

int main() {
	// Parameters
	/*
	int *Nxx, *Ny, *Tmax;
	double *xmin, *xmax, *ymin, *ymax, *dt, *kap, *eps, *upc, *q, *alp;
	int *Nx;// = (int*) malloc(sizeof(int));

	readConfig("test/config.txt", Nx, Ny, xmin, xmax, ymin, ymax, Tmax, 
		dt, kap, eps, upc, q, alp);

	printf("Nx: %d", Nx);
	*/

	// Allocate 3 arrays on CPU
	//int rows_U, cols_U, rows_V, cols_V;

	/* Domain definition */
	int Nx = 5; // x axis in matrix columns
	int Ny = 5; // y axis in matrix rows
	int T = 5;

	// for simplicity we are going to use square arrays
	// int N = 5;
	// rows_U = N;
	// cols_U = N;
	// rows_V = N;
	// cols_V = N;

	/* Memory allocation for matrices in host*/
	// Initial condition
	double *h_U0 = (double *)malloc(Nx * Ny * sizeof(double)); 

	// Vector field
	double *h_V1 = (double *)malloc(Nx * Ny * sizeof(double)); 
	double *h_V2 = (double *)malloc(Nx * Ny * sizeof(double));

	// Temperature solutions
	double *h_U = (double *)malloc(Nx * Ny * T * sizeof(double));

	/* Read initial conditions */
	readInput("test/U_5.txt", h_U0, Ny, Nx);
	readInput("test/V_5.txt", h_V1, Ny, Nx);
	readInput("test/V_5.txt", h_V2, Ny, Nx);

	//char *line;
	//readConf("test/config.txt"); 

  // printf("U0=\n");
	// printMatrix(h_U0, Ny, Nx);

  // printf("V1=\n");
	// printMatrix(h_V1, Ny, Nx);

	// printf("V2=\n");
	// printMatrix(h_V2, Ny, Nx);

	//clock_t begin = clock();

	//solver(h_U, h_V, h_C, rows_U, cols_U, rows_V, cols_V);
	solver(h_U0, h_V1, h_V2, h_U, Nx, Ny, T);

	//clock_t end = clock();
	//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	printf("U=\n");
	printApproximations(h_U, Nx, Ny, T);
	
	//printf("Time: %lf\n", time_spent);

	// Free CPU memory
	free(h_U0);
	free(h_V1);
	free(h_V2);
	free(h_U);

	return 0;
}