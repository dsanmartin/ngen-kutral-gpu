#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "c/include/files.h"
#include "cu/include/solver.cuh"

int main(int argc, char *argv[]) {
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

	/* Parameters */
	double kappa = atof(argv[1]);//1e-3;
	double dx = 0.1, dy = 0.1, dt = atof(argv[2]);

	/* Domain definition */
	int Nx = 128; // x axis in matrix columns
	int Ny = 128; // y axis in matrix rows
	int T = 50;

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
	readInput("test/U128.txt", h_U0, Ny, Nx);
	//readInput("test/V_5.txt", h_V1, Ny, Nx);
	//readInput("test/V_5.txt", h_V2, Ny, Nx);

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
	solver(h_U0, h_V1, h_V2, h_U, Nx, Ny, T, dx, dy, dt, kappa);

	//clock_t end = clock();
	//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

	//printf("U=\n");
	//printApproximations(h_U, Nx, Ny, T);
	saveApproximation("test/output/Ua.txt", h_U, Nx, Ny, T);
	
	//printf("Time: %lf\n", time_spent);

	// Free CPU memory
	free(h_U0);
	free(h_V1);
	free(h_V2);
	free(h_U);

	return 0;
}