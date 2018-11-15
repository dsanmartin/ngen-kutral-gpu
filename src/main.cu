#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "c/include/files.h"
#include "cu/include/solver.cuh"

// int main(int argc, char *argv[]) {
// 	int m = 128;
// 	int n = 128;
	
// 	double *A = (double *)malloc(m * n * sizeof(double)); 
// 	double *B = (double *)malloc(m * n * sizeof(double)); 
// 	double *C = (double *)malloc(m * m * sizeof(double)); 
// 	readInput("test/A.txt", A, m, n);
// 	readInput("test/B.txt", B, n, m);
// 	//printf("A: \n");
// 	//printMatrix(A, m, n);
// 	//printf("\nB: \n");
// 	//printMatrix(B, n, m);

// 	matmul(A, B, C, m, n);

// 	//printf("\nC: \n");
// 	//printMatrix(C, m, m);
// 	saveApproximation("test/C.txt", C, m, m, 1);
// 	free(A);
// 	free(B);
// 	free(C);

// 	return 0;
// }

int main(int argc, char *argv[]) {
	// Parameters
	/*
	int *Nxx, *Ny, *Tmax;
	double *xmin, *xmax, *ymin, *ymax, *dt, *kap, *eps, *upc, *q, *alp;
	int *Nx;// = (int*) malloc(sizeof(int));

	readConfig("test/config.txt", Nx, Ny, xmin, xmax, ymin, ymax, Tmax, 
		dt, kap, eps, upc, q, alp);

	printf("Nx: %d", Nx);
	char *line;
	readConf("test/config.txt"); 
	*/

	/* Parameters */
	double kappa = 1e-1;//atof(argv[1]);//1e-3;
	double epsilon = 3e-1;
	double upc = 3;
	double q = 1;
	double alpha = 1e-3;
	double dt = 1e-2;//atof(argv[2]);
	double xmin = 0;
	double xmax = 90;
	double ymin = 0;
	double ymax = 90;

	/* Domain definition */
	int Nx = 128; // x axis in matrix columns
	int Ny = 128; // y axis in matrix rows
	int T = 5000;

	double dx = (xmax - xmin) / Nx;
	double dy = (ymax - ymin) / Ny;

	//printf("kappa: %.8f, dt: %.8f\n", kappa, dt);
	//printf("dx: %.8f, dy: %.8f\n", dx, dy);

	/* Memory allocation for matrices in host*/
	// Temperature initial condition
	double *h_U0 = (double *)malloc(Ny * Nx * sizeof(double)); 

	// Fuel initial condition
	double *h_B0 = (double *)malloc(Ny * Nx * sizeof(double)); 

	// Vector field
	double *h_V1 = (double *)malloc(Ny * Nx * sizeof(double)); 
	double *h_V2 = (double *)malloc(Ny * Nx * sizeof(double));

	// Temperature approximation
	double *h_U = (double *)malloc(T * Ny * Nx * sizeof(double));

	// Fuel approximation
	double *h_B = (double *)malloc(T * Ny * Nx * sizeof(double));

	/* Read initial conditions */
	readInput("test/U128.txt", h_U0, Ny, Nx); // U0
	readInput("test/B128.txt", h_B0, Ny, Nx); // B0
	readInput("test/V1128.txt", h_V1, Ny, Nx); // W1
	readInput("test/V2128.txt", h_V2, Ny, Nx); // W2

	solver(h_U0, h_B0, h_V1, h_V2, h_U, h_B, Nx, Ny, T, dx, dy, dt, kappa, epsilon, upc, q, alpha);

	saveApproximation("test/output/Ua.txt", h_U, Nx, Ny, T);
	saveApproximation("test/output/Ba.txt", h_B, Nx, Ny, T);
	
	// Free CPU memory
	free(h_U0);
	free(h_B0);
	free(h_V1);
	free(h_V2);
	free(h_U);
	free(h_B);

	return 0;
}