#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "c/include/files.h"
#include "c/include/structures.h"
#include "c/include/diffmat.h"
#include "cu/include/solver.cuh"
#include "cu/include/diffmat.cuh"

int main(int argc, char *argv[]) {

	/* PDE parameters */
	Parameters parameters;
	parameters.kappa = 1e-1;//atof(argv[1]);//1e-3;
	parameters.epsilon = 3e-1;
	parameters.upc = 3;
	parameters.q = 1;
	parameters.alpha = 1e-3;
	parameters.x_min = 0;
	parameters.x_max = 90;
	parameters.y_min = 0;
	parameters.y_max = 90;
	parameters.t_max = 1;

	/* Domain definition */
	parameters.L = 5000; // Time resolution
	parameters.M = 128; // Spatial resolution (y-axis - matrix rows)
	parameters.N = 128; // Spatial resolution (x-axis - matrix columns)

	/* Methods */
	parameters.spatial = "FD";
	parameters.time = "RK4";
	
	/* Domain differentials */
	double dx = (parameters.x_max - parameters.x_min) / parameters.N;
	double dy = (parameters.y_max - parameters.y_min) / parameters.M;
	double dt = parameters.t_max / parameters.L;

	printf("Kappa: %f\n", parameters.kappa);
	printf("Spatial: %s\n", parameters.spatial);
	printf("Time: %s\n", parameters.time);

	// int size = M * N;
	// int block_size = 256;
	// int grid_size = (int) ceil((float)size / block_size);

	// double *h_M = (double*) malloc((M + 1) * (N + 1) * sizeof(double));
	// double *h_v = (double*) malloc((N + 1)* sizeof(double));
	// double *d_M, *d_v;
	// double *h_M_CPU = (double*) malloc((M + 1) * (N + 1) * sizeof(double));
	// double *h_v_CPU = (double*) malloc((N + 1) * sizeof(double));

	// cudaMalloc(&d_M, (M + 1) * (N + 1) * sizeof(double));
	// cudaMalloc(&d_v, (N + 1) * sizeof(double));
	// cudaMemcpy(d_M, h_M, (M + 1) * (N + 1)* sizeof(double), cudaMemcpyHostToDevice);
	// cudaMemcpy(d_v, h_v, (N + 1)* sizeof(double), cudaMemcpyHostToDevice);

	// Chebyshev(d_M, d_v, N, grid_size, block_size);

	// cudaMemcpy(h_M, d_M, (M + 1) * (N + 1) * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_v, d_v, (N + 1)* sizeof(double), cudaMemcpyDeviceToHost);
	// printMatrix(h_v, 1, N + 1);
	// printMatrix(h_M, M + 1, N + 1);

	// //FD2(h_M_CPU, M, dx);
	// Chebyshev(h_M_CPU, h_v_CPU, N);
	// printMatrix(h_v_CPU, 1, N + 1);
	// printMatrix(h_M_CPU, M + 1, N + 1);

	// /* Memory allocation for matrices in host*/
	// // Lemperature initial condition
	// double *h_U0 = (double *)malloc(M * N * sizeof(double)); 

	// // Fuel initial condition
	// double *h_B0 = (double *)malloc(M * N * sizeof(double)); 

	// // Vector field
	// double *h_V1 = (double *)malloc(M * N * sizeof(double)); 
	// double *h_V2 = (double *)malloc(M * N * sizeof(double));

	// // Lemperature approximation
	// double *h_U = (double *)malloc(L * M * N * sizeof(double));

	// // Fuel approximation
	// double *h_B = (double *)malloc(L * M * N * sizeof(double));

	// /* Read initial conditions */
	// readInput("test/U128.txt", h_U0, M, N); // U0
	// readInput("test/B128.txt", h_B0, M, N); // B0
	// readInput("test/V1128.txt", h_V1, M, N); // W1
	// readInput("test/V2128.txt", h_V2, M, N); // W2

	// solver(h_U0, h_B0, h_V1, h_V2, h_U, h_B, N, M, L, dx, dy, dt, kappa, epsilon, upc, q, alpha);

	// saveApproximation("test/output/Ua.txt", h_U, N, M, L);
	// saveApproximation("test/output/Ba.txt", h_B, N, M, L);
	
	// // Free CPU memory
	// free(h_U0);
	// free(h_B0);
	// free(h_V1);
	// free(h_V2);
	// free(h_U);
	// free(h_B);

	return 0;
}