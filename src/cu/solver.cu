#include <stdio.h>
#include "include/solver.cuh"
#include "include/diffmat.cuh"
#include "../c/include/files.h"

void RHS(double *Unew, double *Uold, double *V1, double *V2, double *Dx, double *Dy, double *Dxx, double *Dyy,
	double *tmp, double kappa, double dt, int Nx, int Ny) {

	int lda=Ny, ldb=Nx, ldc=Ny;
	const double alf = kappa * dt;
	const double bet = 1;
	const double *alpha = &alf;
	const double *beta = &bet;
	const double v1 = -0.707107 * dt;
	const double v2 = -0.707107 * dt; 
	const double *av1 = &v1;
	const double *av2 = &v2;

	// Create handles for CUBLAS
	cublasHandle_t handle, handle2, handle3, handle4;
	cublasCreate(&handle);
	cublasCreate(&handle2);
	cublasCreate(&handle3);
	cublasCreate(&handle4);

	// For euler method, copy u_old to u_new
	cudaMemcpy(Unew, Uold, Nx * Ny * sizeof(double), cudaMemcpyDeviceToDevice);

	/* Compute Diffusion */
	// Compute: kappa*dt D_yy U_old + "U_old"
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Ny, Nx, Nx, alpha, Dyy, lda, Uold, ldb, beta, Unew, ldc);
	
	// Wait first matrix multiplication to reuse it
	cudaDeviceSynchronize();

	// Compute: kappa*dt U_old D_xx^T + kappa*dt D_yy U_old + "U_old"
	cublasDgemm(handle2, CUBLAS_OP_N, CUBLAS_OP_T, Ny, Nx, Nx, alpha, Uold, lda, Dxx, ldb, beta, Unew, ldc);
	/* End Diffusion computation */

	// Wait diffusion computation
	cudaDeviceSynchronize();

	/* Compute Convection*/
	// Compute: -v2 * dt D_y U_old + diffusion
	cublasDgemm(handle3, CUBLAS_OP_N, CUBLAS_OP_N, Ny, Nx, Nx, av2, Dy, lda, Uold, ldb, beta, Unew, ldc);

	// Wait for convection in y
	cudaDeviceSynchronize();

	// Compute: -v1 * dt U_old * D_x^T + (diffusion - y convection)
	cublasDgemm(handle4, CUBLAS_OP_N, CUBLAS_OP_T, Ny, Nx, Nx, av1, Uold, lda, Dx, ldb, beta, Unew, ldc);
	/* End convection computation */

	// Destroy the handles
	cublasDestroy(handle);
	cublasDestroy(handle2);
	cublasDestroy(handle3);
	cublasDestroy(handle4);
}


void solver(double *h_U0, double *h_V1, double *h_V2, double *h_U, int Nx, int Ny, int T, 
	double dx, double dy, double dt, double kappa) {
	double *d_U, *d_V1, *d_V2, *d_Dx, *d_Dy, *d_Dxx, *d_Dyy, *d_tmp;

	/* Create differentiation matrices for second derivative */
	double *h_Dx = (double *)malloc(Nx * Nx * sizeof(double));
	double *h_Dy = (double *)malloc(Ny * Ny * sizeof(double));
	double *h_Dxx = (double *)malloc(Nx * Nx * sizeof(double));
	double *h_Dyy = (double *)malloc(Ny * Ny * sizeof(double));
	double *h_tmp = (double *)malloc(Ny * Nx * sizeof(double)); // Temporal matrix for computations
	FD1(h_Dx, Nx, dx); // Fill differentiation matrix without boundaries
	FD1(h_Dy, Ny, dy); // Fill differentiation matrix without boundaries
	FD2(h_Dxx, Nx, dx); // Fill second differentiation matrix without boundaries
	FD2(h_Dyy, Ny, dy); // Fill second differentiation matrix without boundaries

	/* Copy initial condition to temperatures approximation */
	memcpy(h_U, h_U0, (Nx * Ny) * sizeof(double));

	/* Memory allocation for matrices in GPU */
	cudaMalloc(&d_U, T * Ny * Nx * sizeof(double));
	cudaMalloc(&d_Dx, Nx * Nx * sizeof(double));
	cudaMalloc(&d_Dy, Ny * Ny * sizeof(double));
	cudaMalloc(&d_Dxx, Nx * Nx * sizeof(double));
	cudaMalloc(&d_Dyy, Ny * Ny * sizeof(double));
	cudaMalloc(&d_V1, Ny * Nx * sizeof(double));
	cudaMalloc(&d_V2, Ny * Nx * sizeof(double));
	cudaMalloc(&d_tmp, Ny * Nx * sizeof(double));

	/* Copy to GPU */
	cudaMemcpy(d_U, h_U, Ny * Nx * T * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dx, h_Dx, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dy, h_Dy, Ny * Ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dxx, h_Dxx, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dyy, h_Dyy, Ny * Ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V1, h_V1, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2, h_V2, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_tmp, h_tmp, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);

	/* ODE Solver 	*/
	for (int t = 1; t < T; t++) {
		//printf("t: %d", t * Nx * Ny);
		//printMatrix(&d_U[t * Nx * Ny], Ny, Nx);
		//RHS(&d_U[t * (Nx-1) * (Ny-1)], &d_U[(t - 1) * (Nx-1) * (Ny-1)], d_V1, d_V2, d_Dxx, d_Dyy, kappa, dt, Nx-2, Ny-2);
		RHS(&d_U[t * Nx * Ny], &d_U[(t - 1) * Nx * Ny], d_V1, d_V2, d_Dx, d_Dy, d_Dxx, d_Dyy, d_tmp, kappa, dt, Nx, Ny);
	}
	
	// Copy from device to host
	cudaMemcpy(h_U, d_U, T * Ny * Nx * sizeof(double), cudaMemcpyDeviceToHost);

	// Free device memory
	cudaFree(d_U);
	cudaFree(d_V1);
	cudaFree(d_V2);
	cudaFree(d_Dx);
	cudaFree(d_Dy);
	cudaFree(d_Dxx);
	cudaFree(d_Dyy);
	cudaFree(d_tmp);

	// Free host memory
	free(h_Dx);
	free(h_Dy);
	free(h_Dxx);
	free(h_Dyy);
	free(h_tmp);
}