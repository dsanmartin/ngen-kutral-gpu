#include <stdio.h>
#include "include/solver.cuh"
#include "include/diffmat.cuh"
#include "../c/include/files.h"

// Multiply the arrays A and B on GPU and save the result in C, C(m,n) = A(m,k) * B(k,n)
void gpuBlasMmulBK(const double *A, const double *B, double *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
	
	// Destroy the handle
	cublasDestroy(handle);
}

// Multiply the arrays A and B on GPU and save the result in C, C(m,n) = A(m,k) * B(k,n)
void gpuBlasMmul(const double *A, const double *B, double *C, const int m, const int n) {
	int lda=m,ldb=n,ldc=m;
	const double alf = 1;
	const double bet = 0;
	const double *alpha = &alf;
	const double *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha, A, lda, B, ldb, beta, C, ldc);
	
	// Destroy the handle
	cublasDestroy(handle);
}

/*
void laplacian(double lap*, double *U, double *D2N, double kappa, int Nx, int Ny) {
	double *Uxx, *Uyy;
	cudaMalloc(&Uxx, Ny * Nx * sizeof(double));
	cudaMalloc(&Uyy, Ny * Nx * sizeof(double));

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, alpha, A, lda, B, ldb, beta, C, ldc);
	
	// Destroy the handle
	cublasDestroy(handle);

	cudaFree(Uxx);
	cudaFree(Uyy);
}
*/


void RHS(double *Unew, double *Uold, double *V1, double *V2, double *Dxx, double *Dyy, double kappa, double dt, int Nx, int Ny) {
	int lda=Ny, ldb=Nx, ldc=Ny;
	const double alf = kappa * dt;
	const double bet = 1;
	const double *alpha = &alf;
	const double *beta = &bet;

	// For euler method, copy u_old to u_new
	cudaMemcpy(Unew, Uold, Nx * Ny * sizeof(double), cudaMemcpyDeviceToDevice);

	/* Compute Laplacian */
	// Create handles for CUBLAS
	cublasHandle_t handle, handle2;
	cublasCreate(&handle);
	cublasCreate(&handle2);

	// Compute: kappa*dt D_yy U_old + "U_old"
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Ny, Nx, Nx, alpha, Dyy, lda, Uold, ldb, beta, Unew, ldc);
	
	// Wait first matrix multiplication to reuse it
	cudaDeviceSynchronize();

	// Compute: kappa*dt U_old D_xx^T + kappa*dt D_yy U_old + "U_old"
	cublasDgemm(handle2, CUBLAS_OP_N, CUBLAS_OP_T, Ny, Nx, Nx, alpha, Uold, lda, Dxx, ldb, beta, Unew, ldc);
	
	// Destroy the handles
	cublasDestroy(handle);
	cublasDestroy(handle2);
	/* End Laplacian computation */

}


void solver(double *h_U0, double *h_V1, double *h_V2, double *h_U, int Nx, int Ny, int T, 
	double dx, double dy, double dt, double kappa) {
	double *d_U, *d_V1, *d_V2, *d_Dxx, *d_Dyy;

	/* Create differentiation matrices for second derivative */
	// double *h_Dxx = (double *)malloc((Nx-2) * (Nx-2) * sizeof(double));
	// double *h_Dyy = (double *)malloc((Ny-2) * (Ny-2) * sizeof(double));
	double *h_Dxx = (double *)malloc(Nx * Nx * sizeof(double));
	double *h_Dyy = (double *)malloc(Ny * Ny * sizeof(double));
	//FD2(h_Dxx, Nx - 2, dx); // Fill differentiation matrix without boundaries
	//FD2(h_Dyy, Ny - 2, dy); // Fill differentiation matrix without boundaries
	FD2(h_Dxx, Nx, dx); // Fill differentiation matrix without boundaries
	FD2(h_Dyy, Ny, dy); // Fill differentiation matrix without boundaries

	/* Copy initial condition to temperatures approximation */
	memcpy(h_U, h_U0, (Nx * Ny) * sizeof(double));

	/* Memory allocation for matrices in GPU */
	cudaMalloc(&d_U, T * Ny * Nx * sizeof(double));
	// cudaMalloc(&d_Dxx, (Nx - 2) * (Nx - 2) * sizeof(double));
	// cudaMalloc(&d_Dyy, (Ny - 2) * (Ny - 2) * sizeof(double));
	cudaMalloc(&d_Dxx, Nx * Nx * sizeof(double));
	cudaMalloc(&d_Dyy, Ny * Ny * sizeof(double));
	cudaMalloc(&d_V1, Ny * Nx * sizeof(double));
	cudaMalloc(&d_V2, Ny * Nx * sizeof(double));

	/* Copy to GPU */
	cudaMemcpy(d_U, h_U, Ny * Nx * T * sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_Dxx, h_Dxx, (Nx - 2) * (Nx - 2) * sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_Dyy, h_Dyy, (Ny - 2) * (Ny - 2)* sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dxx, h_Dxx, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dyy, h_Dyy, Ny * Ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V1, h_V1, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2, h_V2, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);

	/* ODE Solver 	*/
	for (int t = 1; t < T; t++) {
		//printf("t: %d", t * Nx * Ny);
		//printMatrix(&d_U[t * Nx * Ny], Ny, Nx);
		//RHS(&d_U[t * (Nx-1) * (Ny-1)], &d_U[(t - 1) * (Nx-1) * (Ny-1)], d_V1, d_V2, d_Dxx, d_Dyy, kappa, dt, Nx-2, Ny-2);
		RHS(&d_U[t * Nx * Ny], &d_U[(t - 1) * Nx * Ny], d_V1, d_V2, d_Dxx, d_Dyy, kappa, dt, Nx, Ny);
	}

	// Matrix multiplication
	//gpuBlasMmul(d_U, d_V1, d_V2, Ny, Nx, Nx);
	
	// Copy from GPU to CPU
	cudaMemcpy(h_U, d_U, T * Ny * Nx * sizeof(double), cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_U);
	cudaFree(d_V1);
	cudaFree(d_V2);
	cudaFree(d_Dxx);
	cudaFree(d_Dyy);

	free(h_Dxx);
	free(h_Dyy);
}