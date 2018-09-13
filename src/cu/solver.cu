#include <stdio.h>
#include "include/solver.cuh"
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


void RHS(double *U, double *V1, double *V2, int Nx, int Ny) {
	double *lap;//, *Uxx, *Uyy;
	cudaMalloc(&lap, Ny * Nx * sizeof(double));
	//cudaMalloc(&Uxx, Ny * Nx * sizeof(double));
	//cudaMalloc(&Uyy, Ny * Nx * sizeof(double));

	gpuBlasMmul(V1, , Uxx, Ny, Nx);

	cudaFree(lap);
	//cudaFree(Uxx);
	//cudaFree(Uyy);
}

//void ODESolver(double)

void solver(double *h_U0, double *h_V1, double *h_V2, double *h_U, int Nx, int Ny, int T) {

	/* Copy initial condition to temperature approximation */
	memcpy(h_U, h_U0, (Nx * Ny) * sizeof(double));
	//printMatrix(h_U0, Nx, Ny);

	/* Memory allocation for matrices in GPU */
	double *d_V1, *d_V2, *d_U; // *d_U0,
	cudaMalloc(&d_U, T * Ny * Nx * sizeof(double));
	cudaMalloc(&d_V1, Ny * Nx * sizeof(double));
	cudaMalloc(&d_V2, Ny * Nx * sizeof(double));

	/* Copy to GPU */
	cudaMemcpy(d_U, h_U, Ny * Nx * T * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V1, h_V1, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2, h_V2, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	
	/* ODE Solver */
	for (int t = 0; t < T; t++) {
		//printf("t: %d", t * Nx * Ny);
		//printMatrix(&d_U[t * Nx * Ny], Ny, Nx);
		RHS(&d_U[t * Nx * Ny], d_V1, d_V2, Nx, Ny);
	}

	// Matrix multiplication
	//gpuBlasMmul(d_U, d_V1, d_V2, Ny, Nx, Nx);
	
	// Copy from GPU to CPU
	cudaMemcpy(h_U, d_U, T * Ny * Nx * sizeof(double), cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_U);
	cudaFree(d_V1);
	cudaFree(d_V2);
}