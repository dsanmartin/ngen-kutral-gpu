#include "include/solver.cuh"
#include "../c/include/files.h"

// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gpuBlasMmul(const float *A, const float *B, float *C, const int m, const int k, const int n) {
	int lda=m,ldb=k,ldc=m;
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;

	// Create a handle for CUBLAS
	cublasHandle_t handle;
	cublasCreate(&handle);

	// Do the actual multiplication
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

	// Destroy the handle
	cublasDestroy(handle);
}

void solver(float *h_U, float *h_V, float *h_C, int rows_U, int cols_U, int rows_V, int cols_V) {

  // Allocate matrices in GPU
	float *d_U, *d_V, *d_C;
	cudaMalloc(&d_U, rows_U * cols_U * sizeof(float));
	cudaMalloc(&d_V, rows_V * cols_V * sizeof(float));
	cudaMalloc(&d_C, rows_U * cols_V * sizeof(float));

	// Copy to GPU
	cudaMemcpy(d_U, h_U, rows_U * cols_U * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_V, h_V, rows_V * cols_V * sizeof(float), cudaMemcpyHostToDevice);
	
	// Matrix multiplication
	gpuBlasMmul(d_U, d_V, d_C, rows_U, cols_U, cols_V);
	
	// Copy from GPU to CPU
	cudaMemcpy(h_C, d_C, rows_U * cols_V * sizeof(float), cudaMemcpyDeviceToHost);

	// Free GPU memory
	cudaFree(d_U);
	cudaFree(d_V);
	cudaFree(d_C);
}