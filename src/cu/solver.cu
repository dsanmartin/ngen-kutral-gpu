#include <stdio.h>
#include "include/solver.cuh"
#include "include/diffmat.cuh"
#include "../c/include/files.h"

__device__ double s(double u, double upc) {
	if (u >= upc)
		return 1;
	else
		return 0;
}

__device__ double f(double u, double b, double alpha, double epsilon, double upc) {
	return s(u, upc) * b * exp(u / (1 + epsilon * u)) - alpha * u;
}

__device__ double g(double u, double b, double epsilon, double q, double upc) {
	return -s(u, upc) * (epsilon / q) * b * exp(u /(1 + epsilon * u));
}

__global__ void f_kernel(const double* __restrict__ d_U, const double* __restrict__ d_B, double *d_F, 
	double alpha, double epsilon, double upc, int Nx, int Ny) {
	
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		d_F[tId] = f(d_U[tId], d_B[tId], alpha, epsilon, upc);
	}
}

__global__ void g_kernel(const double* __restrict__ d_U, const double* __restrict__ d_B, double *d_G, 
	double epsilon, double q, double upc, double dt, int Nx, int Ny) {
	
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		d_G[tId] = d_B[tId] + dt * g(d_U[tId], d_B[tId], epsilon, q, upc);
	}
}

__global__ void fuel(const double* __restrict__ d_F, const double* __restrict__ d_tmp, double *d_U, double dt, int Nx, int Ny) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		d_U[tId] = dt * d_F[tId] + d_tmp[tId];
	}
}

/* Boundary conditions */
// For U
__global__ void h1(double *U, int Nx, int Ny) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		int i = tId % Nx;
		int j = tId / Nx;
		if (i == 0 || i == (Nx-1) || j == 0 || j == (Ny-1))
			U[j + i * Nx] = 0;
	}
}

// For B
__global__ void h2(double *B, int Nx, int Ny) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		int i = tId % Nx;
		int j = tId / Nx;
		if (i == 0 || i == (Nx-1) || j == 0 || j == (Ny-1))
			B[j + i * Nx] = 0;
	}
}

void RHS(double *Unew, double *Uold, double *Bnew, double *Bold, double *V1, double *V2, double *Dx, double *Dy, 
	double *Dxx, double *Dyy,	double *F, double kappa, double epsilon, double upc, double q, double alpha, 
	double dt, int Nx, int Ny) {

	int lda=Ny, ldb=Nx, ldc=Ny;
	const double alf = kappa * dt;
	const double bet = 1;
	const double *alph = &alf;
	const double *beta = &bet;
	const double v1 = -0.707107 * dt;
	const double v2 = -0.707107 * dt; 
	const double *av1 = &v1;
	const double *av2 = &v2;
	const double fdt = dt;
	const double *ffdt = &fdt;

	// For fuel
	int size = Nx * Ny;
	int block_size = 256;
	int grid_size = (int) ceil((float)size / block_size);

	/* Compute Fuel */
	g_kernel<<<grid_size, block_size>>>(Uold, Bold, Bnew, epsilon, q, upc, dt, Nx, Ny);
	/* End fuel computation */

	// Create handles for CUBLAS
	cublasHandle_t handle, handle2, handle3, handle4, handle5;
	cublasCreate(&handle);
	cublasCreate(&handle2);
	cublasCreate(&handle3);
	cublasCreate(&handle4);
	cublasCreate(&handle5);

	// For euler method, copy u_old to u_new
	cudaMemcpy(Unew, Uold, Nx * Ny * sizeof(double), cudaMemcpyDeviceToDevice);

	/* Compute Diffusion */
	// Compute: kappa*dt D_yy U_old + "U_old"
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Ny, Nx, Nx, alph, Dyy, lda, Uold, ldb, beta, Unew, ldc);
	
	// Wait first matrix multiplication to reuse it
	cudaDeviceSynchronize();

	// Compute: kappa*dt U_old D_xx^T + kappa*dt D_yy U_old + "U_old"
	cublasDgemm(handle2, CUBLAS_OP_N, CUBLAS_OP_T, Ny, Nx, Nx, alph, Uold, lda, Dxx, ldb, beta, Unew, ldc);
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
	
	cudaDeviceSynchronize();

	/* Compute Reaction */
	f_kernel<<<grid_size, block_size>>>(Uold, Bold, F, alpha, epsilon, upc, Nx, Ny);

	cudaDeviceSynchronize();

	// double *d_tmp;
	// cudaMalloc(&d_tmp, Ny * Nx * sizeof(double));
	// cudaMemcpy(d_tmp, Unew, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	// fuel<<<grid_size, block_size>>>(F, d_tmp, Unew, dt, Nx, Ny);
	// cudaFree(d_tmp);
	//free(h_tmp);

	//cudaDeviceSynchronize();

	cublasDgeam(handle5, CUBLAS_OP_N, CUBLAS_OP_N, Ny, Nx, beta, Unew, ldc, ffdt, F, ldb, Unew, ldc);
	/* End reaction computation */

	cudaDeviceSynchronize();

	// Destroy the handles
	cublasDestroy(handle);
	cublasDestroy(handle2);
	cublasDestroy(handle3);
	cublasDestroy(handle4);
	cublasDestroy(handle5);

	/* Boundary conditions */
	h1<<<grid_size, block_size>>>(Unew, Nx, Ny);
	h2<<<grid_size, block_size>>>(Bnew, Nx, Ny);
}


void solver(double *h_U0, double *h_B0, double *h_V1, double *h_V2, double *h_U, double *h_B, int Nx, int Ny, int T, 
	double dx, double dy, double dt, double kappa, double epsilon, double upc, double q, double alpha) {
	
	double *d_U, *d_B, *d_V1, *d_V2, *d_Dx, *d_Dy, *d_Dxx, *d_Dyy, *d_F;

	/* Create differentiation matrices for second derivative */
	double *h_Dx = (double *)malloc(Nx * Nx * sizeof(double));
	double *h_Dy = (double *)malloc(Ny * Ny * sizeof(double));
	double *h_Dxx = (double *)malloc(Nx * Nx * sizeof(double));
	double *h_Dyy = (double *)malloc(Ny * Ny * sizeof(double));
	double *h_F = (double *)malloc(Ny * Nx * sizeof(double)); // Temporal matrix for fuel computation
	FD1(h_Dx, Nx, dx); // Fill differentiation matrix without boundaries
	FD1(h_Dy, Ny, dy); // Fill differentiation matrix without boundaries
	FD2(h_Dxx, Nx, dx); // Fill second differentiation matrix without boundaries
	FD2(h_Dyy, Ny, dy); // Fill second differentiation matrix without boundaries

	/* Copy initial condition to temperatures approximation */
	memcpy(h_U, h_U0, (Nx * Ny) * sizeof(double));

	/* Copy initial condition to fuels approximation */
	memcpy(h_B, h_B0, (Nx * Ny) * sizeof(double));

	/* Memory allocation for matrices in GPU */
	cudaMalloc(&d_U, T * Ny * Nx * sizeof(double));
	cudaMalloc(&d_B, T * Ny * Nx * sizeof(double));
	cudaMalloc(&d_Dx, Nx * Nx * sizeof(double));
	cudaMalloc(&d_Dy, Ny * Ny * sizeof(double));
	cudaMalloc(&d_Dxx, Nx * Nx * sizeof(double));
	cudaMalloc(&d_Dyy, Ny * Ny * sizeof(double));
	cudaMalloc(&d_V1, Ny * Nx * sizeof(double));
	cudaMalloc(&d_V2, Ny * Nx * sizeof(double));
	cudaMalloc(&d_F, Ny * Nx * sizeof(double));

	/* Copy to GPU */
	cudaMemcpy(d_U, h_U, T * Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, T * Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dx, h_Dx, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dy, h_Dy, Ny * Ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dxx, h_Dxx, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dyy, h_Dyy, Ny * Ny * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V1, h_V1, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_V2, h_V2, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, h_F, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);

	/* ODE Solver 	*/
	for (int t = 1; t < T; t++) {
		//printf("t: %d", t * Nx * Ny);
		//printMatrix(&d_U[t * Nx * Ny], Ny, Nx);
		//RHS(&d_U[t * (Nx-1) * (Ny-1)], &d_U[(t - 1) * (Nx-1) * (Ny-1)], d_V1, d_V2, d_Dxx, d_Dyy, kappa, dt, Nx-2, Ny-2);
		RHS(&d_U[t * Ny * Nx], &d_U[(t - 1) * Ny * Nx], &d_B[t * Ny * Nx], &d_B[(t - 1) * Ny * Nx],
			d_V1, d_V2, d_Dx, d_Dy, d_Dxx, d_Dyy, d_F, kappa, epsilon, upc, q, alpha, dt, Nx, Ny);
	}
	
	/* Copy from device to host */
	cudaMemcpy(h_U, d_U, T * Ny * Nx * sizeof(double), cudaMemcpyDeviceToHost); // U
	cudaMemcpy(h_B, d_B, T * Ny * Nx * sizeof(double), cudaMemcpyDeviceToHost); // B

	// Free device memory
	cudaFree(d_U);
	cudaFree(d_B);
	cudaFree(d_V1);
	cudaFree(d_V2);
	cudaFree(d_Dx);
	cudaFree(d_Dy);
	cudaFree(d_Dxx);
	cudaFree(d_Dyy);
	cudaFree(d_F);

	// Free host memory
	free(h_Dx);
	free(h_Dy);
	free(h_Dxx);
	free(h_Dyy);
	free(h_F);
}