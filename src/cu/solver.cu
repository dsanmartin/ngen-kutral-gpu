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

__global__ void transposeKernel(const double *A, double *AT, int m, int n) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < m * n) {
    int i = tId % m; // Row index
    int j = tId / m; // Col index
    AT[i * n + j] = A[j * m + i];
  }
}

__global__ void matmulKernel(const double *A, const double *B, double *C, int m, int n) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < m * n) {
    int i = tId % m; // Row index
    int j = tId / m; // Col index
    double c = 0;
    
    for (int k = 0; k < n; k++) {
      c += A[k * m + i] * B[j * n + k];
    }

    C[tId] = c;
  }
}

__global__ void fKernel(const double* __restrict__ d_U, const double* __restrict__ d_B, double *d_F, 
	double alpha, double epsilon, double upc, int Nx, int Ny) {
	
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		d_F[tId] = f(d_U[tId], d_B[tId], alpha, epsilon, upc);
	}
}

__global__ void gKernel(const double* __restrict__ d_U, const double* __restrict__ d_B, double *d_G, 
	double epsilon, double q, double upc, int Nx, int Ny) {
	
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		d_G[tId] = g(d_U[tId], d_B[tId], epsilon, q, upc);
	}
}

/* Boundary conditions */
// For U
__global__ void h1(double *U, int Nx, int Ny) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		int i = tId % Ny;
		int j = tId / Ny;
		if (i == 0 || i == (Ny-1) || j == 0 || j == (Nx-1))
			U[j + i * Nx] = 0;
	}
}

// For B
__global__ void h2(double *B, int Nx, int Ny) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < Ny * Nx) {
		int i = tId % Ny;
		int j = tId / Ny;
		if (i == 0 || i == (Ny-1) || j == 0 || j == (Nx-1))
			B[j + i * Nx] = 0;
	}
}

__global__ void RHSEulerKernel(double *Uold, double *Bold, double *V1, double *V2, double *Ux, double *Uy, 
	double *Uxx, double *Uyy, double *F, double *G, double *Unew, double *Bnew, double kappa, double epsilon, double upc, double q, double alpha, 
	double dt, int Nx, int Ny) {
		
		int tId = threadIdx.x + blockIdx.x * blockDim.x;
		if (tId < Ny * Nx) {
			double diffusion = kappa  * (Uxx[tId] + Uyy[tId]);
			double convection = V1[tId] * Ux[tId] + V2[tId] * Uy[tId];
			double reaction = F[tId];			
			Unew[tId] = Uold[tId] + dt * (diffusion - convection + reaction);
			Bnew[tId] = Bold[tId] + dt * G[tId];
		}
}

void RHS(double *Unew, double *Uold, double *Bnew, double *Bold, double *V1, double *V2, double *Dx, double *Dy, 
	double *Dxx, double *Dyy, double kappa, double epsilon, double upc, double q, double alpha, 
	double dt, int Nx, int Ny, int grid_size, int block_size, double *d_Uy, double *d_Uyy,
	double *d_DxxT, double *d_DxT, double *d_Ux, double *d_Uxx, double *d_F, double *d_G) {

	/* Compute partial derivatives */
	
	// First derivarive w/r to y
	matmulKernel<<<grid_size, block_size>>>(Dy, Uold, d_Uy, Ny, Nx); // Uy
	
	// Second derivative w/r to y
	matmulKernel<<<grid_size, block_size>>>(Dyy, Uold, d_Uyy, Ny, Nx); // Uyy

	// Transpose x differentiation matrices
	transposeKernel<<<grid_size, block_size>>>(Dx, d_DxT, Nx, Nx); // DxT
	transposeKernel<<<grid_size, block_size>>>(Dxx, d_DxxT, Nx, Nx); // DxxT

	// First derivative w/r to x
	matmulKernel<<<grid_size, block_size>>>(Uold, d_DxT, d_Ux, Ny, Nx);

	// Second derivative w/r to x
	matmulKernel<<<grid_size, block_size>>>(Uold, d_DxxT, d_Uxx, Ny, Nx);

	// Compute F
	fKernel<<<grid_size, block_size>>>(Uold, Bold, d_F, alpha, epsilon, upc, Nx, Ny);

	// Compute G
	gKernel<<<grid_size, block_size>>>(Uold, Bold, d_G, epsilon, q, upc, Nx, Ny);

	RHSEulerKernel<<<grid_size, block_size>>>(Uold, Bold, V1, V2, d_Ux, d_Uy, d_Uxx, d_Uyy, d_F, d_G, 
		Unew, Bnew, kappa, epsilon, upc, q, alpha, dt, Nx, Ny);

	/* Boundary conditions */
	h1<<<grid_size, block_size>>>(Unew, Nx, Ny); // Temperature
	h2<<<grid_size, block_size>>>(Bnew, Nx, Ny); // Fuel
}

void eulerMethod(double *d_U, double *d_B, double *d_V1, double *d_V2, 
	double *d_Dx, double *d_Dy, double *d_Dxx, double *d_Dyy, 
	double kappa, double epsilon, double upc, double q, double alpha, double dt, int Nx, int Ny, int T,
	int grid_size, int block_size, double *d_Uy, double *d_Uyy,
	double *d_DxxT, double *d_DxT, double *d_Ux, double *d_Uxx, double *d_F, double *d_G) {

	for (int t = 1; t < T; t++) {
		RHS(&d_U[t * Ny * Nx], &d_U[(t - 1) * Ny * Nx], &d_B[t * Ny * Nx], &d_B[(t - 1) * Ny * Nx],
			d_V1, d_V2, d_Dx, d_Dy, d_Dxx, d_Dyy, kappa, epsilon, upc, q, alpha, dt, Nx, Ny,
			grid_size, block_size, d_Uy, d_Uyy, d_DxxT, d_DxT, d_Ux, d_Uxx, d_F, d_G);
	}
}

void solver(double *h_U0, double *h_B0, double *h_V1, double *h_V2, double *h_U, double *h_B, int Nx, int Ny, int T, 
	double dx, double dy, double dt, double kappa, double epsilon, double upc, double q, double alpha) {

		int size = Ny * Nx;
		int block_size = 256;
		int grid_size = (int) ceil((float)size / block_size);

		double *d_U, *d_B, *d_V1, *d_V2, *d_Dx, *d_Dy, *d_Dxx, *d_Dyy; //*d_F;
		double *DxT, *DxxT, *Ux, *UxT, *Uy, *Uxx, *UxxT, *Uyy, *F, *G;
		double *d_DxT, *d_DxxT, *d_Ux, *d_UxT, *d_Uy, *d_Uxx, *d_UxxT, *d_Uyy, *d_F, *d_G;

		/* Create differentiation matrices for second derivative */
		double *h_Dx = (double *)malloc(Nx * Nx * sizeof(double));
		double *h_Dy = (double *)malloc(Ny * Ny * sizeof(double));
		double *h_Dxx = (double *)malloc(Nx * Nx * sizeof(double));
		double *h_Dyy = (double *)malloc(Ny * Ny * sizeof(double));
		//double *h_F = (double *)malloc(Ny * Nx * sizeof(double)); // Temporal matrix for fuel computation
		DxT = (double *)malloc(Nx * Nx * sizeof(double));
		DxxT = (double *)malloc(Nx * Nx * sizeof(double));
		Ux = (double *)malloc(Nx * Ny * sizeof(double));
		UxT = (double *)malloc(Ny * Nx * sizeof(double));
		Uy = (double *)malloc(Ny * Nx * sizeof(double));
		Uxx = (double *)malloc(Nx * Ny * sizeof(double));
		UxxT = (double *)malloc(Ny * Nx * sizeof(double));
		Uyy = (double *)malloc(Ny * Nx * sizeof(double));
		F = (double *)malloc(Ny * Nx * sizeof(double));
		G = (double *)malloc(Ny * Nx * sizeof(double));


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
		//cudaMalloc(&d_F, Ny * Nx * sizeof(double));
		cudaMalloc(&d_DxT, Nx * Nx * sizeof(double));
		cudaMalloc(&d_DxxT, Nx * Nx * sizeof(double));
		cudaMalloc(&d_Ux, Nx * Ny * sizeof(double));
		cudaMalloc(&d_UxT, Ny * Nx * sizeof(double));
		cudaMalloc(&d_Uy, Ny * Nx * sizeof(double));
		cudaMalloc(&d_Uxx, Nx * Ny * sizeof(double));
		cudaMalloc(&d_UxxT, Ny * Nx * sizeof(double));
		cudaMalloc(&d_Uyy, Ny * Nx * sizeof(double));
		cudaMalloc(&d_F, Ny * Nx * sizeof(double));
		cudaMalloc(&d_G, Ny * Nx * sizeof(double));

		/* Copy to GPU */
		cudaMemcpy(d_U, h_U, T * Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_B, h_B, T * Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Dx, h_Dx, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Dy, h_Dy, Ny * Ny * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Dxx, h_Dxx, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Dyy, h_Dyy, Ny * Ny * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_V1, h_V1, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_V2, h_V2, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_F, h_F, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_DxT, DxT, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_DxxT, DxxT, Nx * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Ux, Ux, Nx * Ny * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_UxT, UxT, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Uy, Uy, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Uxx, Uxx, Nx * Ny * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_UxxT, Uxx, Nx * Ny * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_Uyy, Uyy, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_F, F, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(d_G, G, Ny * Nx * sizeof(double), cudaMemcpyHostToDevice);

		/* ODE Solver 	*/
		eulerMethod(d_U, d_B, d_V1, d_V2, d_Dx, d_Dy, d_Dxx, d_Dyy, 
			kappa, epsilon, upc, q, alpha, dt, Nx, Ny, T, grid_size, 
			block_size, d_Uy, d_Uyy, d_DxxT, d_DxT, d_Ux, d_Uxx, d_F, d_G);
		
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
		//cudaFree(d_F);
		cudaFree(d_DxT);
		cudaFree(d_DxxT);
		cudaFree(d_Ux);
		cudaFree(d_UxT);
		cudaFree(d_Uy);
		cudaFree(d_Uxx);
		cudaFree(d_UxxT);
		cudaFree(d_Uyy);
		cudaFree(d_F);
		cudaFree(d_G);

		// Free host memory
		free(h_Dx);
		free(h_Dy);
		free(h_Dxx);
		free(h_Dyy);
		//free(h_F);
		free(DxT);
		free(DxxT);
		free(Ux);
		free(UxT);
		free(Uy);
		free(Uxx);
		free(UxxT);
		free(Uyy);
		free(F);
		free(G);
}