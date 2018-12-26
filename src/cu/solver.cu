#include <stdio.h>
#include "include/solver.cuh"
#include "include/diffmat.cuh"
#include "../c/include/files.h"
#include "../c/include/utils.h"

__constant__ double buffer[256];

__device__ double gaussian(double A, double sigma_x, double sigma_y, double x, double y) {
  return A * exp((x * x) / sigma_x + (y * y) / sigma_y);
}
__device__ double u0(double x, double y) {
  return gaussian(6.0, -20.0, -20.0, x, y);
}

__device__ double b0(double x, double y) {
  return 1;
}

__device__ double v1(double x, double y) {
  return 0.70710678118;
}

__device__ double v2(double x, double y) {
  return 0.70710678118;
}

__device__ double s(double u, double upc) {
  return u >= upc;
}

__device__ double f(Parameters parameters, double u, double b) {
	return s(u, parameters.upc) * b * exp(u / (1 + parameters.epsilon * u)) - parameters.alpha * u;
}

__device__ double g(Parameters parameters, double u, double b) {
	return -s(u, parameters.upc) * (parameters.epsilon / parameters.q) * b * exp(u /(1 + parameters.epsilon * u));
}

__global__ void U0(Parameters parameters, double *U) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < parameters.N * parameters.M) {
    int i = tId % parameters.M; // Row index
    int j = tId / parameters.M; // Col index
    U[j * parameters.M + i] = u0(buffer[j] + parameters.x_ign, buffer[parameters.N + i] + parameters.y_ign);
  }
}

__global__ void B0(double *B, int N, int M) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < N * M) {
    int i = tId % M; // Row index
    int j = tId / M; // Col index
    B[j * M + i] = b0(buffer[j], buffer[N + i]);
  }
}

__global__ void RHSEuler(Parameters parameters, DiffMats DM, double *vector, double dt) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < parameters.M * parameters.N) {
    int i = tId % parameters.M; // Row index
    int j = tId / parameters.M; // Col index
    double u_new = 0; // Boundary conditions
		double b_new = 0; // Boundary conditions

		/* Get actual value of approximations */
		double u = vector[j * parameters.N + i];
		double b = vector[j * parameters.N + i + parameters.M * parameters.N];

		/* PDE */
    if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) {
			
			/* Evaluate vector field */
      double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
			double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
			
			/* Compute derivatives */
      double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
      int m = parameters.M;
      //int n = parameters.N;
      for (int k = 0; k < parameters.N; k++) {
        ux += vector[k * m + i] * DM.Dx[k * m + j];
        uy += DM.Dy[k * m + i] * vector[j * m + k];
        uxx += vector[k * m + i] * DM.Dxx[k * m + j];
        uyy += DM.Dyy[k * m + i] * vector[j * m + k];
      }

			/* Compute PDE */
      double diffusion = parameters.kappa * (uxx + uyy);
      double convection = v_v1 * ux + v_v2 * uy;
      double reaction = f(parameters, u, b);
      double fuel = g(parameters, u, b);
      u_new = diffusion - convection + reaction;
      b_new = fuel;
		}
		/* Update values using Euler method */
    vector[tId] = u + dt * u_new;
    vector[tId + parameters.M * parameters.N] = b + dt * b_new;
  }
}

void ODESolver(double *U, double *B, DiffMats DM, Parameters parameters, double dt) {
  int size = parameters.M * parameters.N;
  int block_size = 256;
  int grid_size = (int) ceil((float)size / block_size);

  /* Host vector */
	double *h_y = (double *) malloc(2 * size * sizeof(double));

  /* Device vectors */
  double *d_y;

  /* Device memory allocation */
  cudaMalloc(&d_y, 2 * size * sizeof(double));

  /* Copy initial conditions to vector */
  cudaMemcpy(d_y, U, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y + size, B, size * sizeof(double), cudaMemcpyHostToDevice);
  
  for (int k = 1; k <= parameters.L; k++) { 
    RHSEuler<<<grid_size, block_size>>>(parameters, DM, d_y, dt);   
  }

  /* Save last approximation value */
  cudaMemcpy(U, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(B, d_y + size, size * sizeof(double), cudaMemcpyDeviceToHost);

  /* Memory free */
  cudaFree(d_y);
  free(h_y);
}


void solver(Parameters parameters) {
  /* Kernel parameters */
	int size = parameters.M * parameters.N;
	int block_size = 256;
  int grid_size = (int) ceil((float)size / block_size);

  /* Simulation name */
  char U0_save[20] = "test/output/U0_";
  char B0_save[20] = "test/output/B0_";
  char U_save[20] = "test/output/U_";
  char B_save[20] = "test/output/B_";
  

  /* Domain differentials */
	double dx = (parameters.x_max - parameters.x_min) / (parameters.N-1);
	double dy = (parameters.y_max - parameters.y_min) / (parameters.M-1);
  double dt = parameters.t_max / parameters.L;
  
  /* Domain vectors */
  double *x = (double *) malloc(parameters.N * sizeof(double));
  double *y = (double *) malloc(parameters.M * sizeof(double));

  /* Temperature approximation */
  double *h_U = (double *) malloc(size * sizeof(double));
  double *h_B = (double *) malloc(size * sizeof(double));

  /* Differentiation Matrices */
  double *h_Dx = (double *) malloc(parameters.N * parameters.N * sizeof(double));
  double *h_Dxx = (double *) malloc(parameters.N * parameters.N * sizeof(double));
  double *h_Dy = (double *) malloc(parameters.M * parameters.M * sizeof(double));
  double *h_Dyy = (double *) malloc(parameters.M * parameters.M * sizeof(double));

  /* Device arrays */
  double *d_U, *d_B, *d_Dx, *d_Dy, *d_Dxx, *d_Dyy;
  
  DiffMats DM;
  
  /* Fill spatial domain vectors */
  fillVector(x, dx, parameters.N);
  fillVector(y, dy, parameters.M);

  /* Random fuel */
	randomArray(h_B, parameters.M,  parameters.N);

  /* Device memory allocation */
  cudaMalloc(&d_U, size * sizeof(double));
  cudaMalloc(&d_B, size * sizeof(double));
  cudaMalloc(&d_Dx, parameters.N * parameters.N * sizeof(double));
  cudaMalloc(&d_Dy, parameters.M * parameters.M * sizeof(double));
  cudaMalloc(&d_Dxx, parameters.N * parameters.N * sizeof(double));
  cudaMalloc(&d_Dyy, parameters.M * parameters.M * sizeof(double));

  /* Copy from host to device */
  cudaMemcpy(d_U, h_U, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Dx, h_Dx, parameters.N * parameters.N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Dy, h_Dy, parameters.M * parameters.M * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Dxx, h_Dxx, parameters.N * parameters.N * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Dyy, h_Dyy, parameters.M * parameters.M * sizeof(double), cudaMemcpyHostToDevice);

  /* Copy spatial domain to constant memory */
  cudaMemcpyToSymbol(buffer, x, parameters.N * sizeof(double), 0, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(buffer, y, parameters.M * sizeof(double), parameters.N * sizeof(double), cudaMemcpyHostToDevice);

  /* Fill differentiation matrices */
  FD1Kernel<<<grid_size, block_size>>>(d_Dx, parameters.N, dx);
  FD1Kernel<<<grid_size, block_size>>>(d_Dy, parameters.M, dy);
  FD2Kernel<<<grid_size, block_size>>>(d_Dxx, parameters.N, dx);
  FD2Kernel<<<grid_size, block_size>>>(d_Dyy, parameters.M, dy);
  DM.Dx = d_Dx;
  DM.Dy = d_Dy;
  DM.Dxx = d_Dxx;
	DM.Dyy = d_Dyy;

  /* Compute initial contitions */
  U0<<<grid_size, block_size>>>(parameters, d_U); // Temperature
  //B0<<<grid_size, block_size>>>(d_B, parameters.N, parameters.M); // Fuel

  cudaDeviceSynchronize();

  /* Save initial conditions */
  cudaMemcpy(h_U, d_U, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, size * sizeof(double), cudaMemcpyDeviceToHost);
  saveApproximation(strcat(strcat(U0_save, parameters.sim_name), ".txt"), h_U, parameters.N, parameters.M, 1);
  saveApproximation(strcat(strcat(B0_save, parameters.sim_name), ".txt"), h_B, parameters.N, parameters.M, 1);
  
  /* ODE Integration */
  ODESolver(d_U, d_B, DM, parameters, dt);

  /* Copy approximations to host */
  cudaMemcpy(h_U, d_U, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, size * sizeof(double), cudaMemcpyDeviceToHost);

  /* Save  */
  saveApproximation(strcat(strcat(U_save, parameters.sim_name), ".txt"), h_U, parameters.N, parameters.M, 1);
  saveApproximation(strcat(strcat(B_save, parameters.sim_name), ".txt"), h_B, parameters.N, parameters.M, 1);

  /* Memory free */
  cudaFree(d_U);
  cudaFree(d_B);
  cudaFree(d_Dx);
  cudaFree(d_Dy);
  cudaFree(d_Dxx);
  cudaFree(d_Dyy);
  free(x);
  free(y);
  free(h_U);
  free(h_B);
  free(h_Dx);
  free(h_Dy);
  free(h_Dxx);
  free(h_Dyy);
}

