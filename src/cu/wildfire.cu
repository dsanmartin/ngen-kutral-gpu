#include <stdio.h>
#include "include/wildfire.cuh"
#include "include/diffmat.cuh"
#include "../c/include/files.h"
#include "../c/include/utils.h"

__constant__ double buffer[256];

/* Gaussian kernel */
__device__ double gaussian(double A, double sigma_x, double sigma_y, double x, double y) {
  return A * exp((x * x) / sigma_x + (y * y) / sigma_y);
}

/* Temperature initial condition */
__device__ double u0(double x, double y) {
  return gaussian(6.0, -20.0, -20.0, x, y);
}

/* Fuel initial condition */
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

__global__ void B0(Parameters parameters, double *B) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < parameters.N * parameters.M) {
    int i = tId % parameters.M; // Row index
    int j = tId / parameters.M; // Col index
    double b = 0;
    if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1))
      b = b0(buffer[j], buffer[parameters.N + i]);
    B[j * parameters.M + i] = b;
  }
}

__device__ double RHSU(Parameters parameters, DiffMats DM, double *Y, int i, int j) {
  /* Get actual value of approximations */
  double u = Y[j * parameters.N + i];
  double b = Y[j * parameters.N + i + parameters.M * parameters.N];

  /* Evaluate vector field */
  double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
  double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
  
  /* Compute derivatives */
  double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
  int m = parameters.M;
  //int n = parameters.N;
  for (int k = 0; k < parameters.N; k++) {
    ux += Y[k * m + i] * DM.Dx[k * m + j];
    uy += DM.Dy[k * m + i] * Y[j * m + k];
    uxx += Y[k * m + i] * DM.Dxx[k * m + j];
    uyy += DM.Dyy[k * m + i] * Y[j * m + k];
  }

  /* Compute PDE */
  double diffusion = parameters.kappa * (uxx + uyy);
  double convection = v_v1 * ux + v_v2 * uy;
  double reaction = f(parameters, u, b);
  return diffusion - convection + reaction;
}

__device__ double RHSB(Parameters parameters, double *Y, int i, int j) {
  double u = Y[j * parameters.N + i - parameters.M * parameters.N];
  double b = Y[j * parameters.N + i];
  return g(parameters, u, b);
}

__device__ float kU(Parameters parameters, DiffMats DM, double *vector, double a, double c, int i, int j) {
  /* Actual u and b */
  double u = vector[j * parameters.N + i] + a;
  double b = vector[j * parameters.N + i + parameters.M * parameters.N] + c;
    
  /* Evaluate vector field */
  double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
  double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
  
  /* Compute derivatives */
  double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
  int m = parameters.M;
  //int n = parameters.N;
  for (int k = 0; k < parameters.N; k++) {
    ux += (vector[k * m + i] + a) * DM.Dx[k * m + j];
    uy += DM.Dy[k * m + i] * (vector[j * m + k] + a);
    uxx += (vector[k * m + i] + a) * DM.Dxx[k * m + j];
    uyy += DM.Dyy[k * m + i] * (vector[j * m + k] + a);
  }

  /* Compute RHS PDE */
  double diffusion = parameters.kappa * (uxx + uyy);
  double convection = v_v1 * ux + v_v2 * uy;
  double reaction = f(parameters, u, b);

  return diffusion - convection + reaction;
}

/* 10.718ms */
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

__global__ void RHSRK4(Parameters parameters, DiffMats DM, double *vector, double dt) {
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

      double u_k1 = kU(parameters, DM, vector, 0.0, 0.0, i, j);
      double b_k1 = g(parameters, u, b);
      double u_k2 = kU(parameters, DM, vector, 0.5 * dt * u_k1, 0.5 * dt * b_k1, i, j);
      double b_k2 = g(parameters, u + 0.5 * dt * u_k1, b + 0.5 * dt * b_k1);
      double u_k3 = kU(parameters, DM, vector, 0.5 * dt * u_k2, 0.5 * dt * b_k2, i, j);
      double b_k3 = g(parameters, u + 0.5 * dt * u_k2, b + 0.5 * dt * b_k2);
      double u_k4 = kU(parameters, DM, vector, dt * u_k3, dt * b_k3, i, j);
      double b_k4 = g(parameters, u + dt * u_k3, b + dt * b_k3);

      u_new = (1.0/6.0) * dt * (u_k1 + 2 * u_k2 + 2 * u_k3 + u_k4);
      b_new = (1.0/6.0) * dt * (b_k1 + 2 * b_k2 + 2 * b_k3 + b_k4);
		}
		/* Update values using Euler method */
    vector[tId] = u + u_new;
    vector[tId + parameters.M * parameters.N] = b + b_new;
  }
}

/* 11.213ms */
__global__ void EulerKernel(Parameters parameters, DiffMats DM, double *Y, double dt) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < 2 * parameters.M * parameters.N) {
    int i = tId % parameters.M; // Row index
    int j = tId / parameters.M; // Col index
    double y_old = Y[tId];
    double y_new = 0; // Boundary condition

    /* Inside domain */
    if (!(i == 0 || i == parameters.M - 1 || i == 2 * parameters.M - 1 || j == 0 || j == parameters.N - 1 || j == 2 * parameters.N - 1)) {
      if (tId < parameters.M * parameters.N) { // For temperature
        y_new = RHSU(parameters, DM, Y, i, j); 
      } else { // For fuel
        y_new = RHSB(parameters, Y, i, j);
      }
    }
    Y[tId] = y_old + dt * y_new;
  }
}

__global__ void sumVector(Parameters parameters, double *c, double *a, double *b, double scalar) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < 2 * parameters.M * parameters.N) {
    c[tId] = a[tId] + scalar * b[tId];
  }
}

__global__ void RK4Kernel(Parameters parameters, DiffMats DM, double *Y, double *k1, double *k2, double *k3, double *k4, double dt) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < 2 * parameters.M * parameters.N) {
    int i = tId % parameters.M; // Row index
    int j = tId / parameters.M; // Col index
    double y_old = Y[tId];
    double y_new = 0; // Boundary condition
    double y_k1 = 0, y_k2 = 0, y_k3 = 0, y_k4 = 0; 
    //printf("%f\n", y_old);

    /* Inside domain */
    if (!(i == 0 || i == parameters.M - 1 || i == 2 * parameters.M - 1 || j == 0 || j == parameters.N - 1 || j == 2 * parameters.N - 1)) {
      if (tId < parameters.M * parameters.N) { // For temperature
        y_k1 = RHSU(parameters, DM, k1, i, j); 
        y_k2 = RHSU(parameters, DM, k2, i, j); 
        y_k3 = RHSU(parameters, DM, k3, i, j); 
        y_k4 = RHSU(parameters, DM, k4, i, j); 
      } else { // For fuel
        y_k1 = RHSB(parameters, k1, i, j);
        y_k2 = RHSB(parameters, k2, i, j);
        y_k3 = RHSB(parameters, k3, i, j);
        y_k4 = RHSB(parameters, k4, i, j);
      }
      //printf("%f %f %f %f\n", y_k1, y_k2, y_k3, y_k4);
      y_new = y_k1 + 2 * y_k2 + 2 * y_k3 + y_k4;
    }
    Y[tId] = y_old + (1.0 / 6.0) * dt * y_new;
  }
}

void ODESolver(Parameters parameters, DiffMats DM, double *d_Y, double dt) {
  int size = 2 * parameters.M * parameters.N;
  int block_size = 256;
  int grid_size = (int) ceil((float)size / block_size);  

  if (strcmp(parameters.time, "Euler") == 0) {
    printf("Euler method in time\n");
    for (int k = 1; k <= parameters.L; k++) { 
      RHSEuler<<<grid_size, block_size>>>(parameters, DM, d_Y, dt);
      //EulerKernel<<<grid_size, block_size>>>(parameters, DM, d_Y, dt);
    }
  } else if (strcmp(parameters.time, "RK4") == 0) {
    printf("RK4 method in time \n");
    double *d_k1, *d_k2, *d_k3, *d_k4;

    cudaMalloc(&d_k1, size * sizeof(double));
    cudaMalloc(&d_k2, size * sizeof(double));
    cudaMalloc(&d_k3, size * sizeof(double));
    cudaMalloc(&d_k4, size * sizeof(double));
    cudaMemset(d_k1, 0, size * sizeof(double));
    cudaMemset(d_k2, 0, size * sizeof(double));
    cudaMemset(d_k3, 0, size * sizeof(double));
    cudaMemset(d_k4, 0, size * sizeof(double));

    for (int k = 1; k <= parameters.L; k++) { 
      //RHSRK4<<<grid_size, block_size>>>(parameters, DM, d_Y, dt);  
      sumVector<<<grid_size, block_size>>>(parameters, d_k1, d_Y, d_k1, 0);
      cudaDeviceSynchronize();
      sumVector<<<grid_size, block_size>>>(parameters, d_k2, d_Y, d_k1, 0.5 * dt);
      cudaDeviceSynchronize();
      sumVector<<<grid_size, block_size>>>(parameters, d_k3, d_Y, d_k2, 0.5 * dt);
      cudaDeviceSynchronize();
      sumVector<<<grid_size, block_size>>>(parameters, d_k4, d_Y, d_k3, dt);
      cudaDeviceSynchronize();
      RK4Kernel<<<grid_size, block_size>>>(parameters, DM, d_Y, d_k1, d_k2, d_k3, d_k4, dt);   
    }

    cudaFree(d_k1);
    cudaFree(d_k2);
    cudaFree(d_k3);
    cudaFree(d_k4);
  }
}


void wildfire(Parameters parameters) {
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

  printf("dx: %f\n", dx);
  printf("dy: %f\n", dy);
  printf("dt: %f\n", dt);
  
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
  double *d_U, *d_B, *d_Y, *d_Dx, *d_Dy, *d_Dxx, *d_Dyy;
  
  DiffMats DM;
  
  /* Fill spatial domain vectors */
  fillVector(x, dx, parameters.N);
  fillVector(y, dy, parameters.M);

  /* Random fuel */
	//randomArray(h_B, parameters.M,  parameters.N);

  /* Device memory allocation */
  cudaMalloc(&d_U, size * sizeof(double));
  cudaMalloc(&d_B, size * sizeof(double));
  cudaMalloc(&d_Dx, parameters.N * parameters.N * sizeof(double));
  cudaMalloc(&d_Dy, parameters.M * parameters.M * sizeof(double));
  cudaMalloc(&d_Dxx, parameters.N * parameters.N * sizeof(double));
  cudaMalloc(&d_Dyy, parameters.M * parameters.M * sizeof(double));
  cudaMalloc(&d_Y,  2 * parameters.M * parameters.N * sizeof(double));

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
  B0<<<grid_size, block_size>>>(parameters, d_B); // Fuel
  cudaDeviceSynchronize();

  /* Save initial conditions */
  cudaMemcpy(h_U, d_U, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, size * sizeof(double), cudaMemcpyDeviceToHost);
  saveApproximation(strcat(strcat(U0_save, parameters.sim_name), ".txt"), h_U, parameters.N, parameters.M, 1);
  saveApproximation(strcat(strcat(B0_save, parameters.sim_name), ".txt"), h_B, parameters.N, parameters.M, 1);
  
  /* Copy initial conditions to vector */
  cudaMemcpy(d_Y, h_U, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_Y + size, h_B, size * sizeof(double), cudaMemcpyHostToDevice);

  /* ODE Integration */
  ODESolver(parameters, DM, d_Y, dt);

  cudaDeviceSynchronize();

  /* Copy approximations to host */
  cudaMemcpy(h_U, d_Y, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_Y + size, size * sizeof(double), cudaMemcpyDeviceToHost);

  /* Save  */
  saveApproximation(strcat(strcat(U_save, parameters.sim_name), ".txt"), h_U, parameters.N, parameters.M, 1);
  saveApproximation(strcat(strcat(B_save, parameters.sim_name), ".txt"), h_B, parameters.N, parameters.M, 1);

  /* Memory free */
  cudaFree(d_Y);
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

