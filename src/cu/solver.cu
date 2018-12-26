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
  return 1;//(double) rand() / (double)RAND_MAX ;
}

__device__ double v1(double x, double y) {
  return 0.70710678118;//3.0;
}

__device__ double v2(double x, double y) {
  return 0.70710678118;//3.0;
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

__global__ void U0(double *U, int N, int M) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < N * M) {
    int i = tId % M; // Row index
    int j = tId / M; // Col index
    U[j * M + i] = u0(buffer[j] - 20.0, buffer[N + i] - 20.0);
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

__global__ void RHS(Parameters parameters, DiffMats DM, double *vector) {
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

      // double dx = (parameters.x_max - parameters.x_min) / (parameters.N-1);
	    // double dy = (parameters.y_max - parameters.y_min) / (parameters.M-1);
      // double ux = (vector[(j+1) * m + i] - vector[(j-1) * m + i]) / (2*dx);
      // double uxx = (vector[(j+1) * m + i] - 2 * vector[j * m + i] + vector[(j-1) * m + i]) / (dx * dx);
      // double uy = (vector[j * m + i + 1] - vector[j * m + i - 1]) / (2*dy);
      // double uyy = (vector[j * m + i + 1] - 2 * vector[j * m + i] + vector[j * m + i - 1]) / (dy * dy);
			/* Compute PDE */
      double diffusion = parameters.kappa * (uxx + uyy);
      double convection = v_v1 * ux + v_v2 * uy;
      double reaction = f(parameters, u, b);
      double fuel = g(parameters, u, b);
      u_new = diffusion - convection + reaction;
      b_new = fuel;
		}
		/* Update values */
    vector[tId] = u_new;
		vector[tId + parameters.M * parameters.N] = b_new;
		// old_vector[tId] = u;
		// old_vector[tId + parameters.M * parameters.N] = b;
  }
}

__global__ void EulerKernel(double *y, double *y_old, double dt, int size) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < size) {
    double u_old = y_old[tId];
    double b_old = y_old[size + tId];
    double u_new = y[tId];    
    double b_new = y[size + tId];
    y[tId] = u_old + dt * u_new;
    y[tId + size] = b_old + dt * b_new;
  }
}

__global__ void RK4Kernel(double *y, double *y_old, double dt, int size) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < size) {
    double u_old = y_old[tId];
    double b_old = y_old[size + tId];
    double u_k1 = y[tId];    
    double u_k2 = u_old + 0.5 * dt * u_k1;
    double u_k3 = u_old + 0.5 * dt * u_k2;
    double u_k4 = u_old + dt * u_k3;
    double b_k1 = y[size + tId];
    double b_k2 = b_old + 0.5 * dt * b_k1;
    double b_k3 = b_old + 0.5 * dt * b_k2;
    double b_k4 = b_old + dt * b_k3;
    // y[tId] = y_tmp + (1.0/6.0) * dt * (k1 + 2 * k2 + 2 * k3 + k4);
    y[tId] = u_old + (1.0/6.0) * dt * (u_k1 + 2 * u_k2 + 2 * u_k3 + u_k4);
    y[tId + size] = b_old + (1.0/6.0) * dt * (b_k1 + 2 * b_k2 + 2 * b_k3 + b_k4);
  }
}

__global__ void RHS2(Parameters parameters, DiffMats DM, double *vector, double dt) {
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
		/* Update values */
    vector[tId] = u + dt * u_new;
    vector[tId + parameters.M * parameters.N] = b + dt * b_new;
  }
}

// __global__ void RHS3(Parameters parameters, DiffMats DM, double *vector, double dt) {
//   int tId = threadIdx.x + blockIdx.x * blockDim.x;
//   if (tId < parameters.M * parameters.N) {
//     int i = tId % parameters.M; // Row index
//     int j = tId / parameters.M; // Col index
//     double u_new = 0; // Boundary conditions
// 		double b_new = 0; // Boundary conditions

// 		/* Get actual value of approximations */
// 		double u = vector[j * parameters.N + i];
// 		double b = vector[j * parameters.N + i + parameters.M * parameters.N];
// 		double u1 = vector[j * parameters.N + i];
// 		double b1 = vector[j * parameters.N + i + parameters.M * parameters.N];
// 		double u2 = vector[j * parameters.N + i];
// 		double b2 = vector[j * parameters.N + i + parameters.M * parameters.N];
// 		double u3 = vector[j * parameters.N + i];
// 		double b3 = vector[j * parameters.N + i + parameters.M * parameters.N];
// 		double u4 = vector[j * parameters.N + i];
// 		double b4 = vector[j * parameters.N + i + parameters.M * parameters.N];

// 		/* PDE */
//     if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) {
			
// 			/* Evaluate vector field */
//       double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
// 			double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
			
// 			/* Compute derivatives */
// 			double ux1 = 0.0, uy1 = 0.0, uxx1 = 0.0, uyy1 = 0.0;
// 			double ux2 = 0.0, uy2 = 0.0, uxx2 = 0.0, uyy2 = 0.0;
// 			double ux3 = 0.0, uy3 = 0.0, uxx3 = 0.0, uyy3 = 0.0;
// 			double ux4 = 0.0, uy4 = 0.0, uxx4 = 0.0, uyy4 = 0.0;
//       int m = parameters.M;
//       for (int k = 0; k < parameters.N; k++) {
// 				// k1
//         ux1 += vector[k * m + i] * DM.Dx[k * m + j];
//         uy1 += DM.Dy[k * m + i] * vector[j * m + k];
//         uxx1 += vector[k * m + i] * DM.Dxx[k * m + j];
// 				uyy1 += DM.Dyy[k * m + i] * vector[j * m + k];
// 				// k2
// 				ux2 += (vector[k * m + i] + dt * DM.Dx[k * m + j];
//         uy2 += DM.Dy[k * m + i] * vector[j * m + k];
//         uxx2 += vector[k * m + i] * DM.Dxx[k * m + j];
// 				uyy2 += DM.Dyy[k * m + i] * vector[j * m + k];
// 				// k3
// 				ux3 += vector[k * m + i] * DM.Dx[k * m + j];
//         uy3 += DM.Dy[k * m + i] * vector[j * m + k];
//         uxx3 += vector[k * m + i] * DM.Dxx[k * m + j];
// 				uyy3 += DM.Dyy[k * m + i] * vector[j * m + k];
// 				// k4
// 				ux4 += vector[k * m + i] * DM.Dx[k * m + j];
//         uy4 += DM.Dy[k * m + i] * vector[j * m + k];
//         uxx4 += vector[k * m + i] * DM.Dxx[k * m + j];
//         uyy4 += DM.Dyy[k * m + i] * vector[j * m + k];
//       }

// 			/* Compute PDE */
//       double diffusion = parameters.kappa * (uxx + uyy);
//       double convection = v_v1 * ux + v_v2 * uy;
//       double reaction = f(parameters, u, b);
//       double fuel = g(parameters, u, b);
//       u_new = diffusion - convection + reaction;
//       b_new = fuel;
// 		}
// 		/* Update values */
//     vector[tId] = u + dt * u_new;
//     vector[tId + parameters.M * parameters.N] = b + dt * b_new;
//   }
// }

__global__ void pibarrathebest(double *challa, int M, int N) {
  int tId = threadIdx.x + blockIdx.x * blockDim.x;
  if (tId < M + N) {
    challa[tId] = buffer[tId];
  }
}

void ODESolver(double *U, double *B, DiffMats DM, Parameters parameters, double dt) {
  int size = parameters.M * parameters.N;
  int block_size = 256;
  int grid_size = (int) ceil((float)size / block_size);

  /* Host vector */
	double *h_y = (double *) malloc(2 * size * sizeof(double));
	double *h_U = (double *) malloc(size * sizeof(double));
	double *h_B = (double *) malloc(size * sizeof(double));

  /* Device vectors */
  double *d_y, *d_y_tmp;

  /* Device memory allocation */
  cudaMalloc(&d_y, 2 * size * sizeof(double));
  cudaMalloc(&d_y_tmp, 2 * size * sizeof(double));

  /* Copy initial conditions to vector */
  cudaMemcpy(d_y, U, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y + size, B, size * sizeof(double), cudaMemcpyHostToDevice);
  //cudaMemcpy(d_y_tmp, U, size * sizeof(double), cudaMemcpyHostToDevice);
	//cudaMemcpy(d_y_tmp + size, B, size * sizeof(double), cudaMemcpyHostToDevice);
	
	// cudaMemcpy(h_U, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_B, d_y + size, size * sizeof(double), cudaMemcpyDeviceToHost);
	
	// saveApproximation("test/output/Utest.txt", h_U, parameters.N, parameters.M, 1);
	// saveApproximation("test/output/Btest.txt", h_B, parameters.N, parameters.M, 1);
  
  for (int k = 1; k <= parameters.L; k++) { 
    RHS2<<<grid_size, block_size>>>(parameters, DM, d_y, dt);   
		//cudaMemcpy(d_y_tmp, d_y, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice);
		//cudaDeviceSynchronize();
		//RHS<<<grid_size, block_size>>>(parameters, DM, d_y, d_y_tmp);
    //cudaDeviceSynchronize();
    //RK4Kernel<<<grid_size, block_size>>>(d_y, d_y_tmp, dt, size);
    //EulerKernel<<<grid_size, block_size>>>(d_y, d_y_tmp, dt, size);
    //cudaMemcpy(d_y_tmp, d_y, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  //cudaMemcpy(h_y, d_y, 2 * size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(U, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(B, d_y + size, size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_y);
  cudaFree(d_y_tmp);
  free(h_y);
}

void ODESolver2(double *U, double *B, DiffMats DM, Parameters parameters, double dt) {
  int size = parameters.M * parameters.N;
  int block_size = 256;
  int grid_size = (int) ceil((float)size / block_size);

  /* Host vector */
	double *h_y = (double *) malloc(2 * size * sizeof(double));
	double *h_k1 = (double *) malloc(2 * size * sizeof(double));
	double *h_k2 = (double *) malloc(2 * size * sizeof(double));
	double *h_k3 = (double *) malloc(2 * size * sizeof(double));
	double *h_k4 = (double *) malloc(2 * size * sizeof(double));

  /* Device vectors */
  double *d_y, *d_k1, *d_k2, *d_k3, *d_k4;

  /* Device memory allocation */
  cudaMalloc(&d_y, 2 * size * sizeof(double));
	cudaMalloc(&d_k1, 2 * size * sizeof(double));
	cudaMalloc(&d_k2, 2 * size * sizeof(double));
	cudaMalloc(&d_k3, 2 * size * sizeof(double));
	cudaMalloc(&d_k4, 2 * size * sizeof(double));

  /* Copy initial conditions to vector */
  cudaMemcpy(d_y, U, size * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y + size, B, size * sizeof(double), cudaMemcpyHostToDevice);
  
  for (int k = 1; k <= parameters.L; k++) { 
    RHS<<<grid_size, block_size>>>(parameters, DM, d_y, dt);   
		//cudaMemcpy(d_y_tmp, d_y, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice);
		//cudaDeviceSynchronize();
		//RHS<<<grid_size, block_size>>>(parameters, DM, d_y, d_y_tmp);
    //cudaDeviceSynchronize();
    //RK4Kernel<<<grid_size, block_size>>>(d_y, d_y_tmp, dt, size);
    //EulerKernel<<<grid_size, block_size>>>(d_y, d_y_tmp, dt, size);
    //cudaMemcpy(d_y_tmp, d_y, 2 * size * sizeof(double), cudaMemcpyDeviceToDevice);
  }

  //cudaMemcpy(h_y, d_y, 2 * size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(U, d_y, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(B, d_y + size, size * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_y);
	cudaFree(d_k1);
	cudaFree(d_k2);
	cudaFree(d_k3);
	cudaFree(d_k4);
  free(h_y);
}

void solver(Parameters parameters) {
  /* Kernel parameters */
	int size = parameters.M * parameters.N;
	int block_size = 256;
  int grid_size = (int) ceil((float)size / block_size);
  
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

	randomArray(h_B, parameters.M * parameters.N);

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
	
	// cudaMemcpy(h_Dx, d_Dx, size * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_Dxx, d_Dxx, size * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_Dy, d_Dy, size * sizeof(double), cudaMemcpyDeviceToHost);
	// cudaMemcpy(h_Dyy, d_Dyy, size * sizeof(double), cudaMemcpyDeviceToHost);
	// printMatrix(h_Dy, parameters.M, parameters.N);
	// printMatrix(h_Dyy, parameters.M, parameters.N);
	// printMatrix(h_Dx, parameters.M, parameters.N);
	// printMatrix(h_Dxx, parameters.M, parameters.N);

  /* Compute initial contitions */
  U0<<<grid_size, block_size>>>(d_U, parameters.N, parameters.M); // Temperature
  //B0<<<grid_size, block_size>>>(d_B, parameters.N, parameters.M); // Fuel

  cudaDeviceSynchronize();

  /* Save initial conditions */
  cudaMemcpy(h_U, d_U, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, size * sizeof(double), cudaMemcpyDeviceToHost);
  saveApproximation("test/output/U0.txt", h_U, parameters.N, parameters.M, 1);
  saveApproximation("test/output/B0.txt", h_B, parameters.N, parameters.M, 1);
  
  /* ODE Integration */
  ODESolver(d_U, d_B, DM, parameters, dt);

  /* Copy approximations to host */
  cudaMemcpy(h_U, d_U, size * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, size * sizeof(double), cudaMemcpyDeviceToHost);

  /* Save  */
  saveApproximation("test/output/Uaa.txt", h_U, parameters.N, parameters.M, 1);
  saveApproximation("test/output/Baa.txt", h_B, parameters.N, parameters.M, 1);

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

