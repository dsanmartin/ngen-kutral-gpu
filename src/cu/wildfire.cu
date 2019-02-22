#include <stdio.h>
#include "include/wildfire.cuh"
#include "include/diffmat.cuh"
#include "include/utils.cuh"
#include "../c/include/files.h"
#include "../c/include/utils.h"

#define DB 256 // Threads per block
#define DG(size) (size + DB - 1) / DB // Blocks per grid

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

__global__ void U0(Parameters parameters, double *U, double x_ign, double y_ign) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < parameters.N * parameters.M) {
		int i = tId % parameters.M; // Row index
		int j = tId / parameters.M; // Col index
		double u = 0; // Boundary condition
		if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1))
			u = u0(buffer[j] - x_ign, buffer[parameters.N + i] - y_ign);
		U[j * parameters.M + i] = u;
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
	double u = Y[j * parameters.M + i];
	double b = Y[j * parameters.M + i + parameters.M * parameters.N];

	/* Evaluate vector field */
	double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
	double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
	
	/* Compute derivatives */
	double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
	int m = parameters.M;
	int n = parameters.N;
	for (int k = 0; k < parameters.N; k++) {
		ux += Y[k * m + i] * DM.Dx[k * n + j];
		uy += DM.Dy[k * m + i] * Y[j * m + k];
		uxx += Y[k * m + i] * DM.Dxx[k * n + j];
		uyy += DM.Dyy[k * m + i] * Y[j * m + k];
	}

	/* Compute PDE */
	double diffusion = parameters.kappa * (uxx + uyy);
	double convection = v_v1 * ux + v_v2 * uy;
	double reaction = f(parameters, u, b);
	return diffusion - convection + reaction;
}

__device__ double RHSB(Parameters parameters, double *Y, int i, int j) {
	double u = Y[j * parameters.M + i];
	double b = Y[j * parameters.M + i + parameters.M * parameters.N];
	return g(parameters, u, b);
}

__global__ void simulationBlock(Parameters parameters, DiffMats DM, double *Y, double *Y_old, double dt) {
	int sim = blockIdx.x;
	int index = threadIdx.x;
	int offset = 2 * sim  * parameters.M * parameters.N;
	for (int k = 1; k <= parameters.L; k++) { 

		for (int nodes = 0; nodes <= 2 * parameters.M * parameters.N; nodes++) 
			Y_old[nodes] = Y[offset + nodes];

		while (index < parameters.M * parameters.N) {
			int i = index % parameters.M; // Row index
			int j = index / parameters.M; // Col index
			double u_new = 0; // Boundary conditions
			double b_new = 0; // Boundary conditions
			
			int gindex = offset + j * parameters.M + i;

			/* Get actual value of approximations */
			double u_old = Y_old[gindex];
			double b_old = Y_old[gindex + parameters.M * parameters.N];

			/* PDE */
			if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) { // Inside domain
				double fuel = g(parameters, u_old, b_old);
				u_new = RHSU(parameters, DM, Y_old + offset, i, j);
				b_new = fuel;
			}
			/* Update values using Euler method */
			Y[gindex] = u_old + dt * u_new;
			Y[gindex + parameters.M * parameters.N] = b_old + dt * b_new;
			index += blockDim.x;
		}
		__syncthreads();
	}
}

/*
	Right hand side using Euler method.
	This approach use all threads to compute each node of all simulations.
	Kernel time: 67.005ms 
*/
__global__ void RHSEuler(Parameters parameters, DiffMats DM, double *Y, double *Y_old, double dt) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	int n_sim = parameters.x_ign_n * parameters.y_ign_n;
  if (tId < n_sim * parameters.M * parameters.N) {
		int sim = tId / (parameters.M * parameters.N);
		int i = (tId - sim * parameters.M * parameters.N) % parameters.M; // Row index
		int j = (tId - sim * parameters.M * parameters.N) / parameters.M; // Col index
    double u_new = 0; // Boundary conditions
		double b_new = 0; // Boundary conditions
		int offset = 2 * sim  * parameters.M * parameters.N;
		int gindex = offset + j * parameters.M + i;

		/* Get actual value of approximations */
		double u_old = Y_old[gindex];
		double b_old = Y_old[gindex + parameters.M * parameters.N];

		/* PDE */
    if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) { // Inside domain
      double fuel = g(parameters, u_old, b_old);
      u_new = RHSU(parameters, DM, Y_old + offset, i, j);
      b_new = fuel;
		}

		/* Update values using Euler method */
    Y[gindex] = u_old + dt * u_new;
		Y[gindex + parameters.M * parameters.N] = b_old + dt * b_new;
  }
}

/*
	Right hand side using Euler method.
	This approach use each block for a single simulation.
	Kernel time: 54.407ms 
*/
__global__ void RHSEulerBlock(Parameters parameters, DiffMats DM, double *Y, double *Y_old, double dt) {
	int sim = blockIdx.x;
	int index = threadIdx.x;
  while (index < parameters.M * parameters.N) {
		int i = index % parameters.M; // Row index
		int j = index / parameters.M; // Col index
    double u_new = 0; // Boundary conditions
		double b_new = 0; // Boundary conditions
		int offset = 2 * sim  * parameters.M * parameters.N;
		int gindex = offset + j * parameters.M + i;

		/* Get actual value of approximations */
		double u_old = Y_old[gindex];
		double b_old = Y_old[gindex + parameters.M * parameters.N];

		/* PDE */
    if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) { // Inside domain
      double fuel = g(parameters, u_old, b_old);
      u_new = RHSU(parameters, DM, Y_old + offset, i, j);
      b_new = fuel;
		}
		/* Update values using Euler method */
    Y[gindex] = u_old + dt * u_new;
		Y[gindex + parameters.M * parameters.N] = b_old + dt * b_new;
		index += blockDim.x;
	}
}

__global__ void sumVector(Parameters parameters, double *c, double *a, double *b, double scalar, int size) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	if (tId < size) {
		c[tId] = a[tId] + scalar * b[tId];
	}
}

/* Compute RHS using all threads 43.5ms */
__global__ void RHSvec(Parameters parameters, DiffMats DM, double *k, double *vec) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	int n_sim = parameters.x_ign_n * parameters.y_ign_n;
	if (tId < n_sim * parameters.M * parameters.N) {
		int sim = tId / (parameters.M * parameters.N);
		int i = (tId - sim * parameters.M * parameters.N) % parameters.M; // Row index
		int j = (tId - sim * parameters.M * parameters.N) / parameters.M; // Col index
		int offset = 2 * sim  * parameters.M * parameters.N;
		int gindex = offset + j * parameters.M + i;

		int u_index = gindex;//j * parameters.M + i;
		int b_index = parameters.M * parameters.N + u_index;
		double u_k = 0;
		double b_k = 0;
		if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) {
			u_k = RHSU(parameters, DM, vec + offset, i, j);
			b_k = g(parameters, vec[u_index], vec[b_index]);
		}
		k[u_index] = u_k;
		k[b_index] = b_k;
	}
}

/* Compute RHS using a block per simulation */
__global__ void RHSvecBlock(Parameters parameters, DiffMats DM, double *k, double *vec) {
	int sim = blockIdx.x;
	int index = threadIdx.x;
	while (index < parameters.M * parameters.N) {
		int i = index % parameters.M; // Row index
		int j = index / parameters.M; // Col index
		int offset = 2 * sim  * parameters.M * parameters.N;
		int gindex = offset + j * parameters.M + i;

		int u_index = gindex;
		int b_index = parameters.M * parameters.N + u_index;
		double u_k = 0;
		double b_k = 0;
		if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) {
			u_k = RHSU(parameters, DM, vec + offset, i, j);
			b_k = g(parameters, vec[u_index], vec[b_index]);
		}
		k[u_index] = u_k;
		k[b_index] = b_k;
		index += blockDim.x;
	}
}

/*
	Right hand side using RK4 method.
	This approach use all threads to compute each node of all simulations.
	Kernel time: 1.8705ms 
*/
__global__ void RHSRK4(Parameters parameters, DiffMats DM, double *Y, double *Y_old, 
	double *k1, double *k2, double *k3, double *k4, double dt) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	int n_sim = parameters.x_ign_n * parameters.y_ign_n;
  if (tId < n_sim * parameters.M * parameters.N) {
		int sim = tId / (parameters.M * parameters.N);
		int i = (tId - sim * parameters.M * parameters.N) % parameters.M; // Row index
		int j = (tId - sim * parameters.M * parameters.N) / parameters.M; // Col index
    double u_new = 0; // Boundary conditions
		double b_new = 0; // Boundary conditions
		int offset = 2 * sim  * parameters.M * parameters.N;
		int gindex = offset + j * parameters.M + i;

		/* Get actual value of approximations */
		double u_old = Y_old[gindex];
		double b_old = Y_old[gindex + parameters.M * parameters.N];
		double u_k1 = 0, u_k2 = 0, u_k3 = 0, u_k4 = 0;
		double b_k1 = 0, b_k2 = 0, b_k3 = 0, b_k4 = 0; 

		/* PDE */
    if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) { // Inside domain
			u_k1 = k1[gindex];
			u_k2 = k2[gindex];
			u_k3 = k3[gindex];
			u_k4 = k4[gindex];
			b_k1 = k1[gindex + parameters.M * parameters.N];
			b_k2 = k2[gindex + parameters.M * parameters.N];
			b_k3 = k3[gindex + parameters.M * parameters.N];
			b_k4 = k4[gindex + parameters.M * parameters.N];
			u_new = u_k1 + 2 * u_k2 + 2 * u_k3 + u_k4;
			b_new = b_k1 + 2 * b_k2 + 2 * b_k3 + b_k4;
		}

		/* Update values using RK4 method */
    Y[gindex] = u_old + (1.0 / 6.0) * dt * u_new;
		Y[gindex + parameters.M * parameters.N] = b_old + (1.0 / 6.0) * dt * b_new;
  }
}

/*
	Right hand side using RK4 method.
	This approach use each block for a single simulation.
	Kernel time: 2.642ms
*/
__global__ void RHSRK4Block(Parameters parameters, DiffMats DM, double *Y, double *Y_old,
	double *k1, double *k2, double *k3, double *k4, double dt) {
	int sim = blockIdx.x;
	int index = threadIdx.x;
  while (index < parameters.M * parameters.N) {
		int i = index % parameters.M; // Row index
		int j = index / parameters.M; // Col index
    double u_new = 0; // Boundary conditions
		double b_new = 0; // Boundary conditions
		int offset = 2 * sim  * parameters.M * parameters.N;
		int gindex = offset + j * parameters.M + i;

		/* Get actual value of approximations */
		double u_old = Y_old[gindex];
		double b_old = Y_old[gindex + parameters.M * parameters.N];
		double u_k1 = 0, u_k2 = 0, u_k3 = 0, u_k4 = 0;
		double b_k1 = 0, b_k2 = 0, b_k3 = 0, b_k4 = 0; 

		/* PDE */
    if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) { // Inside domain
      u_k1 = k1[gindex];
			u_k2 = k2[gindex];
			u_k3 = k3[gindex];
			u_k4 = k4[gindex];
			b_k1 = k1[gindex + parameters.M * parameters.N];
			b_k2 = k2[gindex + parameters.M * parameters.N];
			b_k3 = k3[gindex + parameters.M * parameters.N];
			b_k4 = k4[gindex + parameters.M * parameters.N];
			u_new = u_k1 + 2 * u_k2 + 2 * u_k3 + u_k4;
			b_new = b_k1 + 2 * b_k2 + 2 * b_k3 + b_k4;
		}

		/* Update values using RK4 method */
    Y[gindex] = u_old + (1.0 / 6.0) * dt * u_new;
		Y[gindex + parameters.M * parameters.N] = b_old + (1.0 / 6.0) * dt * b_new;
		index += blockDim.x;
	}
	__syncthreads();
}

void ODESolver(Parameters parameters, DiffMats DM, double *d_Y, double dt) {
	int n_sim = parameters.x_ign_n * parameters.y_ign_n; // Number of wildfire simulations
	int size = 2 * n_sim * parameters.M * parameters.N;

	/* Time method */
	if (strcmp(parameters.time, "Euler") == 0) {

		printf("Euler in time\n");
		printf("dt: %f\n", dt);

		/* Temporal array for previous time step */
		double *d_Y_tmp;
		cudaMalloc(&d_Y_tmp, size * sizeof(double));

		/* GPU parallel approach */
		if (strcmp(parameters.approach, "all") == 0) {
			for (int k = 1; k <= parameters.L; k++) { 
				cudaMemcpy(d_Y_tmp, d_Y, size * sizeof(double), cudaMemcpyDeviceToDevice);
				RHSEuler<<<DG(size), DB>>>(parameters, DM, d_Y, d_Y_tmp, dt);
			}
		} else if (strcmp(parameters.approach, "block") == 0) {
			for (int k = 1; k <= parameters.L; k++) { 
				cudaMemcpy(d_Y_tmp, d_Y, size * sizeof(double), cudaMemcpyDeviceToDevice);
				RHSEulerBlock<<<n_sim, DB>>>(parameters, DM, d_Y, d_Y_tmp, dt);
			}
		}

		cudaFree(d_Y_tmp);

	} else if (strcmp(parameters.time, "RK4") == 0) {

		printf("RK4 in time \n");
		printf("dt: %f\n", dt);

		/* Temporal arrays for previous ks */
		double *d_Y_tmp, *d_k1, *d_k2, *d_k3, *d_k4, *d_ktmp;

		cudaMalloc(&d_k1, size * sizeof(double));
		cudaMalloc(&d_k2, size * sizeof(double));
		cudaMalloc(&d_k3, size * sizeof(double));
		cudaMalloc(&d_k4, size * sizeof(double));
		cudaMalloc(&d_ktmp, size * sizeof(double));
		cudaMalloc(&d_Y_tmp, size * sizeof(double));
		cudaMemset(d_k1, 0, size * sizeof(double));
		cudaMemset(d_k2, 0, size * sizeof(double));
		cudaMemset(d_k3, 0, size * sizeof(double));
		cudaMemset(d_k4, 0, size * sizeof(double));
		cudaMemset(d_ktmp, 0, size * sizeof(double));

		/* GPU parallel approach */
		if (strcmp(parameters.approach, "all") == 0) {
			for (int k = 1; k <= parameters.L; k++) { 
				cudaMemcpy(d_Y_tmp, d_Y, size * sizeof(double), cudaMemcpyDeviceToDevice); // Y_{t-1}
				RHSvec<<<DG(size), DB>>>(parameters, DM, d_k1, d_Y_tmp); // Compute k1
				sumVector<<<DG(size), DB>>>(parameters, d_ktmp, d_Y_tmp, d_k1, 0.5*dt, size); // Y_{t-1} + 0.5*dt*k1
				RHSvec<<<DG(size), DB>>>(parameters, DM, d_k2, d_ktmp); // Compute k2
				sumVector<<<DG(size), DB>>>(parameters, d_ktmp, d_Y_tmp, d_k2, 0.5 * dt, size); // Y_{t-1} + 0.5*dt*k2
				RHSvec<<<DG(size), DB>>>(parameters, DM, d_k3, d_ktmp); // Compute k3
				sumVector<<<DG(size), DB>>>(parameters, d_ktmp, d_Y_tmp, d_k3, dt, size); // Y_{t-1} + dt*k3
				RHSvec<<<DG(size), DB>>>(parameters, DM, d_k4, d_ktmp); // Compute k4
				RHSRK4<<<DG(size), DB>>>(parameters, DM, d_Y, d_Y_tmp, d_k1, d_k2, d_k3, d_k4, dt); // RK4
			}
		} else if (strcmp(parameters.approach, "block") == 0) {
			for (int k = 1; k <= parameters.L; k++) { 
				cudaMemcpy(d_Y_tmp, d_Y, size * sizeof(double), cudaMemcpyDeviceToDevice); // Y_{t-1}
				RHSvecBlock<<<n_sim, DB>>>(parameters, DM, d_k1, d_Y_tmp); // Compute k1
				//RHSvec<<<DG(size), DB>>>(parameters, DM, d_k1, d_Y_tmp);
				cudaDeviceSynchronize();
				sumVector<<<DG(size), DB>>>(parameters, d_ktmp, d_Y_tmp, d_k1, 0.5*dt, size); // Y_{t-1} + 0.5*dt*k1
				//cudaDeviceSynchronize();
				RHSvecBlock<<<n_sim, DB>>>(parameters, DM, d_k2, d_ktmp); // Compute k2
				//RHSvec<<<DG(size), DB>>>(parameters, DM, d_k2, d_ktmp);
				cudaDeviceSynchronize();
				sumVector<<<DG(size), DB>>>(parameters, d_ktmp, d_Y_tmp, d_k2, 0.5 * dt, size); // Y_{t-1} + 0.5*dt*k2
				//cudaDeviceSynchronize();
				RHSvecBlock<<<n_sim, DB>>>(parameters, DM, d_k3, d_ktmp); // Compute k3
				//RHSvec<<<DG(size), DB>>>(parameters, DM, d_k3, d_ktmp);
				cudaDeviceSynchronize();
				sumVector<<<DG(size), DB>>>(parameters, d_ktmp, d_Y_tmp, d_k3, dt, size); // Y_{t-1} + dt*k3
				//cudaDeviceSynchronize();
				RHSvecBlock<<<n_sim, DB>>>(parameters, DM, d_k4, d_ktmp); // Compute k4
				//RHSvec<<<DG(size), DB>>>(parameters, DM, d_k4, d_ktmp);
				cudaDeviceSynchronize();
				RHSRK4Block<<<n_sim, DB>>>(parameters, DM, d_Y, d_Y_tmp, d_k1, d_k2, d_k3, d_k4, dt); // RK4
				//cudaDeviceSynchronize();
			}
		}

		cudaFree(d_Y_tmp);
		cudaFree(d_k1);
		cudaFree(d_k2);
		cudaFree(d_k3);
		cudaFree(d_k4);
		cudaFree(d_ktmp);
	}
}

void fillInitialConditions(Parameters parameters, double *d_sims, int save) {
	/* Initial wildfire focus */
	double dx_ign, dy_ign, x_ign, y_ign;

	if (parameters.x_ign_n * parameters.y_ign_n > 1) {
		dx_ign = (parameters.x_ign_max - parameters.x_ign_min) / (parameters.x_ign_n - 1);
		dy_ign = (parameters.y_ign_max - parameters.y_ign_min) / (parameters.y_ign_n - 1);	
	} else {
		dx_ign = 1;
		dy_ign = 1;
	}

	/* To save IC */
	char sim_name[40];

	/* Temporal arrays */
	double *d_tmp;
	double *h_tmp = (double *) malloc(parameters.M * parameters.N * sizeof(double));

	cudaMalloc(&d_tmp, parameters.M * parameters.N * sizeof(double));
	cudaMemset(d_tmp, 0, parameters.M * parameters.N  * sizeof(double));
	
	/* Fill initial conditions according to ignitions points */
	int sim_ = 0;
	for (int i=0; i < parameters.y_ign_n; i++) {
		for (int j=0; j < parameters.x_ign_n; j++) {		

			/* Coordinates of ignition point */
			x_ign = parameters.x_ign_min + dx_ign * j;	
			y_ign = parameters.y_ign_min + dy_ign * i;

			/* Compute initial condition for temperature */
			U0<<<DG(parameters.M * parameters.N), DB>>>(parameters, d_tmp, x_ign, y_ign);
			cudaMemcpy(d_sims + 2*sim_*parameters.M*parameters.N, 
				d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
			
			/* Save temperature IC */
			if (save) {
				cudaMemcpy(h_tmp, d_sims + 2*sim_*parameters.M*parameters.N, 
					parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToHost);
				memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
				sprintf(sim_name, "%s/%s0_%d%d.txt", parameters.dir, "U", i, j); // Simulation name
				//sprintf(sim_name, "test/output/%s0_%d%d.txt", "U", i, j); // Simulation name
				saveApproximation(sim_name, h_tmp, parameters.M, parameters.N); // Save U0
			}
			
			/* Compute initial condition for fuel */
			B0<<<DG(parameters.M * parameters.N), DB>>>(parameters, d_tmp);
			cudaMemcpy(d_sims + (2*sim_+1) * parameters.M * parameters.N, 
				d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
			
			/* Save fuel IC */
			if (save) {
				cudaMemcpy(h_tmp, d_sims + (2*sim_+1) * parameters.M * parameters.N, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToHost);
				memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
				sprintf(sim_name, "%s/%s0_%d%d.txt", parameters.dir, "B", i, j); // Simulation name
				//sprintf(sim_name, "test/output/%s0_%d%d.txt", "B", i, j); // Simulation name
				saveApproximation(sim_name, h_tmp, parameters.M, parameters.N);	// Save B0	 
			}		

			sim_++;		
		}		
	}

	/* Free memory */
	cudaFree(d_tmp);
	free(h_tmp);
}

void saveResults(Parameters parameters, double *h_sims) {
	/* Simulation name */
	char sim_name[40];

	/* Temporal array */
	double *h_tmp = (double *) malloc(parameters.M * parameters.N * sizeof(double));

	int sim_ = 0;
	for (int i=0; i < parameters.y_ign_n; i++) {
		for (int j=0; j < parameters.x_ign_n; j++) {	

			/* Temperature */
			memcpy(h_tmp, h_sims + 2*sim_*parameters.M*parameters.N, 
				parameters.M * parameters.N * sizeof(double));
			memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
			sprintf(sim_name, "%s/%s_%d%d.txt", parameters.dir, "U", i, j); // Simulation name
			//sprintf(sim_name, "test/output/%s_%d%d.txt", "U", i, j); // Simulation name
			saveApproximation(sim_name, h_tmp, parameters.M, parameters.N); // Save U

			/* Fuel */
			memcpy(h_tmp, h_sims + (2*sim_ + 1)*parameters.M*parameters.N, 
				parameters.M * parameters.N * sizeof(double));
			memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
			sprintf(sim_name, "%s/%s_%d%d.txt", parameters.dir,"B", i, j); // Simulation name
			//sprintf(sim_name, "test/output/%s_%d%d.txt", "B", i, j); // Simulation name
			saveApproximation(sim_name, h_tmp, parameters.M, parameters.N);	// Save B	

			sim_++;
		}
	}
}

void wildfire(Parameters parameters) {

	/* Log file with parameters info */
	char log_name[100];
	sprintf(log_name, "%s/log.txt", parameters.dir);
	FILE *log = fopen(log_name, "w");

	/* Kernel Parameters */
	int n_sim = parameters.x_ign_n * parameters.y_ign_n; // Number of wildfire simulations
	int size = 2 * n_sim * parameters.M * parameters.N;

	/* Domain differentials */
	double dx = (parameters.x_max - parameters.x_min) / (parameters.N-1);
	double dy = (parameters.y_max - parameters.y_min) / (parameters.M-1);
	double dt = parameters.t_max / parameters.L;

	/* Memory for simulations */
	double *h_sims = (double *) malloc(size * sizeof(double));
	double *d_sims;	
	
	/* Domain vectors */
	double *h_x = (double *) malloc(parameters.N * sizeof(double));
	double *h_y = (double *) malloc(parameters.M * sizeof(double));
	double *d_x, *d_y;

	/* Write parameters in log */
	fprintf(log, "Simulation ID: %s\n", parameters.sim_id);
	fprintf(log, "Number of numerical simulations: %d\n", parameters.x_ign_n * parameters.y_ign_n);
	fprintf(log, "Parallel approach: %s\n", parameters.approach);
	fprintf(log, "\nIgnition points\n");
	fprintf(log, "----------------\n");
	fprintf(log, "%d in x, %d in y\n", parameters.x_ign_n, parameters.y_ign_n);
	fprintf(log, "Domain: [%f, %f]x[%f, %f]\n", parameters.x_ign_min, parameters.x_ign_max,
	parameters.y_ign_min, parameters.y_ign_max);
	fprintf(log, "\nSpace\n");
	fprintf(log, "------\n");	
	fprintf(log, "Domain: [%f, %f]x[%f, %f]\n", parameters.x_min, parameters.x_max, 
		parameters.y_min, parameters.y_max);
	fprintf(log, "Method: %s\n", parameters.spatial);
	fprintf(log, "M: %d\n", parameters.M);
	fprintf(log, "N: %d\n", parameters.N);
	fprintf(log, "dx: %f\n", dx);
	fprintf(log, "dy: %f\n", dy);
	fprintf(log, "\nTime\n");
	fprintf(log, "------\n");	
	fprintf(log, "Domain: [0, %f]\n", parameters.t_max);
	fprintf(log, "Method: %s\n", parameters.time);
	fprintf(log, "L: %d\n", parameters.L);
	fprintf(log, "dt: %f\n", dt);		
	fclose(log);

	/* Differentiation Matrices */
	
	// Struct for matrices
	DiffMats DM; 

	// Host arrays
	double *h_Dx = (double *) malloc(parameters.N * parameters.N * sizeof(double));
	double *h_Dxx = (double *) malloc(parameters.N * parameters.N * sizeof(double));
	double *h_Dy = (double *) malloc(parameters.M * parameters.M * sizeof(double));
	double *h_Dyy = (double *) malloc(parameters.M * parameters.M * sizeof(double));

	// Device arrays
	double *d_Dx, *d_Dy, *d_Dxx, *d_Dyy;

	/* Device memory allocation */
	cudaMalloc(&d_sims, size * sizeof(double));
	cudaMalloc(&d_x, parameters.N * sizeof(double));
	cudaMalloc(&d_y, parameters.M * sizeof(double));
	cudaMalloc(&d_Dx, parameters.N * parameters.N * sizeof(double));
	cudaMalloc(&d_Dy, parameters.M * parameters.M * sizeof(double));
	cudaMalloc(&d_Dxx, parameters.N * parameters.N * sizeof(double));
	cudaMalloc(&d_Dyy, parameters.M * parameters.M * sizeof(double));

	/* Copy from host to device */
	cudaMemcpy(d_sims, h_sims, size * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, parameters.N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, parameters.M * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dx, h_Dx, parameters.N * parameters.N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dy, h_Dy, parameters.M * parameters.M * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dxx, h_Dxx, parameters.N * parameters.N * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Dyy, h_Dyy, parameters.M * parameters.M * sizeof(double), cudaMemcpyHostToDevice);

	/* Fill spatial domain and differentiation matrices */
	if (strcmp(parameters.spatial, "FD") == 0) {		
		// Spatial domain
		printf("Finite Difference in space\n");
		printf("dx: %f\n", dx);
		printf("dy: %f\n", dy);	

		// fprintf(log, "Finite Difference in space\n");
		// fprintf(log, "dx: %f\n", dx);
		// fprintf(log, "dy: %f\n", dy);	

		fillVectorKernel<<<DG(parameters.M * parameters.N), DB>>>(d_x, dx, parameters.N);
		fillVectorKernel<<<DG(parameters.M * parameters.N), DB>>>(d_y, dy, parameters.M);

		// Differentiation matrices
		FD1Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dx, parameters.N, dx);
		FD1Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dy, parameters.M, dy);
		FD2Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dxx, parameters.N, dx);
		FD2Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dyy, parameters.M, dy);

	} else if (strcmp(parameters.spatial, "Cheb") == 0) {
		// Spatial domain
		printf("Chebyshev in space\n");
		//fprintf(log, "Chebyshev in space\n");

		ChebyshevNodes<<<DG(parameters.M * parameters.N), DB>>>(d_x, parameters.N - 1);
		ChebyshevNodes<<<DG(parameters.M * parameters.N), DB>>>(d_y, parameters.M - 1);

		// Differentiation matrices
		ChebyshevMatrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dx, d_x, parameters.N - 1);
		ChebyshevMatrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dy, d_y, parameters.M - 1);
		Chebyshev2Matrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dxx, d_Dx, parameters.N - 1);
		Chebyshev2Matrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dyy, d_Dy, parameters.M - 1);

	} else {
		printf("Spatial domain error\n");
		exit(0);
	}	
	
	/* Copy spatial domain to constant memory */
	cudaMemcpyToSymbol(buffer, d_x, parameters.N * sizeof(double), 0, cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(buffer, d_y, parameters.M * sizeof(double), parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
	
	DM.Dx = d_Dx;
	DM.Dy = d_Dy;
	DM.Dxx = d_Dxx;
	DM.Dyy = d_Dyy;

	/* Fill initial conditions */	
	fillInitialConditions(parameters, d_sims, 1);

	/* ODE Integration */
	ODESolver(parameters, DM, d_sims, dt);
	// double *d_tmp;
	// cudaMalloc(&d_tmp, 2 * parameters.M * parameters.N * sizeof(double));
	// cudaMemset(d_tmp, 0, 2 * parameters.M * parameters.N  * sizeof(double));

	// simulationBlock<<<n_sim, DB, DB * sizeof(double)>>>(parameters, DM, d_sims, d_tmp, dt);

	//cudaDeviceSynchronize();

	/* Copy approximations to host */
	cudaMemcpy(h_sims, d_sims, size * sizeof(double), cudaMemcpyDeviceToHost);

	/* Save */
	saveResults(parameters, h_sims);

	/* Memory free */
	cudaFree(d_sims);
	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_Dx);
	cudaFree(d_Dy);
	cudaFree(d_Dxx);
	cudaFree(d_Dyy);
	free(h_sims);
	free(h_x);
	free(h_y);
	free(h_Dx);
	free(h_Dy);
	free(h_Dxx);
	free(h_Dyy);
}