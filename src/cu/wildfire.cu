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
			u = u0(buffer[j] + x_ign, buffer[parameters.N + i] + y_ign);
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

__device__ double RHSU2(Parameters parameters, DiffMats DM, double *Y, double u, double b, int i, int j, int id) {
	/* Get actual value of approximations */
	int offset = (id < parameters.N) ? 0 : parameters.N;
	// double u = Y[j * parameters.M + i];
	// double b = Y[j * parameters.M + i + parameters.M * parameters.N];

	/* Evaluate vector field */
	double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
	double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
	
	/* Compute derivatives */
	double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
	int m = parameters.M;
	int n = parameters.N;
	
	for (int k = 0; k < parameters.N; k++) {
		ux += Y[offset + k] * DM.Dx[k * n + j];
		uy += DM.Dy[k * m + i] * Y[offset + k];
		uxx += Y[offset + k] * DM.Dxx[k * n + j];
		uyy += DM.Dyy[k * m + i] * Y[offset + k];
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

__device__ double RHSU3(Parameters parameters, DiffMats DM, double *Y, int i, int j, int offset) {
	/* Get actual value of approximations */
	double u = Y[offset + j * parameters.M + i];
	double b = Y[offset + j * parameters.M + i + parameters.M * parameters.N];

	/* Evaluate vector field */
	double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
	double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
	
	/* Compute derivatives */
	double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
	int m = parameters.M;
	int n = parameters.N;
	for (int k = 0; k < parameters.N; k++) {
		ux += Y[offset + k * m + i] * DM.Dx[k * n + j];
		uy += DM.Dy[k * m + i] * Y[offset + j * m + k];
		uxx += Y[offset + k * m + i] * DM.Dxx[k * n + j];
		uyy += DM.Dyy[k * m + i] * Y[offset + j * m + k];
	}

	/* Compute PDE */
	double diffusion = parameters.kappa * (uxx + uyy);
	double convection = v_v1 * ux + v_v2 * uy;
	double reaction = f(parameters, u, b);
	return diffusion - convection + reaction;
}

__device__ double RHSB3(Parameters parameters, double *Y, int i, int j, int offset) {
	double u = Y[offset + j * parameters.M + i];
	double b = Y[offset + j * parameters.M + i + parameters.M * parameters.N];
	return g(parameters, u, b);
}

__global__ void simulation(Parameters parameters, DiffMats DM, double *Y, double dt) {	
	int N_exps = parameters.x_ign_n * parameters.y_ign_n;
	if (blockIdx.x < N_exps) {
		int idx = threadIdx.x; //blockDim.x * blockIdx.x + threadIdx.x;
		for (int k = 1; k <= parameters.L; k++) { 
			int i = idx % parameters.M; // Row index
			int j = idx / parameters.M; // Col index
			double y_old = Y[idx];
			double y_new = 0; // Boundary condition

			/* Inside domain */
			if (!(i == 0 || i == parameters.M - 1 || i == 2 * parameters.M - 1 || j == 0 || j == parameters.N - 1 || j == 2 * parameters.N - 1)) {
				if (idx < parameters.M * parameters.N) { // For temperature
					y_new = RHSU(parameters, DM, Y, i, j); 
				} else { // For fuel 
					y_new = RHSB(parameters, Y, i, j);
				}
			}
			Y[idx] = y_old + dt * y_new;
		}
	}
}

// __global__ void simulation(Parameters parameters, DiffMats DM, double *Y, double dt) {	
// 	int N_exps = parameters.x_ign_n * parameters.y_ign_n;
// 	if (blockIdx.x < N_exps) {
// 		int idx = theadblockDim.x * blockIdx.x + threadIdx.x;
// 		for (int k = 1; k <= parameters.L; k++) { 
// 			int i = idx % parameters.M; // Row index
// 			int j = idx / parameters.M; // Col index
// 			double y_old = Y[idx];
// 			double y_new = 0; // Boundary condition

// 			/* Inside domain */
// 			if (!(i == 0 || i == parameters.M - 1 || i == 2 * parameters.M - 1 || j == 0 || j == parameters.N - 1 || j == 2 * parameters.N - 1)) {
// 				if (idx < parameters.M * parameters.N) { // For temperature
// 					y_new = RHSU(parameters, DM, Y, i, j); 
// 				} else { // For fuel 
// 					y_new = RHSB(parameters, Y, i, j);
// 				}
// 			}
// 			Y[idx] = y_old + dt * y_new;
// 		}
// 	}
// }

__global__ void simulation2(Parameters parameters, DiffMats DM, double *Y, double dt) {
	extern __shared__ double sm[];
	//int loop = (parameters.M * parameters.N) / blockDim.x;
	int loop = 0;
	for (int k = 0; k < parameters.L; k++) {
		//for (int sub = 0; sub < loop; sub++) {
		while (loop < parameters.M * parameters.N) {
			//int index = blockDim.x * sub + threadIdx.x;
			int index = threadIdx.x + loop;
	
			int i = index % parameters.M; // Row index
			int j = index / parameters.M; // Col index
			int gindex = blockIdx.x * 2 * parameters.M * parameters.N + j * parameters.M + i;

			sm[threadIdx.x] = Y[blockIdx.x * 2 * parameters.M * parameters.N + i * parameters.M + j];
			//sm[tId] = Y[i * parameters.M + j];
			double u_old = Y[gindex];
			double b_old = Y[gindex + parameters.M * parameters.N];
			double u_new = 0; // Boundary condition
			double b_new = 0; // Boundary condition
			//double u_new_2 = 0; // Boundary condition
			//double b_new_2 = 0; // Boundary condition

			// if(blockIdx.x == 0 && sub == 0 && k == 0){
			// 	for (int asd = 2*128*128; asd < 4*parameters.M * parameters.N; asd++) {
			// 		if (Y[asd] > 0)
			// 			printf("%f\n",Y[asd]);
			// 	}
			// 	//printf("%f %d %d %d %d\n",sm[threadIdx.x], i, j, i * parameters.M + j, blockIdx.x * 2 * parameters.M * parameters.N + i * parameters.M + j);
			// }

			/* Inside domain */
			if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) {
				u_new = RHSU2(parameters, DM, sm, u_old, b_old, i, j, threadIdx.x); 
				b_new = g(parameters, u_old, b_old);//RHSB(parameters, sm, i, j);

			}
			//u_new = 1;
			Y[gindex] = u_old + dt * u_new;
			Y[gindex + parameters.M * parameters.N] = b_old + dt * b_new;
			loop += threadIdx.x;
		}
		__syncthreads();
	}
}

__global__ void simulation3(Parameters parameters, DiffMats DM, double *Y, double *Y_old, double dt, int size) {
	int tId = threadIdx.x + blockIdx.x * blockDim.x;
	//size = parameters.M * parameters.N;
	if (tId < size) {
		int sim = tId / (2 * parameters.M * parameters.N);
		int i = tId % parameters.M;//(tId - sim * 2 * parameters.M * parameters.N) % (parameters.M); // Row index
		int j = tId / parameters.M;//(tId - sim * 2 * parameters.M * parameters.N) / (parameters.M); // Col index
		// int i = (tId - sim * 2 * parameters.M * parameters.N) % parameters.M; // Row index
		// int j = (tId - sim * 2 * parameters.M * parameters.N) / (2 * parameters.M); // Col index
		double y_old = Y_old[tId];
		double y_new = 0; // Boundary condition
		if (sim > 0) printf("sim: %d\n", sim);

		/* Inside domain */
		if (!(i == 0 || i == parameters.M - 1 || i == 2 * parameters.M - 1 || j == 0 || j == parameters.N - 1 || j == 2 * parameters.N - 1)) {
		//if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) {
			if (tId < parameters.M * parameters.N) { // For temperature
				y_new = RHSU(parameters, DM, Y_old, i, j); 

			} else { // For fuel
				y_new = RHSB(parameters, Y_old, i, j);
			}
			//if (tId > 2 * sim * parameters.M * parameters.N && tId < (2 * sim + 1) * parameters.N * parameters.M) { // For temperature
			//if (i < parameters.M && j < parameters.M) {
			// 	y_new = RHSU(parameters, DM, Y_old + sim * 2 * parameters.M * parameters.N, i, j);
			// 	//y_new = RHSU3(parameters, DM, Y_old, i, j, sim * 2 * parameters.M * parameters.N);
			// } else { // For fuel
			// 	y_new = RHSB(parameters, Y_old + sim * 2 * parameters.M * parameters.N, i, j);
			// 	//y_new = RHSB3(parameters, Y_old, i, j, sim * 2 * parameters.M * parameters.N);
			// }
			//printf("yold: %f, ynew: %f\n", y_old, y_new);
		}
		Y[tId] = y_old  + dt * y_new;
		//__syncthreads();
		//Y[tId] = y_new;
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
			
			/* Evaluate vector field */
      double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
			double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
			
			/* Compute derivatives */
      double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
      int m = parameters.M;
      int n = parameters.N;
      for (int k = 0; k < parameters.N; k++) {
        ux += Y_old[offset + k * m + i] * DM.Dx[k * n + j];
        uy += DM.Dy[k * m + i] * Y_old[offset + j * m + k];
        uxx += Y_old[offset + k * m + i] * DM.Dxx[k * n + j];
        uyy += DM.Dyy[k * m + i] * Y_old[offset + j * m + k];				
      }

			/* Compute PDE */
      double diffusion = parameters.kappa * (uxx + uyy);
      double convection = v_v1 * ux + v_v2 * uy;
      double reaction = f(parameters, u_old, b_old);
      double fuel = g(parameters, u_old, b_old);
      u_new = diffusion - convection + reaction;
      b_new = fuel;
		}
		/* Update values using Euler method */
    Y[gindex] = u_old + dt * u_new;
		Y[gindex + parameters.M * parameters.N] = b_old + dt * b_new;
  }
}

/*
	Right hand side using Euler method.
	This approach use all threads to compute each node of all simulations.
	Kernel time: 54.407ms 
*/
__global__ void RHSEulerBlock(Parameters parameters, DiffMats DM, double *Y, double *Y_old, double dt) {
	//int tId = threadIdx.x;// + blockIdx.x * blockDim.x;
	//int n_sim = parameters.x_ign_n * parameters.y_ign_n;
	int sim = blockIdx.x;
	//int loop = 0;
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
			
			/* Evaluate vector field */
      double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
			double v_v2 = v2(buffer[j], buffer[parameters.N + i]);  
			
			/* Compute derivatives */
      double ux = 0.0, uy = 0.0, uxx = 0.0, uyy = 0.0;
      int m = parameters.M;
      int n = parameters.N;
      for (int k = 0; k < parameters.N; k++) {
        ux += Y_old[offset + k * m + i] * DM.Dx[k * n + j];
        uy += DM.Dy[k * m + i] * Y_old[offset + j * m + k];
        uxx += Y_old[offset + k * m + i] * DM.Dxx[k * n + j];
        uyy += DM.Dyy[k * m + i] * Y_old[offset + j * m + k];				
      }

			/* Compute PDE */
      double diffusion = parameters.kappa * (uxx + uyy);
      double convection = v_v1 * ux + v_v2 * uy;
      double reaction = f(parameters, u_old, b_old);
      double fuel = g(parameters, u_old, b_old);
      u_new = diffusion - convection + reaction;
      b_new = fuel;
		}
		/* Update values using Euler method */
    Y[gindex] = u_old + dt * u_new;
		Y[gindex + parameters.M * parameters.N] = b_old + dt * b_new;
		index += blockDim.x;
	}
	__syncthreads();
}

void ODESolver(Parameters parameters, DiffMats DM, double *d_Y, double dt) {
	int n_sim = parameters.x_ign_n * parameters.y_ign_n; // Number of wildfire simulations
	int size = 2 * n_sim * parameters.M * parameters.N;

	double *d_Y_tmp;
	cudaMalloc(&d_Y_tmp, size * sizeof(double));

	for (int k = 1; k <= parameters.L; k++) { 
		cudaMemcpy(d_Y_tmp, d_Y, size * sizeof(double), cudaMemcpyDeviceToDevice);
		//simulation2<<<DG(size), DB>>>(parameters, DM, d_Y, d_Y_tmp, dt);
		//simulation3<<<DG(size), DB>>>(parameters, DM, d_Y, d_Y_tmp, dt, size);
		//RHSEuler<<<DG(size), DB>>>(parameters, DM, d_Y, d_Y_tmp, dt);
		RHSEulerBlock<<<n_sim, DB>>>(parameters, DM, d_Y, d_Y_tmp, dt);
	}

	cudaFree(d_Y_tmp);
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
	//double dx_ign = (parameters.x_ign_max - parameters.x_ign_min) / (parameters.x_ign_n - 1);
	//double dy_ign = (parameters.y_ign_max - parameters.y_ign_min) / (parameters.y_ign_n - 1);	
	//double x_ign, y_ign;

	/* To save IC */
	char sim_name[25];

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
			// cudaMemcpy(d_sims + (2 * parameters.M * parameters.N * (j * parameters.y_ign_n + i)), 
			// 	d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_sims + 2*sim_*parameters.M*parameters.N, 
				d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
			
			/* Save temperature IC */
			if (save) {
				// cudaMemcpy(h_tmp, d_sims + (2 * parameters.M * parameters.N * (j * parameters.y_ign_n + i)), 
				// 	parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_tmp, d_sims + 2*sim_*parameters.M*parameters.N, 
					parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToHost);
				memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
				sprintf(sim_name, "test/output/%s0_%d%d.txt", "U", i, j); // Simulation name
				saveApproximation(sim_name, h_tmp, parameters.M, parameters.N); // Save U0
			}
			
			/* Compute initial condition for fuel */
			B0<<<DG(parameters.M * parameters.N), DB>>>(parameters, d_tmp);
			// cudaMemcpy(d_sims + parameters.M * parameters.N + (parameters.M * parameters.N * (j*parameters.y_ign_n + i)), 
			// 	d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_sims + (2*sim_+1) * parameters.M * parameters.N, 
				d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
			
			/* Save fuel IC */
			if (save) {
				//cudaMemcpy(h_tmp, d_sims + parameters.M * parameters.N + (parameters.M * parameters.N * (j*parameters.y_ign_n + i)), parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToHost);
				cudaMemcpy(h_tmp, d_sims + (2*sim_+1) * parameters.M * parameters.N, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToHost);

				memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
				sprintf(sim_name, "test/output/%s0_%d%d.txt", "B", i, j); // Simulation name
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
	char sim_name[25];

	/* Temporal array */
	double *h_tmp = (double *) malloc(parameters.M * parameters.N * sizeof(double));

	int sim_ = 0;
	for (int i=0; i < parameters.y_ign_n; i++) {
		for (int j=0; j < parameters.x_ign_n; j++) {	

			/* Temperature */
			// memcpy(h_tmp, h_sims + (2 * parameters.M * parameters.N * (j * parameters.y_ign_n + i)), 
			// 	parameters.M * parameters.N * sizeof(double));
			memcpy(h_tmp, h_sims + 2*sim_*parameters.M*parameters.N, 
				parameters.M * parameters.N * sizeof(double));
			memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
			sprintf(sim_name, "test/output/%s_%d%d.txt", "U", i, j); // Simulation name
			saveApproximation(sim_name, h_tmp, parameters.M, parameters.N); // Save U

			/* Fuel */
			// memcpy(h_tmp, h_sims + parameters.M * parameters.N + (parameters.M * parameters.N * (j*parameters.y_ign_n + i)), 
			// 	parameters.M * parameters.N * sizeof(double));
			memcpy(h_tmp, h_sims + (2*sim_ + 1)*parameters.M*parameters.N, 
				parameters.M * parameters.N * sizeof(double));
			memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
			sprintf(sim_name, "test/output/%s_%d%d.txt", "B", i, j); // Simulation name
			saveApproximation(sim_name, h_tmp, parameters.M, parameters.N);	// Save B	

			sim_++;
		}
	}
}

void wildfire(Parameters parameters) {

	/* Kernel Parameters */
	int n_sim = parameters.x_ign_n * parameters.y_ign_n; // Number of wildfire simulations
	int size = 2 * n_sim * parameters.M * parameters.N;
	//int grid_size = (int) ceil((float)size / THREADS_PER_BLOCK);
	//int grid_size_DM = (parameters.M * parameters.N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	/* Domain differentials */
	double dx = (parameters.x_max - parameters.x_min) / (parameters.N-1);
	double dy = (parameters.y_max - parameters.y_min) / (parameters.M-1);
	double dt = parameters.t_max / parameters.L;

	printf("dx: %f\n", dx);
	printf("dy: %f\n", dy);
	printf("dt: %f\n", dt);
	printf("Simulations size: %d\n", size);
	//printf("Size: %d\n", DG(size));

	/* Memory for simulations */
	double *h_sims = (double *) malloc(size * sizeof(double));
	double *d_sims;	
	
	/* Domain vectors */
	double *h_x = (double *) malloc(parameters.N * sizeof(double));
	double *h_y = (double *) malloc(parameters.M * sizeof(double));
	double *d_x, *d_y;

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

	/* Fill spatial domain */
	fillVectorKernel<<<DG(parameters.M * parameters.N), DB>>>(d_x, dx, parameters.N);
	fillVectorKernel<<<DG(parameters.M * parameters.N), DB>>>(d_y, dy, parameters.N);

	/* Copy spatial domain to constant memory */
	cudaMemcpyToSymbol(buffer, d_x, parameters.N * sizeof(double), 0, cudaMemcpyDeviceToDevice);
	cudaMemcpyToSymbol(buffer, d_y, parameters.M * sizeof(double), parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);

	/* Fill differentiation matrices */
	FD1Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dx, parameters.N, dx);
	FD1Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dy, parameters.M, dy);
	FD2Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dxx, parameters.N, dx);
	FD2Kernel<<<DG(parameters.M * parameters.N), DB>>>(d_Dyy, parameters.M, dy);
	DM.Dx = d_Dx;
	DM.Dy = d_Dy;
	DM.Dxx = d_Dxx;
	DM.Dyy = d_Dyy;

	/* Fill initial conditions */	
	fillInitialConditions(parameters, d_sims, 1);

	// /* ODE Integration */
	ODESolver(parameters, DM, d_sims, dt);
	//simulation2<<<100, DB, DB * sizeof(double)>>>(parameters, DM, d_sims, dt);

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

