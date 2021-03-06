/**
 * @file wildfire.cu
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Wildfire simulations solver for GPU
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */
 
#include <stdio.h>
#include <time.h>
#include "include/wildfire.cuh"
#include "include/diffmat.cuh"
#include "include/utils.cuh"
#include "../c/include/files.h"
#include "../c/include/utils.h"

#define DB parameters.threads// Threads per block
#define DG parameters.blocks // Blocks per grid

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

/**
 * @brief Vector field first component
 * 
 * @param x \f$ x \f$ value
 * @param y \f$ y \f$ value
 */
__device__ double v1(double x, double y) {
    return 0.70710678118;
}

/**
 * @brief Vector field second component
 * 
 * @param x \f$ x \f$ value
 * @param y \f$ y \f$ value
 */
__device__ double v2(double x, double y) {
    return 0.70710678118;
}

/**
 * @brief Temprature initial condition
 * 
 * @param parameters Model and numerical parameters
 * @param U Pointer to temperature array
 * @param x_ign x coordinate of ignition point
 * @param y_ign y coordinate of ignition point
 */
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

/**
 * @brief Fuel initial condition
 * 
 * @param parameters Model and numerical parameters
 * @param B Pointer to fuel array
 */
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

/**
 * @brief PDE RHS computation
 * 
 * @param parameters Model an numerical parameters
 * @param DM Differentiation matrix structure 
 * @param k Pointer to new array
 * @param vec Pointer to old array
 */
__global__ void RHSvec(Parameters parameters, DiffMats DM, double *k, double *vec) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int n_sim = parameters.x_ign_n * parameters.y_ign_n;
    while (tId < n_sim * parameters.M * parameters.N) {
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

            // Get actual value of approximations
            double u = vec[u_index];
            double b = vec[b_index];

            // Evaluate vector field 
            double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
            double v_v2 = v2(buffer[j], buffer[parameters.N + i]);

            // Get stencil 
            double u_r = vec[offset + (j+1) * parameters.M + i];
            double u_l = vec[offset + (j-1) * parameters.M + i];
            double u_u = vec[offset + j * parameters.M + i + 1];
            double u_d = vec[offset + j * parameters.M + i - 1];

            // Compute derivatives 
            double ux = (u_r - u_l) / (2 * parameters.dx);
            double uy = (u_u - u_d) / (2 * parameters.dy);
            double uxx = (u_r - 2 * u + u_l) / (parameters.dx * parameters.dx);
            double uyy = (u_u - 2 * u + u_d) / (parameters.dy * parameters.dy);

            // Compute temperature terms
            double diffusion = parameters.kappa * (uxx + uyy);
            double convection = v_v1 * ux + v_v2 * uy;
            double p = (u >= parameters.upc) * b * exp(u /(1 + parameters.epsilon * u));
            double reaction = p - parameters.alpha * u;

            // Compute PDE RHS 
            u_k = diffusion - convection + reaction;
            b_k = - (parameters.epsilon / parameters.q) * p;
        }

        k[u_index] = u_k;
        k[b_index] = b_k;
        tId += gridDim.x * blockDim.x;
    }
}

__global__ void RHSvec2(Parameters parameters, DiffMats DM, double *k, double *vec) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    int n_sim = parameters.x_ign_n * parameters.y_ign_n;
    int offset = n_sim * parameters.M * parameters.N;
    while (tId < n_sim * parameters.M * parameters.N) {
        int sim = tId / (parameters.M * parameters.N);
        int i = (tId - sim * parameters.M * parameters.N) % parameters.M; // Row index
        int j = (tId - sim * parameters.M * parameters.N) / parameters.M; // Col index
        int u_index = sim * parameters.M * parameters.N + j * parameters.M + i;
        int b_index = offset + sim * parameters.M * parameters.N + j * parameters.M + i;

        double u_k = 0;
        double b_k = 0;
        if (!(i == 0 || i == parameters.M - 1 || j == 0 || j == parameters.N - 1)) {
            int offset2 = sim * parameters.M * parameters.N;

            /* Get actual value of approximations */
            double u = vec[u_index];
            double b = vec[b_index];

            /* Evaluate vector field */
            double v_v1 = v1(buffer[j], buffer[parameters.N + i]);
            double v_v2 = v2(buffer[j], buffer[parameters.N + i]);

            /* Get stencil */
            double u_r = vec[offset2 + (j+1) * parameters.M + i];
            double u_l = vec[offset2 + (j-1) * parameters.M + i];
            double u_u = vec[offset2 + j * parameters.M + i + 1];
            double u_d = vec[offset2 + j * parameters.M + i - 1];

            /* Compute derivatives */
            double ux = (u_r - u_l) / (2 * parameters.dx);
            double uy = (u_u - u_d) / (2 * parameters.dy);
            double uxx = (u_r - 2 * u + u_l) / (parameters.dx * parameters.dx);
            double uyy = (u_u - 2 * u + u_d) / (parameters.dy * parameters.dy);

            /* Compute temperature terms */
            double diffusion = parameters.kappa * (uxx + uyy);
            double convection = v_v1 * ux + v_v2 * uy;
            double p = (u >= parameters.upc) * b * exp(u /(1 + parameters.epsilon * u));
            double reaction = p - parameters.alpha * u;

            /* Compute PDE RHS */
            u_k = diffusion - convection + reaction;
            b_k = - (parameters.epsilon / parameters.q) * p;
        }

        k[u_index] = u_k;
        k[b_index] = b_k;
        tId += gridDim.x * blockDim.x;
    }
}

__global__ void RHSvec3(Parameters parameters, DiffMats DM, double *k, double *vec) {

    extern __shared__ double usm[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    //int sim = bid % 16;
    int shift = 8;

    int offset = bid * shift;

    int i = tid % shift;
    int j = tid / shift;

    int ii = i + offset;

    //printf("i: %d, j: %d, ii: %d, tid: %d, bid: %d\n", i, j, ii, tid, bid);

    /* Copy memory */
    usm[tid] = vec[j * parameters.M + ii];

    __syncthreads();


    // int tId = threadIdx.x + blockIdx.x * blockDim.x;
    // int n_sim = parameters.x_ign_n * parameters.y_ign_n;
    // while (tId < n_sim * parameters.M * parameters.N) {
    // 	int sim = tId / (parameters.M * parameters.N);
    // 	int i = (tId - sim * parameters.M * parameters.N) % parameters.M; // Row index
    // 	int j = (tId - sim * parameters.M * parameters.N) / parameters.M; // Col index
    // 	int offset = 2 * sim  * parameters.M * parameters.N;
    // 	int gindex = offset + j * parameters.M + i;

    // 	int u_index = gindex;//j * parameters.M + i;
    // 	int b_index = parameters.M * parameters.N + u_index;
    double u_k = 0;
    double b_k = 0;

    if (!(i == bid * shift || i == (bid + 1) * shift - 1 || j == 0 || j == parameters.N - 1)) {
        /* Get actual value of approximations */
        double u = usm[j * shift + i];
        double b = vec[j * parameters.M + ii];

        /* Evaluate vector field */
        double v_v1 = v1(buffer[j], buffer[parameters.N + ii]);
        double v_v2 = v2(buffer[j], buffer[parameters.N + ii]);

        /* Get stencil */
        double u_r = usm[(j+1) * shift + i];
        double u_l = usm[(j-1) * shift + i];
        double u_u = usm[j * shift + i + 1];
        double u_d = usm[j * shift + i - 1];

        // double u_r = 0;
        // double u_l = 0;
        // double u_u = 0;
        // double u_d = 0;

        /* Compute derivatives */
        double ux = (u_r - u_l) / (2 * parameters.dx);
        double uy = (u_u - u_d) / (2 * parameters.dy);
        double uxx = (u_r - 2 * u + u_l) / (parameters.dx * parameters.dx);
        double uyy = (u_u - 2 * u + u_d) / (parameters.dy * parameters.dy);

        /* Compute temperature terms*/
        double diffusion = parameters.kappa * (uxx + uyy);
        double convection = v_v1 * ux + v_v2 * uy;
        double p = (u >= parameters.upc) * b * exp(u /(1 + parameters.epsilon * u));
        double reaction = p - parameters.alpha * u;

        /* Compute PDE RHS */
        u_k = diffusion - convection + reaction;
        b_k = - (parameters.epsilon / parameters.q) * p;
    }

    k[j * parameters.M + ii] = u_k;
    k[parameters.M * parameters.N + j * parameters.M + ii] = b_k;
    // 	tId += gridDim.x * blockDim.x;
    // }
}

/**
 * @brief Element-wise a + scalar * b
 * 
 * @param parameters Model and numerical parameters
 * @param c Pointer to result array
 * @param a Pointer to first array
 * @param b Pointer to second array
 * @param scalar Some scalar
 * @param size Size of arrays
 */
__global__ void sumVector(Parameters parameters, double *c, double *a, double *b, double scalar, int size) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    while (tId < size) {
        c[tId] = a[tId] + scalar * b[tId];
        tId += gridDim.x * blockDim.x;
    }
}

/**
 * @brief Euler scheme for integration
 * 
 * @param parameters Model and numerical parameters
 * @param Y_new Pointer to new approximation
 * @param Y_old Pointer to old approximation
 * @param F Pointer to function evaluation
 * @param dt \f$ \Delta t \f$
 * @param size Y size
 */
__global__ void EulerScheme(Parameters parameters, double *Y_new, double *Y_old, double *F, double dt, int size) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    while (tId < size) {
        Y_new[tId] = Y_old[tId] + dt * F[tId];
        tId += gridDim.x * blockDim.x;
    }
}

/**
 * @brief Runge-Kutta 4th order scheme for integration
 * 
 * @param parameters Model and numerical parameters
 * @param Y_new Pointer to new approximation
 * @param Y_old Pointer to old approximation
 * @param k1 Pointer to k1 array
 * @param k2 Pointer to k2 array
 * @param k3 Pointer to k3 array
 * @param k4 Pointer to k4 array
 * @param dt \f$ \Delta t \f$
 * @param size Y size
 */
__global__ void RK4Scheme(Parameters parameters, double *Y_new, double *Y_old, double *k1,
    double *k2, double *k3, double *k4, double dt, int size) {
    int tId = threadIdx.x + blockIdx.x * blockDim.x;
    while (tId < size) {
        Y_new[tId] = Y_old[tId] + (1.0 / 6.0) * dt * (k1[tId] + 2 * k2[tId] + 2 * k3[tId] + k4[tId]);
        tId += gridDim.x * blockDim.x;
    }
}

/* ODE solver for MOL */
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

        for (int k = 1; k <= parameters.L; k++) {
            cudaMemcpy(d_Y_tmp, d_Y, size * sizeof(double), cudaMemcpyDeviceToDevice);
            RHSvec<<<DG, DB>>>(parameters, DM, d_Y, d_Y_tmp);
            //RHSvec3<<<16, 1024, 1024 * sizeof(double)>>>(parameters, DM, d_Y, d_Y_tmp);
            EulerScheme<<<DG, DB>>>(parameters, d_Y, d_Y_tmp, d_Y, dt, size);
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

        for (int k = 1; k <= parameters.L; k++) {
            cudaMemcpy(d_Y_tmp, d_Y, size * sizeof(double), cudaMemcpyDeviceToDevice); // Y_{t-1}
            RHSvec<<<DG, DB>>>(parameters, DM, d_k1, d_Y_tmp); // Compute k1
            sumVector<<<DG, DB>>>(parameters, d_ktmp, d_Y_tmp, d_k1, 0.5*dt, size); // Y_{t-1} + 0.5*dt*k1
            RHSvec<<<DG, DB>>>(parameters, DM, d_k2, d_ktmp); // Compute k2
            sumVector<<<DG, DB>>>(parameters, d_ktmp, d_Y_tmp, d_k2, 0.5 * dt, size); // Y_{t-1} + 0.5*dt*k2
            RHSvec<<<DG, DB>>>(parameters, DM, d_k3, d_ktmp); // Compute k3
            sumVector<<<DG, DB>>>(parameters, d_ktmp, d_Y_tmp, d_k3, dt, size); // Y_{t-1} + dt*k3
            RHSvec<<<DG, DB>>>(parameters, DM, d_k4, d_ktmp); // Compute k4
            RK4Scheme<<<DG, DB>>>(parameters, d_Y, d_Y_tmp, d_k1, d_k2, d_k3, d_k4, dt, size);
        }

        // Free memory
        cudaFree(d_Y_tmp);
        cudaFree(d_k1);
        cudaFree(d_k2);
        cudaFree(d_k3);
        cudaFree(d_k4);
        cudaFree(d_ktmp);
    }
}

/**
 * @brief Fill device memory with initial conditions
 * 
 * @param parameters Model and numerical parameters
 * @param d_sims Pointer to fill 
 */
void fillInitialConditions(Parameters parameters, double *d_sims) {
    // Initial wildfire focus 
    double dx_ign = 0.0, dy_ign = 0.0, x_ign, y_ign;

    if (parameters.x_ign_n > 1) {
        dx_ign = (parameters.x_ign_max - parameters.x_ign_min) / (parameters.x_ign_n - 1);
    }

    if (parameters.y_ign_n > 1) {
        dy_ign = (parameters.y_ign_max - parameters.y_ign_min) / (parameters.y_ign_n - 1);
    }

    // Temporal arrays 
    double *d_tmp;

    //int offset = parameters.x_ign_n * parameters.y_ign_n * parameters.M * parameters.N;

    cudaMalloc(&d_tmp, parameters.M * parameters.N * sizeof(double));
    cudaMemset(d_tmp, 0, parameters.M * parameters.N  * sizeof(double));

    // Fill initial conditions according to ignitions points 
    int sim_ = 0;
    for (int i=0; i < parameters.y_ign_n; i++) {
        for (int j=0; j < parameters.x_ign_n; j++) {

            /* Coordinates of ignition point */
            x_ign = parameters.x_ign_min + dx_ign * j;
            y_ign = parameters.y_ign_min + dy_ign * i;

            /* Compute initial condition for temperature */
            U0<<<DG, DB>>>(parameters, d_tmp, x_ign, y_ign);
            cudaMemcpy(d_sims + 2*sim_*parameters.M*parameters.N,
                d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
            // cudaMemcpy(d_sims + sim_*parameters.M*parameters.N,
            // 	d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);


            /* Compute initial condition for fuel */
            B0<<<DG, DB>>>(parameters, d_tmp);
            cudaMemcpy(d_sims + (2*sim_+1) * parameters.M * parameters.N,
                d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);
            // cudaMemcpy(d_sims + offset + sim_ * parameters.M * parameters.N,
            // 	d_tmp, parameters.M * parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);

            sim_++;
        }
    }

    /* Free memory */
    cudaFree(d_tmp);
}

/**
 * @brief Wildfire handler
 * 
 * @param parameters Model and numerical parameters
 */
void wildfire(Parameters parameters) {

    // Variables for times
    clock_t begin, end;
    float milliseconds;
    double time_spent;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Kernel Parameters 
    int n_sim = parameters.x_ign_n * parameters.y_ign_n; // Number of wildfire simulations
    int size = 2 * n_sim * parameters.M * parameters.N;

    // Domain differentials 
    double dx = (parameters.x_max - parameters.x_min) / (parameters.N-1);
    double dy = (parameters.y_max - parameters.y_min) / (parameters.M-1);
    double dt = parameters.t_max / parameters.L;

    // Memory for simulations 
    double *h_sims = (double *) malloc(size * sizeof(double));
    double *d_sims;

    // Domain vectors 
    double *h_x = (double *) malloc(parameters.N * sizeof(double));
    double *h_y = (double *) malloc(parameters.M * sizeof(double));
    double *d_x, *d_y;

    // Print data log
    printf("Simulation ID: %s\n", parameters.sim_id);
    printf("Number of numerical simulations: %d\n", parameters.x_ign_n * parameters.y_ign_n);

    // Differentiation Matrices //

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
    cudaMalloc((void **) &d_sims, size * sizeof(double));
    cudaMalloc((void **) &d_x, parameters.N * sizeof(double));
    cudaMalloc((void **) &d_y, parameters.M * sizeof(double));
    cudaMalloc((void **) &d_Dx, parameters.N * parameters.N * sizeof(double));
    cudaMalloc((void **) &d_Dy, parameters.M * parameters.M * sizeof(double));
    cudaMalloc((void **) &d_Dxx, parameters.N * parameters.N * sizeof(double));
    cudaMalloc((void **) &d_Dyy, parameters.M * parameters.M * sizeof(double));


    // Fill spatial domain and differentiation matrices 
    if (strcmp(parameters.spatial, "FD") == 0) {
        // Spatial domain
        printf("Finite Difference in space\n");
        printf("dx: %f\n", dx);
        printf("dy: %f\n", dy);

        parameters.dx = dx;
        parameters.dy = dy;

        fillVectorKernel<<<DG, DB>>>(d_x, dx, parameters.N);
        fillVectorKernel<<<DG, DB>>>(d_y, dy, parameters.M);

        // Differentiation matrices
        FD1Kernel<<<DG, DB>>>(d_Dx, parameters.N, dx);
        FD1Kernel<<<DG, DB>>>(d_Dy, parameters.M, dy);
        FD2Kernel<<<DG, DB>>>(d_Dxx, parameters.N, dx);
        FD2Kernel<<<DG, DB>>>(d_Dyy, parameters.M, dy);

    } else if (strcmp(parameters.spatial, "Cheb") == 0) {
        /* TODO */
        // Spatial domain
        printf("Chebyshev in space\n");
        //fprintf(log, "Chebyshev in space\n");
        /* Check M y N */

        /*
        ChebyshevNodes<<<DG(parameters.M * parameters.N), DB>>>(d_x, parameters.N - 1);
        ChebyshevNodes<<<DG(parameters.M * parameters.N), DB>>>(d_y, parameters.M - 1);

        // Differentiation matrices
        ChebyshevMatrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dx, d_x, parameters.N - 1);
        ChebyshevMatrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dy, d_y, parameters.M - 1);
        Chebyshev2Matrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dxx, d_Dx, parameters.N - 1);
        Chebyshev2Matrix<<<DG(parameters.M * parameters.N), DB>>>(d_Dyy, d_Dy, parameters.M - 1);
        */
    } else {
        printf("Spatial domain error\n");
        exit(0);
    }

    parameters.dt = dt;

    // Copy spatial domain to constant memory 
    cudaMemcpyToSymbol(buffer, d_x, parameters.N * sizeof(double), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(buffer, d_y, parameters.M * sizeof(double), parameters.N * sizeof(double), cudaMemcpyDeviceToDevice);

    // Fill differentiation matrices
    DM.Dx = d_Dx;
    DM.Dy = d_Dy;
    DM.Dxx = d_Dxx;
    DM.Dyy = d_Dyy;

    // Fill initial conditions 
    fillInitialConditions(parameters, d_sims);

    // Save initial conditions
    if (parameters.save) {
        cudaMemcpy(h_sims, d_sims, size * sizeof(double), cudaMemcpyDeviceToHost);
        save(parameters, h_sims, 1);
    }

    // Timers
    begin = clock();
    cudaEventRecord(start);

    // ODE Integration (solver)
    ODESolver(parameters, DM, d_sims, dt);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution CUDA: %.16f\n", milliseconds*1e-3);

    // Timers 
    end = clock();
    time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    printf("Execution time: %f [s]\n", time_spent);

    // Write log file
    logFile(parameters, time_spent);

    // Copy approximations to host
    cudaMemcpy(h_sims, d_sims, size * sizeof(double), cudaMemcpyDeviceToHost);

    // Save approximations
    if (parameters.save) {
        save(parameters, h_sims, 0);
    }

    // Free memory
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