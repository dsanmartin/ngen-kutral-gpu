/**
 * @file files.c
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Output files handler
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "include/files.h"
// #include "include/structures.h"

/**
 * @brief Generate log file
 * 
 * @param parameters Model and numerical parameters
 * @param sim_time Execution time of simulations
 */
void logFile(Parameters parameters, double sim_time) {
    // Log file with parameters info 
    char log_name[DIR_LEN + 8];
    sprintf(log_name, "%s/log.txt", parameters.dir);
    FILE *log = fopen(log_name, "w");

    // Write parameters in log 
    fprintf(log, "Simulation ID: %s\n", parameters.sim_id);
    fprintf(log, "Number of numerical simulations: %d\n", parameters.x_ign_n * parameters.y_ign_n);
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
    fprintf(log, "dx: %f\n", parameters.dx);
    fprintf(log, "dy: %f\n", parameters.dy);
    fprintf(log, "\nTime\n");
    fprintf(log, "------\n");	
    fprintf(log, "Domain: [0, %f]\n", parameters.t_max);
    fprintf(log, "Method: %s\n", parameters.time);
    fprintf(log, "L: %d\n", parameters.L);
    fprintf(log, "dt: %f\n", parameters.dt);	
    fprintf(log, "\nExecution time: %f [s]\n", sim_time);

    // Close log file
    fclose(log);
}

/**
 * @brief Print approximation matrices
 * 
 * @param A Pointer of matrix
 * @param Nx Number of nodes in \f$ x \f$
 * @param Ny Number of nodes in \f$ y \f$
 * @param T Number of nodes in \f$ t \f$
 */
void printApproximations(const double *A, int Nx, int Ny, int T) {
    for (int k = 0; k < T; k++) {
        for (int i = 0; i < Ny; i++) {
            for (int j = 0; j < Nx; j++) {
                printf("%lf ", A[k * Nx * Ny + j * Nx + i]);	
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Print matrix A storage in column-major format
 * 
 * @param A Pointer to matrix
 * @param rows Number of rows
 * @param cols Number of columns
 */
void printMatrix(const double *A, int rows, int cols) {
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            printf("%lf ", A[j * rows + i]);	
        }
        printf("\n");
    }
    printf("\n");
}

/**
 * @brief Read file to array
 * 
 * @param filename File path name
 * @param A Pointer to fill with data
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 */
void readInput(const char *filename, double *A, int rows, int cols) {
    FILE *file;
    file = fopen(filename, "r");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            if (!fscanf(file, "%lf", &A[j * rows + i])) 
                break;
        }
    }
    fclose(file);
}

/**
 * @brief Save approximations matrices
 * 
 * @param filename File path name
 * @param A Pointer to fill with data
 * @param rows Number of matrix rows
 * @param cols Number of matrix columns
 */
void saveApproximation(const char *filename, const double *A, int rows, int cols) {
    FILE *f = fopen(filename, "w");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(f, "%.16f ", A[j * rows + i]);	
        }
        fprintf(f, "\n");
    }
}

/**
 * @brief Save initial condition or last approximation of each simulation
 * 
 * @param parameters Model and methods parameters
 * @param h_sims Pointer with all simulations
 * @param ic If save initial condition or last approximation
 */
void save(Parameters parameters, double *h_sims, int ic) {
    // Simulation name 
    char sim_name[DIR_LEN + 32];

    //int offset = parameters.x_ign_n * parameters.y_ign_n * parameters.M * parameters.N;

    // Temporal array 
    double *tmp = (double *) malloc(parameters.M * parameters.N * sizeof(double));
    
    int sim_ = 0;
    for (int i = 0; i < parameters.y_ign_n; i++) {
        for (int j = 0; j < parameters.x_ign_n; j++) {	
            // Temperature
            memcpy(tmp, h_sims + 2*sim_*parameters.M*parameters.N, parameters.M * parameters.N * sizeof(double));
            // memcpy(h_tmp, h_sims + sim_*parameters.M*parameters.N, parameters.M * parameters.N * sizeof(double));
            memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
            if (ic)
                sprintf(sim_name, "%s/%s_%d-%d.txt", parameters.dir, "U0", i, j); // Simulation name
            else
                sprintf(sim_name, "%s/%s_%d-%d.txt", parameters.dir, "U", i, j); // Simulation name
            
            saveApproximation(sim_name, tmp, parameters.M, parameters.N); // Save U
            
            /* Fuel */
            memcpy(tmp, h_sims + (2*sim_ + 1)*parameters.M*parameters.N, parameters.M * parameters.N * sizeof(double));
            // memcpy(h_tmp, h_sims + offset + sim_*parameters.M*parameters.N, parameters.M * parameters.N * sizeof(double));
            memset(&sim_name[0], 0, sizeof(sim_name)); // Reset simulation name
            if (ic)
                sprintf(sim_name, "%s/%s_%d-%d.txt", parameters.dir,"B0", i, j); // Simulation name
            else
                sprintf(sim_name, "%s/%s_%d-%d.txt", parameters.dir,"B", i, j); // Simulation name
                
            saveApproximation(sim_name, tmp, parameters.M, parameters.N);	// Save B	
            
            sim_++;
        }
    }
    
    free(tmp);
}