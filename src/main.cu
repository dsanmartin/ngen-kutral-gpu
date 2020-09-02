/**
 * @file main.cu
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief This files handles wildfires simulations
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "c/include/utils.h"
#include "c/include/files.h"
#include "c/include/structures.h"
#include "cu/include/wildfire.cuh"

/**
 * @brief Program main
 * 
 * @param argc 
 * @param argv 
 * @return int 
 */
int main(int argc, char *argv[]) {
    
    // Info for directory simulation
    char buff[32];
    char directory[DIR_LEN];
    time_t now = time (0);
    struct stat st = {0};

    // Structure for model and numerical parameters
    Parameters parameters;

    // PDE parameters
    parameters.kappa = 1e-1; // Inverse of conductivity
    parameters.epsilon = 3e-1; // inverse of activation energy
    parameters.upc = 3.0; // Phase change threshold
    parameters.q = 1.0; // Reaction heat coefficient
    parameters.alpha = 1e-3; // Natural convection
    parameters.x_min = 0; // x domain minimum
    parameters.x_max = 90; // x domain maximum
    parameters.y_min = 0; // y domain minimum
    parameters.y_max = 90; // y domain maximum
    parameters.t_max = 25; // t domain end

    // Domain definition
    parameters.M = atoi(argv[3]); // Spatial resolution (y-axis - matrix rows)
    parameters.N = atoi(argv[4]); // Spatial resolution (x-axis - matrix columns)
    parameters.L = atoi(argv[5]); // Time resolution

    // Methods
    parameters.spatial = argv[1]; // "FD" or "Cheb"
    parameters.time = argv[2]; // "Euler" or "RK4"

    // Ignition points grid 
    parameters.x_ign_min = 20; // x coordinate starting
    parameters.x_ign_max = 70; // x coordinate ending
    parameters.y_ign_min = 20; // y coordinate starting
    parameters.y_ign_max = 70; // y coordinate ending
    parameters.x_ign_n = atoi(argv[6]); // Number of ignition points in x
    parameters.y_ign_n = atoi(argv[7]);// Number of ignition points in y

    // CUDA parameters 
    parameters.threads = atoi(argv[8]); // Threads per blocks
    parameters.blocks = atoi(argv[9]); // Blocks per grid

    // Other parameters
    parameters.save = atoi(argv[10]); // 1 if save
    parameters.exp_id = argv[11]; // Experiment id

    // Show CUDA parameters 
    printf("\nThreads per block: %d\n", parameters.threads);
    printf("Blocks per grid: %d\n", parameters.blocks);

    // if (parameters.x_ign_n == 1 && parameters.y_ign_n == 1) {
    // 	parameters.x_ign_min = parameters.x_ign_max;
    // 	parameters.y_ign_min = parameters.y_ign_max;
    // }

    // Create simulation ID 
    strftime(buff, 16, "%Y%m%d%H%M%S", localtime (&now));
    strcat(buff, parameters.exp_id);
    parameters.sim_id = (const char*) buff;	

    // Create simulation directory 
    sprintf(directory, "test/output/%s/", parameters.sim_id);
    if (stat(directory, &st) == -1) {
        mkdir(directory, 0700);
        parameters.dir = (const char*) directory;
    } 
    
    // Run numerical simulations
    wildfire(parameters);

    return 0;
}
