#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "c/include/files.h"
#include "c/include/structures.h"
#include "cu/include/wildfire.cuh"

int main(int argc, char *argv[]) {

	/* PDE parameters */
	Parameters parameters;
	parameters.kappa = 1e-1;
	parameters.epsilon = 3e-1;
	parameters.upc = 3.0;
	parameters.q = 1.0;
	parameters.alpha = 1e-3;
	parameters.x_min = 0;
	parameters.x_max = 90;
	parameters.y_min = 0;
	parameters.y_max = 90;
	parameters.t_max = 50;

	/* Domain definition */
	parameters.L = 500; // Time resolution
	parameters.M = 128; // Spatial resolution (y-axis - matrix rows)
	parameters.N = 128; // Spatial resolution (x-axis - matrix columns)

	/* Methods */
	parameters.spatial = "FD";
	parameters.time = "Euler";	

	/* Ignition points */
	parameters.x_ign_min = -60;
	parameters.x_ign_max = -20;
	parameters.y_ign_min = -60;
	parameters.y_ign_max = -20;
	parameters.x_ign_n = atoi(argv[1]);//5;
	parameters.y_ign_n = atoi(argv[2]);//5;
	// parameters.x_ign_min = -20;
	// parameters.x_ign_max = -20;
	// parameters.y_ign_min = -20;
	// parameters.y_ign_max = -20;
	// parameters.x_ign_n = 1;
	// parameters.y_ign_n = 1;
	

	/* Simulations */
	wildfire(parameters);

	return 0;
}