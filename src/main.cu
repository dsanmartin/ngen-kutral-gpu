#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "c/include/utils.h"
#include "c/include/files.h"
#include "c/include/structures.h"
#include "cu/include/wildfire.cuh"

int main(int argc, char *argv[]) {
	
	// /* Info for directory simulation */
	// char buff[15];
	// char directory[40];
	// time_t now = time (0);
	// struct stat st = {0};

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
	parameters.L = 75; // Time resolution
	parameters.M = 128; // Spatial resolution (y-axis - matrix rows)
	parameters.N = 128; // Spatial resolution (x-axis - matrix columns)

	/* Methods */
	parameters.spatial = "FD";
	parameters.time = argv[1];//"Euler";
	parameters.approach = argv[4];	

	/* Ignition points */
	parameters.x_ign_min = -20;
	parameters.x_ign_max = -20;
	parameters.y_ign_min = -20;
	parameters.y_ign_max = -20;
	parameters.x_ign_n = atoi(argv[2]);//5;
	parameters.y_ign_n = atoi(argv[3]);//5;
	// parameters.x_ign_min = -20;
	// parameters.x_ign_max = -20;
	// parameters.y_ign_min = -20;
	// parameters.y_ign_max = -20;
	// parameters.x_ign_n = 1;
	// parameters.y_ign_n = 1;

	// /* Simulation ID */
  // strftime (buff, 15, "%Y%m%d%H%M%S", localtime (&now));
	// parameters.sim_id = (const char*) buff;	

	// /* Create simulation directory */
	// sprintf(directory, "test/output/%s/", parameters.sim_id);
	// if (stat(directory, &st) == -1) {
	// 		mkdir(directory, 0700);
	// }

	/* Simulations */
	wildfire(parameters);

	return 0;
}