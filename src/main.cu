#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "c/include/utils.h"
#include "c/include/files.h"
#include "c/include/structures.h"
#include "cu/include/wildfire.cuh"

#define DB(t) t
#define DG(b) b

int main(int argc, char *argv[]) {
	
	/* Info for directory simulation */
	char buff[32];
	char directory[DIR_LEN];
	time_t now = time (0);
	struct stat st = {0};

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
	parameters.t_max = 25;

	/* Domain definition */
	parameters.L = atoi(argv[5]); // Time resolution
	parameters.M = atoi(argv[3]); // Spatial resolution (y-axis - matrix rows)
	parameters.N = atoi(argv[4]); // Spatial resolution (x-axis - matrix columns)

	/* Methods */
	parameters.spatial = argv[1]; // "FD" or "Cheb"
	parameters.time = argv[2]; // "Euler" or "RK4"
	parameters.approach = argv[8]; // Time solver approach "all" threads o "block" per simulation	

	/* Ignition points */
	parameters.x_ign_min = 20;
	parameters.x_ign_max = 70;
	parameters.y_ign_min = 20;
	parameters.y_ign_max = 70;
	parameters.x_ign_n = atoi(argv[6]); // Number of ignition points in x
	parameters.y_ign_n = atoi(argv[7]);// Number of ignition points in y

	parameters.exp_id = argv[9];

	parameters.threads = atoi(argv[10]);
	parameters.blocks = atoi(argv[11]);

	parameters.save = atoi(argv[12]);

	printf("\nThreads per block: %d\n", parameters.threads);
	printf("Blocks per grid: %d\n", parameters.blocks);


	// if (parameters.x_ign_n == 1 && parameters.y_ign_n == 1) {
	// 	parameters.x_ign_min = parameters.x_ign_max;
	// 	parameters.y_ign_min = parameters.y_ign_max;
	// }

	/* Simulation ID */
	strftime(buff, 16, "%Y%m%d%H%M%S", localtime (&now));
	strcat(buff, parameters.exp_id);
	parameters.sim_id = (const char*) buff;	

	/* Create simulation directory */
	sprintf(directory, "test/output/%s/", parameters.sim_id);
	if (stat(directory, &st) == -1) {
		mkdir(directory, 0700);
		parameters.dir = (const char*) directory;
	}

	/* Simulations */
	wildfire(parameters);

	return 0;
}
