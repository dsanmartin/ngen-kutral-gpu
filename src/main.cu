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
	parameters.L = 3000; // Time resolution
	parameters.M = 128; // Spatial resolution (y-axis - matrix rows)
	parameters.N = 128; // Spatial resolution (x-axis - matrix columns)

	/* Methods */
	parameters.spatial = "FD";
	parameters.time = "RK4";	

	double x_shift[] = {-20, -30};//, -40, -50, -60, -70, -80};
	double y_shift[] = {-20, -20};//, -40, -50, -60, -70, -80};
	char char_arr[10];

	for (int i=0; i < sizeof(y_shift) / sizeof(y_shift[0]); i++) {
		for (int j=0; j < sizeof(x_shift) / sizeof(x_shift[0]); j++) {
			parameters.x_ign = x_shift[j];
			parameters.y_ign = y_shift[i];
			sprintf(char_arr, "%d%d", i, j);
			const char* p = char_arr;
			parameters.sim_name = p;
			wildfire(parameters);
		}
	}

	return 0;
}