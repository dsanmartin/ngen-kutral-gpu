#ifndef WILDFIRE_CUH
#define WILDFIRE_CUH
#include "../../c/include/structures.h"

void wildfire(Parameters parameters);
void ODESolver(Parameters parameters, DiffMats DM, double *d_Y, double dt);

#endif