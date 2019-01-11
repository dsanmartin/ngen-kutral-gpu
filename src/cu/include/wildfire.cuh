#ifndef SOLVER_CUH
#define SOLVER_CUH
#include "../../c/include/structures.h"

void wildfire(Parameters parameters);
void ODESolver(double *U, double *B, DiffMats DM, Parameters parameters, double dt);

#endif