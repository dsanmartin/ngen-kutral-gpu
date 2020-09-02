/**
 * @file wildfire.cuh
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Wildfire cuda header
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef WILDFIRE_CUH
#define WILDFIRE_CUH
#include "../../c/include/structures.h"

void wildfire(Parameters parameters);
void ODESolver(Parameters parameters, DiffMats DM, double *d_Y, double dt);

#endif