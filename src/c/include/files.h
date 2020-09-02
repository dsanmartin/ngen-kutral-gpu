/**
 * @file files.h
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief File handler header
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#ifndef FILES_H
#define FILES_H

#include "structures.h"
#include <string.h>

void logFile(Parameters parameters, double time);
void printMatrix(const double *A, int rows, int cols);
void printApproximations(const double *A, int Nx, int Ny, int T);
// void readConfig(const char *filename, int *Nx, int *Ny, 
//   double *xmin, double *xmax, double *ymin, double *ymax, 
//   int *Tmax, double *dt, double *kappa, double *epsilon, 
//   double *upc, double *q, double *alpha);
void readConf(const char *filename);
void readInput(const char *filename, double *A, int rows, int cols);
void saveApproximation(const char *filename, const double *A, int rows, int cols);
void save(Parameters parameters, double *h_sims, int ic);

#endif