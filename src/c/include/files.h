#ifndef FILES_H
#define FILES_H

void printMatrix(const float *A, int rows, int cols);
void readConfig(const char *filename, int *Nx, int *Ny, 
  float *xmin, float *xmax, float *ymin, float *ymax, 
  int *Tmax, float *dt, float *kappa, float *epsilon, 
  float *upc, float *q, float *alpha);
void readConf(const char *filename);
void readInput(const char *filename, float *A, int rows, int cols);

#endif