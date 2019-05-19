#include <stdio.h>
#include <stdlib.h>
#include "include/files.h"

/* Print approximations matrices */
void printApproximations(const double *A, int Nx, int Ny, int T) {
	for (int k = 0; k < T; k++) {
		for (int i = 0; i < Ny; i++) {
			for (int j = 0; j < Nx; j++) {
				printf("%lf ", A[k * Nx * Ny + j * Nx + i]);	
			}
			printf("\n");
		}
		printf("\n");
	}
	printf("\n");
}

// Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void printMatrix(const double *A, int rows, int cols) {
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++){
			printf("%lf ", A[j * rows + i]);	
		}
		printf("\n");
	}
	printf("\n");
}

/*
void readConfig(const char *filename, int *Nx, int *Ny, double *xmin, double *xmax, double *ymin, double *ymax, 
	int *Tmax, double *dt, double *kappa, double *epsilon, double *upc, double *q, double *alpha) {
	FILE *file;
	file = fopen(filename, "r");
  fscanf(file, "%*s = %d", &Nx);
	
	fscanf(file, "%*s = %d", &Ny);
	fscanf(file, "%*s = %f", &xmin);
	fscanf(file, "%*s = %f", &xmax);
	fscanf(file, "%*s = %f", &ymin);
	fscanf(file, "%*s = %f", &ymax);
	fscanf(file, "%*s = %f", &dt);
	fscanf(file, "%*s = %f", &kappa);
	fscanf(file, "%*s = %f", &epsilon);
	fscanf(file, "%*s = %f", &upc);
	fscanf(file, "%*s = %f", &q);
	fscanf(file, "%*s = %f", &alpha);
	fscanf(file, "%*s = %d", &Tmax);
	
	printf("holas %d", Nx);
	fclose(file);
}
*/

/*
void readConf(const char *filename) {
	FILE *file;
	file = fopen(filename, "r");
	int *Nx, Ny, T;
	double xmin, xmax, ymin, ymax;
  fscanf(file, "%*s = %d", &Nx);
	printf("hola %d", Nx);
	fclose(file);
}
*/

/* Read file to array */
void readInput(const char *filename, double *A, int rows, int cols) {
  FILE *file;
  file = fopen(filename, "r");
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      if (!fscanf(file, "%lf", &A[j * rows + i])) 
        break;
    }
  }
  fclose(file);
}

/* Save approximations matrices */
void saveApproximation(const char *filename, const double *A, int rows, int cols) {
	FILE *f = fopen(filename, "w");
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			fprintf(f, "%.16f ", A[j * rows + i]);	
		}
		fprintf(f, "\n");
	}
}
