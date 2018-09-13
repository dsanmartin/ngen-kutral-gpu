#include <stdio.h>
#include <stdlib.h>
#include "include/files.h"

#define MAX_CONFIG_VARIABLE_LEN 20	

// Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void printMatrix(const float *A, int rows, int cols) {
	for(int i = 0; i < rows; ++i){
		for(int j = 0; j < cols; ++j){
			printf("%lf ", A[j * rows + i]);	
		}
		printf("\n");
	}
	printf("\n");
}


void readConfig(const char *filename, int *Nx, int *Ny, float *xmin, float *xmax, float *ymin, float *ymax, 
	int *Tmax, float *dt, float *kappa, float *epsilon, float *upc, float *q, float *alpha) {
	FILE *file;
	file = fopen(filename, "r");
  fscanf(file, "%*s = %d", &Nx);
	/*
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
	*/
	printf("holas %d", Nx);
	fclose(file);
}


/*
void readConf(const char *filename) {
	FILE *file;
	file = fopen(filename, "r");
	int *Nx, Ny, T;
	float xmin, xmax, ymin, ymax;
  fscanf(file, "%*s = %d", &Nx);
	printf("hola %d", Nx);
	fclose(file);
}
*/
// Read file to array
void readInput(const char *filename, float *A, int rows, int cols) {
  //int i, j;
  FILE *file;
  file = fopen(filename, "r");
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
      //Use lf format specifier, %c is for character
      if (!fscanf(file, "%f", &A[j * rows + i])) 
        break;
    }
  }
  fclose(file);
}

