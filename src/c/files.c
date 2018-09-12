#include <stdio.h>
#include "include/files.h"

//Print matrix A(nr_rows_A, nr_cols_A) storage in column-major format
void printMatrix(const float *A, int rows, int cols) {
    for(int i = 0; i < rows; ++i){
        for(int j = 0; j < cols; ++j){
            printf("%lf ", A[j * rows + i]);	
        }
        printf("\n");
    }
    printf("\n");
}

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

