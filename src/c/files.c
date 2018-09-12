#include <stdio.h>
//#include <stdlib.h>
#include "include/files.h"

void hola() {
  printf("hola");
  // test
}

void readInput(double *A, int rows, int cols) {
  //int i, j;
  FILE *file;
  file = fopen("Inputs/test.txt", "r");
  for(int i = 0; i < rows; i++) {
    for(int j = 0; j < cols; j++) {
    //Use lf format specifier, %c is for character
      if (!fscanf(file, "%lf", &A[j * rows + i])) 
        break;
        // mat[i][j] -= '0'; 
       //printf("%lf\n",mat[i][j]); //Use lf format specifier, \n is for new line
      }
  }
  fclose(file);
  // if (file) {
  //     while ((c = getc(file)) != EOF)
  //         putchar(c);
  //     fclose(file);
  // }
}