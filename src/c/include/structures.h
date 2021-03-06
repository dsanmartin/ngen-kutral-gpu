/**
 * @file structures.h
 * @author Daniel San Martin (dsanmartinreyes@gmail.com)
 * @brief Data structures header
 * @version 0.1
 * @date 2020-09-01
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#ifndef STRUCTURES_H
#define STRUCTURES_H

/**
 * @brief Structure for parameters.
 * 
 */
struct _parameters {
    double kappa;
    double epsilon;
    double upc;
    double q;
    double alpha;
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double t_max;
    double x_ign_min;
    double x_ign_max;
    double y_ign_min;
    double y_ign_max;
    double dx;
    double dy;
    double dt;
    int L;
    int M;
    int N;
    int x_ign_n;
    int y_ign_n;
    int threads;
    int blocks;
    int save;
    const char *exp_id;
    const char *spatial;
    const char *time;
    const char *approach;
    const char *sim_id;
    const char *dir;
} typedef Parameters;

/**
 * @brief Structure for differentiation matrices.
 * 
 */
struct _diffmats {
    double *Dx;
    double *Dy;
    double *Dxx;
    double *Dyy;
} typedef DiffMats;

#endif

#define DIR_LEN 1024