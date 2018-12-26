#ifndef STRUCTURES_H
#define STRUCTURES_H

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
	double x_ign;
	double y_ign;
  int L;
  int M;
  int N;
  const char *spatial;
  const char *time;
	const char *sim_name;
} typedef Parameters;

struct _diffmats {
	double *Dx;
	double *Dy;
	double *Dxx;
	double *Dyy;
} typedef DiffMats;

#endif