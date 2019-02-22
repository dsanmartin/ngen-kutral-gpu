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
	double x_ign_min;
	double x_ign_max;
	double y_ign_min;
	double y_ign_max;
  int L;
  int M;
  int N;
	int x_ign_n;
	int y_ign_n;
  const char *spatial;
  const char *time;
	const char *approach;
	const char *sim_id;
	const char *dir;
} typedef Parameters;

struct _diffmats {
	double *Dx;
	double *Dy;
	double *Dxx;
	double *Dyy;
} typedef DiffMats;

#endif