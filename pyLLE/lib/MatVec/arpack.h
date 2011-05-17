#ifndef ARPACK_H
#define ARPACK_H


//interface to needed ARPACK fortran functions
extern "C" void dsaupd_(int *ido, char *bmat, int *n, char *which,
			int *nev, double *tol, double *resid, int *ncv,
			double *v, int *ldv, int *iparam, int *ipntr,
			double *workd, double *workl, int *lworkl,
			int *info);

extern "C" void dseupd_(int *rvec, char *All, int *select, double *d,
			double *z, int *ldz, double *sigma, 
			char *bmat, int *n, char *which, int *nev,
			double *tol, double *resid, int *ncv, double *v,
			int *ldv, int *iparam, int *ipntr, double *workd,
			double *workl, int *lworkl, int *ierr);

#endif /*ARPACK_H*/

