#
#  GCC specifies your g++ compiler
#
GCC = g++
GCC_DEP = -I../include -I/astro/users/vanderplas/lib/cfitsio/include
GCC_INC = $(GCC_DEP) -O3 

#
#  F77 specifies your fortran compiler
#
F77	     = gfortran
F77_INC = #-O3

#
#  PYTHON specifies your python executable
#
PYTHON = /astro/apps/pkg/python64/bin/python

#
#  LFLAGS must link to LAPACK, BLAS, and ARPACK libraries
#
BLAS_FLAGS = -lblas -L/usr/lib64/
LAPACK_FLAGS = -llapack
ARPACK_FLAGS = -larpack -L/astro/users/vanderplas/lib/ARPACK

LFLAGS = -lm $(BLAS_FLAGS) $(LAPACK_FLAGS) $(ARPACK_FLAGS)
