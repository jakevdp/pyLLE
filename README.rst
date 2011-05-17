.. -*- mode: rst -*-

Under Construction!!
====================
This package is still under construction... 
Many of the available C++ routines have not yet been wrapped for python,
but the basic ones have.

About
=====

LLE is a set of python bindings to a C++ package written by Jake Vanderplas

Dependencies
============

The required dependencies to build the software are python >= 2.5,
setuptools, NumPy >= 1.2, SciPy >= 0.7 and a working C++ compiler.

Also requires the FORTRAN libraries BLAS, LAPACK, and ARPACK.
These can be downloaded at:

    http://www.netlib.org/lapack/
    http://www.netlib.org/blas/
    http://www.caam.rice.edu/software/ARPACK/
if python+numpy is installed on your system, you may already have a
working BLAS and LAPACK installation. 


Install
=======

This packages uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --home

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install
