.. -*- mode: rst -*-

Under Construction!!
====================
This package is still under construction... cython wrappers and
distutils installation is still not complete.  I hope to have
it done some time this month!!

About
=====

LLE is a set of python bindings to a C++ package written by Jake Vanderplas

Dependencies
============

The required dependencies to build the software are python >= 2.5,
setuptools, NumPy >= 1.2, SciPy >= 0.7 and a working C++ compiler.

Also requires the FORTRAN libraries BLAS, LAPACK, and ARPACK


Install
=======

Note: package is under construction... this may not all be set up yet...

This packages uses distutils, which is the default way of installing
python modules. To install in your home directory, use::

  python setup.py install --home

To install for all users on Unix/Linux::

  python setup.py build
  sudo python setup.py install
