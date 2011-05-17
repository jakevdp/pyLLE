# run python setup.py build_ext
#  in order to build cpp files from the pyx files

import os

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

MatVec_sources = [ ('../lib/MatVec/%s' % F) \
                       for F in os.listdir('../lib/MatVec') \
                       if F.endswith('.cpp') and not F.endswith('test.cpp') ]
LLE_sources = [('../src/%s' % F) for F in os.listdir('../src') \
                   if F.endswith('.cpp') and not F.endswith('test.cpp') ]

LLE_ext = Extension(
    "LLE",
    ["LLE.pyx"] + MatVec_sources + LLE_sources,
    language="c++",
    include_dirs=['../lib/MatVec','../include'],
    libraries=['stdc++','blas','lapack','arpack'],
    library_dirs = ['/usr/lib'],
    )

Balltree_ext = Extension(
    "ball_tree",                 
    ["ball_tree.pyx"],  
    language="c++",              
    include_dirs=['../include'],  
    libraries=['stdc++'],
    )

setup(cmdclass = {'build_ext': build_ext},
      name='LLE',
      version='1.0',
      ext_modules=[LLE_ext],
      )

setup(cmdclass = {'build_ext': build_ext},
      name='ball_tree',
      version='1.0',
      ext_modules=[Balltree_ext],
      )
