from os.path import join
from os import listdir
import warnings
import numpy

LIBS = ['/usr/lib']

MatVec_sources = ['lib/MatVec/MatVec.cpp', 'lib/MatVec/MatTri.cpp', 
                  'lib/MatVec/MatSym.cpp', 'lib/MatVec/MatVecDecomp.cpp']
LLE_sources = ['src/LLE.cpp', 'src/IRWPCA.cpp']

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError

    config = Configuration('pyLLE', parent_package, top_path)

    config.add_subpackage('python_only')

    config.add_extension('ball_tree',
                         language='c++',
                         sources=[join('wrappers', 'ball_tree.cpp')],
                         depends=[join('include', 'BallTree.h'),
                                  join('include', 'BallTreePoint.h')],
                         include_dirs=['include',numpy.get_include()])
    
    config.add_extension('LLE',
                         language='c++',
                         sources=[join('wrappers','LLE.cpp')]\
                             + MatVec_sources + LLE_sources,
                         libraries=['stdc++','blas','lapack','arpack'],
                         library_dirs = LIBS,
                         include_dirs=['include','lib/MatVec',
                                       numpy.get_include()] )
                         

    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
