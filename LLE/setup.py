from os.path import join
from os import listdir
import warnings
import numpy

LIBS = ['/usr/lib']

MatVec_sources = [ join('lib/MatVec/',F) \
                       for F in listdir('lib/MatVec') \
                       if F.endswith('.cpp') and not F.endswith('test.cpp') ]
LLE_sources = [join('src',F) for F in listdir('src') \
                   if F.endswith('.cpp') and not F.endswith('test.cpp') ]

def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.system_info import get_info, BlasNotFoundError

    config = Configuration('LLE', parent_package, top_path)

    config.add_subpackage('pyLLE')

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
