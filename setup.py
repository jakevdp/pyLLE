#! /usr/bin/env python
#
# Copyright (C) 2011 Jacob Vanderplas <vanderplas@astro.washington.edu>

descr = """python/C++ implementation of Locally Linear Embedding"""

import sys
import os
import shutil

DISTNAME = 'pyLLE'
DESCRIPTION = 'A python module for fast Locally Linear Embedding'
LONG_DESCRIPTION = open('README.rst').read()
MAINTAINER = 'Jacob Vanderplas'
MAINTAINER_EMAIL = 'vanderplas@astro.washington.edu'
#URL = 'http://scikit-learn.sourceforge.net'
LICENSE = 'new BSD'
#DOWNLOAD_URL = 'http://sourceforge.net/projects/scikit-learn/files/'
VERSION = '0.1'

import setuptools  # we are using a setuptools namespace
from numpy.distutils.core import setup

#LIBS should give the location of your systems libraries.  It should contain
#  liblapack.a
#  libblas.a
#  libarpack.a

def configuration(parent_package='', top_path=None):
    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path,
        namespace_packages=['pyLLE'])

    config.add_subpackage('pyLLE')

    return config


if __name__ == "__main__":

    old_path = os.getcwd()
    local_path = os.path.dirname(os.path.abspath(sys.argv[0]))
    # python 3 compatibility stuff.
    # Simplified version of scipy strategy: copy files into
    # build/py3k, and patch them using lib2to3.
    if sys.version_info[0] == 3:
        try:
            import lib2to3cache
        except ImportError:
            pass
        local_path = os.path.join(local_path, 'build', 'py3k')
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
        print("Copying source tree into build/py3k for 2to3 transformation"
              "...")
        shutil.copytree(os.path.join(old_path, 'pyLLE'),
                        os.path.join(local_path, 'pyLLE'))
        import lib2to3.main
        from io import StringIO
        print("Converting to Python3 via 2to3...")
        _old_stdout = sys.stdout
        try:
            sys.stdout = StringIO()  # supress noisy output
            res = lib2to3.main.main("lib2to3.fixes", ['-w'] + [local_path])
        finally:
            sys.stdout = _old_stdout

        if res != 0:
            raise Exception('2to3 failed, exiting ...')

    os.chdir(local_path)
    sys.path.insert(0, local_path)

    setup(configuration=configuration,
          name=DISTNAME,
          maintainer=MAINTAINER,
          include_package_data=True,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          #url=URL,
          version=VERSION,
          #download_url=DOWNLOAD_URL,
          long_description=LONG_DESCRIPTION,
          zip_safe=False,  # the package can run out of an .egg file
          classifiers=[
              'Intended Audience :: Science/Research',
              'Intended Audience :: Developers',
              'License :: OSI Approved',
              'Programming Language :: C++',
              'Programming Language :: Python',
              'Topic :: Software Development',
              'Topic :: Scientific/Engineering',
              'Operating System :: Unix',
             ]
    )
