from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import pkg_resources
from pkg_resources import DistributionNotFound, VersionConflict
import numpy as np

dependencies = [
    'pyopencl >= 2019.1.2'
    'cython'
]
pkg_resources.require(dependencies)

ext  =  [Extension('spikesorting_fullpursuit.c_cython.sort_cython', sources=['spikesorting_fullpursuit/c_cython/sort_cython.pyx'])]
pkg_req = [
    'cython', 'pyopencl (>= 2019.1.2)'
]

setup(name='spikesorting_fullpursuit',
      version='1.0',
      description='Spike sorting algorithm with overlap detection.',
      author='Nathan Hall',
      author_email='nathan.hall@duke.edu',
      url='https://',
      packages=['spikesorting_fullpursuit'],
      ext_modules=cythonize(ext, annotate=False, language_level=3),
      include_dirs=[np.get_include()],
      requires=pkg_req
     )
