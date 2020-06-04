from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext  =  [Extension('src.c_cython.sort_cython', sources=['src/c_cython/sort_cython.pyx'])]
pkg_req = ['numpy', 'Cython']

setup(name='SpikeSorting',
      version='1.0',
      description='Spike sorting algorithm with overlap detection.',
      author='Nathan Hall',
      author_email='nathan.hall@duke.edu',
      url='https://',
      packages=['spikesorting'],
      ext_modules=cythonize(ext, annotate=False, language_level=3),
      include_dirs=[np.get_include()],
      requires=pkg_req
     )
