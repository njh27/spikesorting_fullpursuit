from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np

ext  =  [Extension( "sort_cython", sources=['sort_cython.pyx'] )]
# compile as: $ python recompile.py build_ext --inplace
# compile as: > python.exe recompile.py build_ext --inplace
setup(
    ext_modules = cythonize(ext, annotate=False, language_level=3),
    include_dirs = [np.get_include()]
)
