from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np


ext  =  [Extension( "sort_cython", sources=["sort_cython.pyx"] )]
# compile as: $python3.7 setup.py build_ext --inplace
setup(
    ext_modules = cythonize("sort_cython.pyx", annotate=True, language_level=3),
    include_dirs = [np.get_include()]
)
