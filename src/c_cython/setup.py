from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy as np
# import os
# import platform as sys_platform
#
#
#
# # Find path to sort_cython.pyx relative to current file path
# current_path = os.path.realpath(__file__)
# if sys_platform.system() == 'Windows':
#     compile_path = current_path.split('\\')
#     # slice at -1 if this function at same level as c_cython folder
#     compile_path = [x + '\\' for x in compile_path[0:-1]]
#     compile_path = ''.join(compile_path) + 'src\\c_cython\\sort_cython.pyx'
# else:
#     compile_path = current_path.split('/')
#     # slice at -1 if this function at same level as kernels folder
#     compile_path = [x + '/' for x in compile_path[0:-1]]
#     compile_path = ''.join(compile_path) + 'src/c_cython/sort_cython.pyx'


ext  =  [Extension( "sort_cython", sources=['sort_cython.pyx'] )]
# compile as: $python3.7 setup.py build_ext --inplace
setup(
    ext_modules = cythonize(ext, annotate=False, language_level=3),
    include_dirs = [np.get_include()]
)
