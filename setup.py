#cythonize meshCalc.pyx
from distutils.core import setup, Extension 
from Cython.Build import cythonize
import numpy as np
import platform
import os

ext_modules = [
    Extension(
        "solveWaxSmit",
        ["solveWaxSmit.pyx"],
    )
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs=[np.get_include()]
)   

#run in console under working directory 
#"python setup.py build_ext --inplace"

