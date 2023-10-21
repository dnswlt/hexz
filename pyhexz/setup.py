# Run as 
# python3 setup.py build_ext --inplace
# to build Cython modules.
from setuptools import setup
from Cython.Build import cythonize

setup(
    name='pyhexz',
    ext_modules=cythonize("pyhexz/hexc.pyx"),
)
