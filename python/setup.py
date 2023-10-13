from setuptools import setup
from Cython.Build import cythonize

setup(
    name='hexz',
    ext_modules=cythonize("hexc.pyx"),
)
