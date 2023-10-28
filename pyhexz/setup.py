# Run as
#
#   python3 setup.py build_ext --build-lib=src
#
# to build Cython modules during development and place .so files into src/pyhexz.
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    name="pyhexz",
    ext_modules=cythonize(
        [
            Extension("pyhexz.hexc", ["src/pyhexz/hexc.py"]),
        ],
        language_level="3",
    ),
)
