from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("levy_noise", ["levy_noise.pyx"])
]

setup (
	ext_modules = cythonize(extensions),
)

#HOW TO
#Initialize compiling:
##python my_setup.py build_ext --inplace

#Check whether compiling was successful: 
##find . -name 'levy.cpython-310-x86_64-linux-gnu.so'
#Expected output: 
##./levy.cpython-310-x86_64-linux-gnu.so
##./build/lib.linux-x86_64-cpython-310/levy.cpython-310-x86_64-linux-gnu.so