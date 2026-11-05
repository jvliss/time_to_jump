from setuptools import Extension, setup
from Cython.Build import cythonize

extensions = [
    Extension("levy_noise", ["levy_noise.pyx"])
]

setup (
	ext_modules = cythonize(extensions),
)

#HOW TO
#Compile (run in this folder; requires Cython and a C compiler):
##python my_setup.py build_ext --inplace

#Check whether compiling was successful (prints the location of the compiled module):
##python -c "import levy_noise; print(levy_noise.__file__)"
#Expected output, depending on the platform, e.g.:
##.../levy_noise.cpython-310-x86_64-linux-gnu.so #Linux
##...\levy_noise.cp310-win_amd64.pyd #Windows