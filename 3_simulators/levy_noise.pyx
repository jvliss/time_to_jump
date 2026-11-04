"""
levy_noise.pyx -- alpha-stable (Levy) noise increments for simulating the
Levy-flight model.

random_exponential() and random_levy() are Cython ports of
gsl_ran_exponential() and gsl_ran_levy() from the GNU Scientific Library
(randist/exponential.c, randist/levy.c),
Copyright (C) 1996-2007 James Theiler, Brian Gough.
GSL implements the algorithm of:
Chambers, J. M., Mallows, C. L., & Stuck, B. W. (1976). A method for
simulating stable random variables. Journal of the American Statistical
Association, 71(354), 340-344. https://doi.org/10.1080/01621459.1976.10480344

Because it is derived from GSL, this file is licensed under the GNU General
Public License, version 3 or later (see LICENSE_GPL), not under the MIT
License that covers the other software in this repository. It is distributed
WITHOUT ANY WARRANTY; see the GPL for details.
"""

from libc.math cimport log, log1p, M_PI, tan, sqrt, sin, pow, cos, abs
from libc.stdlib cimport rand, RAND_MAX
cimport cython


@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double random_exponential(double mu):
    
    cdef double u = random_uniform()
    return -mu * log1p(-u)



@cython.nonecheck(False)
@cython.cdivision(True) 
cdef double random_levy(double c, double alpha): 
    cdef double u = M_PI * (random_uniform() - 0.5)
    cdef double v = 0.0
    cdef double t, s

    # Cauchy
    if alpha == 1.0:       
        t = tan(u)
        return c * t

    while v == 0:
        v = random_exponential(1.0)

    # Gaussian
    if alpha == 2.0:            
        t = 2 * sin(u) * sqrt(v)
        return c * t

    # General case
    t = sin(alpha * u) / pow(cos (u), 1 / alpha)
    s = pow(cos ((1 - alpha) * u) / v, (1 - alpha) / alpha)

    return c * t * s

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef api double levy_noise(double alpha, double dt): 
    """
    INPUT:
    alpha     - heavy-tailedness of the noise distro
    dt        - time step (0.001 = 1 ms)

    """

    # Declare variables

    cdef double rhs = pow(dt, 1. / alpha) # pre-compute damping factor for noise
    cdef double c = 1. / sqrt(2)  # Fixed c levy parameter

    noise =  rhs*random_levy(c, alpha)
		
    return noise
