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
