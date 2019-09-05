# TruncatedNormalDistribution.pxd
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy
cimport numpy

from .distributions cimport Distribution

cdef class TruncatedNormalDistribution(Distribution):
	cdef double mu, sigma, lower, upper, log_sigma_theta, two_sigma_squared, log_sigma_sqrt_2_pi
	cdef object min_std

