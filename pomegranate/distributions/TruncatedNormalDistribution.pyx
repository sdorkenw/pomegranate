#!python
#cython: boundscheck=False
#cython: cdivision=True
# TruncatedNormalDistribution.pyx
# Contact: Jacob Schreiber <jmschreiber91@gmail.com>

import numpy as np
from scipy.optimize import minimize

from ..utils cimport _log
from ..utils cimport isnan
from ..utils import check_random_state

from libc.math cimport sqrt as csqrt
from libc.math cimport exp as cexp

# Define some useful constants
DEF NEGINF = float("-inf")
DEF INF = float("inf")
DEF SQRT_2 = 1.414213562373095
DEF SQRT_2_PI = 2.50662827463
DEF LOG_2_PI = 1.83787706641
DEF H_LOG_2_O_PI = -0.2257913526447274
DEF EPS8 = 1e-8

DEF a1 = 0.0705230784
DEF a2 = 0.0422820123
DEF a3 = 0.0092705272
DEF a4 = 0.0001520143
DEF a5 = 0.0002765672
DEF a6 = 0.0000430638


cdef double _erf(double x_in) nogil:
	cdef double x
	cdef double sgn

	if x_in < 0:
		x = - x_in
		sgn = - 1
	else:
		x = x_in
		sgn = 1

	approx_erf = 1 - 1 / (1 + a1 * x + a2 * x ** 2.0 + a3 * x ** 3.0 + a4 * x ** 4.0 + a5 * x ** 5.0 + a6 * x ** 6.0) ** 16.0

	return sgn * approx_erf


cdef class TruncatedNormalDistribution(Distribution):
	"""A normal distribution based on a mean and standard deviation."""

	property parameters:
		def __get__(self):
			return [self.mu, self.sigma, self.lower, self.upper]
		def __set__(self, parameters):
			self.mu, self.sigma, self.lower, self.upper = parameters

	def __init__(self, mean, std, lower=0, upper=1, frozen=False, min_std=0.0):
		self.mu = mean
		self.sigma = std
		self.lower = lower
		self.upper = upper
		self.name = "TruncatedNormalDistribution"
		self.frozen = frozen
		self.summaries = [0, 0, 0, [], []]
		self.min_std = min_std

		self.log_sigma_theta = self._log_sigma_theta()

	def __reduce__(self):
		"""Serialize distribution for pickling."""
		return self.__class__, (self.mu, self.sigma, self.lower, self.upper, self.frozen)

	def _theta(self):
		return _erf((self.upper - self.mu) / (self.sigma * SQRT_2)) - \
			   _erf((self.lower - self.mu) / (self.sigma * SQRT_2)) + EPS8

	def _log_sigma_theta(self):
		return - _log(self.sigma) - _log(self._theta()) + H_LOG_2_O_PI

	cdef void _log_probability(self, double* X, double* log_probability, int n) nogil:
		cdef int i
		for i in range(n):
			if isnan(X[i]):
				log_probability[i] = 0.
			elif  X[i] < self.lower or X[i] > self.upper:
				log_probability[i] = NEGINF
			else:
				log_probability[i] = - (self.mu - X[i])**2.0 / (2 * self.sigma**2.0) + \
									 self.log_sigma_theta

	def sample(self, n=None, random_state=None):
		if n is None:
			n = 1

		random_state = check_random_state(random_state)

		samples = []
		while len(samples) < n:
			broad_sample = random_state.normal(self.mu, self.sigma, 1)[0]
			if broad_sample > self.lower and broad_sample < self.upper:
				samples.append(broad_sample)

		samples = np.array(samples)
		return samples

	cdef double _summarize(self, double* items, double* weights, int n,
		int column_idx, int d) nogil:
		cdef int i, j
		cdef int offset
		cdef double x_sum = 0.0, x2_sum = 0.0, w_sum = 0.0
		cdef double item
		cdef double _min = 10000.0
		cdef double _max = 0.0
		cdef double[:] filtered_items
		cdef double[:] filtered_weights

		with gil:
			filtered_items = np.empty(n, dtype=np.double)
			filtered_weights = np.empty(n, dtype=np.double)

		for i in range(n):
			item = items[i*d + column_idx]

			filtered_items[i] = item
			filtered_weights[i] = weights[i]

			if isnan(item):
				continue

			w_sum += weights[i]
			x_sum += weights[i] * item
			x2_sum += weights[i] * item * item

		with gil:
			self.summaries[0] += w_sum
			self.summaries[1] += x_sum
			self.summaries[2] += x2_sum
			self.summaries[3].extend(list(filtered_items))
			self.summaries[4].extend(list(filtered_weights))

	def from_summaries(self, inertia=0.0):
		"""
		Takes in a series of summaries, represented as a mean, a variance, and
		a weight, and updates the underlying distribution. Notes on how to do
		this for a Gaussian distribution were taken from here:
		http://math.stackexchange.com/questions/453113/how-to-merge-two-gaussians
		"""
		def _fit_func(x):
			a, b = x
			d = TruncatedNormalDistribution(a, b, self.lower, self.upper)

			data = self.summaries[3]
			weights = self.summaries[4]
			logps = d.log_probability(data)

			wsum_logps = np.sum(logps * weights)

			return - wsum_logps

		# If no summaries stored or the summary is frozen, don't do anything.
		if self.summaries[0] < 1e-8 or self.frozen == True:
			return

		mu = self.summaries[1] / self.summaries[0]
		var = self.summaries[2] / self.summaries[0] - self.summaries[1] ** 2.0 / self.summaries[0] ** 2.0
		sigma = csqrt(var)

		bounds_diff = self.upper - self.lower
		mu, sigma = minimize(_fit_func, (mu, sigma), method='SLSQP',
							 bounds=([self.lower - bounds_diff,
									  self.upper + bounds_diff],
									 [self.min_std, bounds_diff*5])).x

		if sigma < self.min_std:
			sigma = self.min_std

		self.mu = self.mu*inertia + mu*(1-inertia)
		self.sigma = self.sigma*inertia + sigma*(1-inertia)
		self.summaries = [0, 0, 0, [], []]
		self.log_sigma_theta = self._log_sigma_theta()


	def clear_summaries(self):
		"""Clear the summary statistics stored in the object."""

		self.summaries = [0, 0, 0, [], []]

	@classmethod
	def blank(cls):
		return TruncatedNormalDistribution(0, 1, 0, 1)
