##########################################
# ADAPTED FROM THE ORIGINAL VERSION met.py
# original text header, license etc. follow
##########################################

"""Perform exact tests of a (site or test) distribution of multinomial count data against
a distribution of equivalent ordered multinomial count data from another (reference or control)
data set.  Both two-sided and one-sided tests can be performed.  One-sided tests require
that categories be ordered.
"""


# met.py
# Multinomial Exact Tests
#
# PURPOSE:
#	Define an object and methods to represent categorical data from two populations
#	and to allow exact multinomial tests of one population against the other.
#	A practical example of relevant data (and the motivation for writing this module) is
#	the number of samples found at different benthic successional stages at site and
#	reference locations.  These categorical data are ordered, and so a one-sided exact
#	multinomial test can be applied.  A method for two-sided exact multinomial tests
#	is also implemented.
#
# USAGE:
#   This module implements a class that is used to store and manipulate related sets of
#	multinomial data (e.g., sediment profile image [SPI] successional stage data
#   for reference and site locations).  Computation of the p value for a test of site data
#	against reference conditions  requires two Python statements (in addition to the
#   module import statement): one to instantiate the Multinom data object, and one to
#   request that it calculate an exact p value.
#   These statements, if issued at the Python interpreter, might look like this:
#       >>> from met import Multinom
#       >>> my_site = Multinom([6, 2, 1], [86, 24, 15])
#       >>> my_site.onesided_exact_test()
#   The 'onesided_exact_test()' and 'twosided_exact_test()' methods are the principal methods
#	of the Multinom object.  Both methoods return the p value for the exact test.
#   This p value is also stored in the Multinom object after an exact test is carried out
#   (attribute 'p_value').  The number of extreme cases found is also retained following
#   a calculation (attribute 'n_extreme_cases').  The cases themselves may optionally be saved
#   during a calculation (through an optional argument to the '..._exact_test()' methods);
#	these are saved as a list of lists, where each of the component lists is one distribution
#	of site sample counts across SPI stages that is more extreme than the reference area.
#
# NOTES:
#	1. The calculation is performed by summing the multinomial probabilities for the
#		observed distribution and all distributions that are more extreme than observed.
#	2. For one-sided tests, "more extreme" means that one or more observations is shifted
#		from a more-reference-like category (e.g., higher successional stage) to a lower one.
#	3. For two-sided tests, "more extreme" means that the probability of an alternative
#		arrangement of site data has a lower probability than the observed arrangement.
#	4. To carry out one-sided tests, categories should be listed from most reference-like
#		to least reference-like.  (For benthic successional data, Stage 3 should be listed
#		first, and stage 1 listed last.)
#
# AUTHOR:
#	Dreas Nielsen (RDN)
#   dnielsen@integral-corp.com
#
# COPYRIGHT AND LICENSE:
#	Copyright (c) 2009, 2019, R. Dreas Nielsen
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# The GNU General Public License is available at <http://www.gnu.org/licenses/>
#
# HISTORY:
#		 Date		 Revisions
#		----------	------------------------------------------------------------------------
#		2007		Implemented 'listprod()', 'fact()', 'multinom_p()',
#					and 'obs_to_probs()'.  Began code for 'extrema()' and other supporting
#					routines.  RDN.
#		1/26/2008	Wrote 'subset()' and 'redistrib()'.  RDN.
#		1/27/2008	Revised 'subset()', completely re-created 'extrema()', and eliminated
#					other incomplete (and now unnecessary) code.  RDN.
#       3/11/2008   Revised init call to '__reset()'.  RDN.
#		6/16/2008	Began 'all_multinom_cases()'.  RDN.
#       1/10/2009   Added usage and function documentation.  RDN.
#		1/11/2009	Renamed module and class for generality.  RDN.
#       1/13/2009   Completed 'all_multinom_cases()'.  This could go in a sub-module because
#                   it is to support further analysis, and is not strictly necessary for the
#                   Multinom class.  Renamed 'Multinom.exact_test()' to 'Multinom.onesided_exact_test()'.
#                   Added 'onesided_exact_likelihood()'.  RDN.
#		1/14/2009	Added 'fill_zeroes' and 'fill_factor' arguments to 'Multinom.onesided_exact_test()'
#					and 'onesided_exact_likelihood()' to eliminate the (random) zeroes problem
#					in the distribution of (typically a small number of) reference area observations.
#					A default fill factor is set to 10 (this is the value by which all reference
#					observations are multiplied before adding 1 to eliminate the zeroes)--this
#					value could be altered, augmented, or replaced by a heuristic based on the
#					total number of reference area measurements, and the distribution among them
#					(i.e., the more reference area measurements there are, the more certain you
#					are likely to be that an observed zero for some stage is a true zero).  RDN.
#		1/18/2004	Changed to normalize likelihoods to the maximum rather than the sum, for
#					compatibility with individually calculated p values
#					(in 'onesided_exact_likelihood()'), and added the normalized likelihood
#					to the output of 'onesided_exact_likelihood()'.  Corrected set/reset of ref_probs
#					in 'onesided_exact_test()' when zero filling is used. RDN.
#		1/19/2009	Added 'ref_samples' argument to 'onesided_exact_likelihood()'.  RDN.
#		1/20/2009	Added 'fillzeroes()' and modified other routines to use it.  RDN.
#		1/21/2009	Added 'twosided_exact_test()' and 'twosided_exact_likelihood().  RDN.
#		1/25/2009	Edited documentation in source.  RDN.
#		2019-09-27	Modified docstrings.  Changed assignment of total counts in
#					'twosided_exact_test()'.  RDN.
#		2019-09-29	Modified to run under Python 3 as well as 2.  Version no. 1.0.0  RDN.
#============================================================================================
import numpy as np
import copy
from functools import reduce
from operator import mul
from scipy.special import gammaln


def listprod( alist ):
	"""Return the product of all elements of a list."""
	if alist == []:
		return 1
	else:
		return reduce(mul, alist)


def fact(n):
	"""Return the factorial."""
	if n is None:
		return 1
	elif n == 1 or n == 0:
		return 1
	else:
		return reduce(mul, range(1, n+1), 1)


def multinomial_log_probability(observed, null_proportions):
	"""Computes log probability of observed given null proportions using the
	gamme log function. """
	N = np.sum(observed)
	first_term = gammaln(N+1)
	second_term = np.sum(observed * np.log(null_proportions))
	third_term = gammaln(observed + 1)
	log_p = first_term + second_term - np.sum(third_term)
	return log_p


def multinomial_probability(obs, probs):
	"""Computes probability of obs given null probs using factorials. """
	return fact(sum(obs))/listprod(list(map(fact, obs))) * listprod([pow(probs[i], obs[i]) for i in range(len(probs))])


class MultinomialGOF(object):

	def __init__(self, observed, null_proportions, p_threshold=1.):
		if np.any(null_proportions == 0):
			raise ValueError("Cannot have zero expected proportions!")
		self.observed = np.around(observed).astype(np.int64)
		self.null_proportions = null_proportions / np.sum(null_proportions)
		self.p_threshold = p_threshold
		self.stop_recursing = False
		self.p_value = 0.0
		self.n_cats = observed.shape[0]
		self.n_counts = np.sum(observed)

	def multinomial_log_probability(self, observed, null_proportions):
		"""Computes log probability of observed given null proportions using the
		gamme log function. """
		first_term = gammaln(self.n_counts+1)
		second_term = np.sum(observed * np.log(null_proportions))
		third_term = gammaln(observed + 1)
		log_p = first_term + second_term - np.sum(third_term)
		return log_p

	def multinom_cases(self, counts, categories, items):
		"""Add to the list of cases a new case consisting of the 'counts' list, after
		distributing 'items' items over 'categories' remaining categories.
		"""
		if self.stop_recursing:
			return
		if categories == 1:
			# There's only one category left to be filled, so put all remaining items in it.
			counts.append(items)
			# p = multinomial_probability(counts, self.null_proportions)
			log_p = self.multinomial_log_probability(np.array(counts), self.null_proportions)
			if log_p <= self.ref_logp:
				self.p_value += np.exp(log_p)
				if self.p_value >= self.p_threshold:
					self.stop_recursing = True
		elif items == 0:
			# There are no more items, so put 0 in all remaining categories
			for n in range(categories):
				counts.append(0)
			# p = multinomial_probability(counts, self.null_proportions)
			log_p = self.multinomial_log_probability(np.array(counts), self.null_proportions)
			if log_p <= self.ref_logp:
				self.p_value += np.exp(log_p)
				if self.p_value >= self.p_threshold:
					self.stop_recursing = True
		else:
			for n in range(items+1):
				newcounts = copy.copy(counts)
				newcounts.append(n)
				self.multinom_cases(newcounts, categories-1, items-n)

	def all_multinom_cases(self, categories, items):
		"""Returns a list of all multinomial combinations (each a list) of 'items' items distributed
		in all possible ways over 'categories' categories."""
		if categories==0 and items>0:
			raise MultinomError("Can't distribute %d items over 0 cases." % items)
		self.multinom_cases([], categories, items)

	def random_perm_test(self, n_perms=1000):
		"""Compute probability of observed data given null proportions using
		a random sampling permutation test from numpy's multinomial random
		sampling function. """
		# self.ref_p = multinomial_probability(self.observed, self.null_proportions)
		self.ref_logp = self.multinomial_log_probability(self.observed, self.null_proportions)
		# p_distribution = np.zeros(n_perms)
		log_p_distribution = np.zeros(n_perms)
		new_draws = np.random.multinomial(self.n_counts, self.null_proportions, size=(n_perms))
		for perm in range(0, n_perms):
			# p_distribution[perm] = multinomial_probability(new_draws[perm, :], self.null_proportions)
			log_p_distribution[perm] = self.multinomial_log_probability(new_draws[perm, :], self.null_proportions)
		# self.p_value = np.count_nonzero(p_distribution <= self.ref_p) / n_perms
		self.p_value = np.count_nonzero(log_p_distribution <= self.ref_logp) / n_perms
		return self.p_value

	def twosided_exact_test(self):
		"""Compute exact probability of observed data given null proprotions
		by summing over all possible instances with a probability less than or
		equal to the probability of the observed data. """
		# ref_p = multinomial_probability(self.observed, self.null_proportions)
		self.ref_logp = self.multinomial_log_probability(self.observed, self.null_proportions)
		self.stop_recursing = False
		self.p_value = 0.0
		self.log_p_value = 0.0
		self.all_multinom_cases(self.n_cats, self.n_counts)
		return self.p_value
