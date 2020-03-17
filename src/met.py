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

import sys
import copy
from operator import mul

_version = "1.0.0"
_vdate = "2019-09-29"

if not sys.version_info < (3,):
	from functools import reduce

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

def multinom_p(probs, obs):
	"""Return the probability of a vector (list) of observations, 'obs', given a probability
	for each class of observation as specified in the vector (list) 'probs'.  The elements
	of 'probs' should sum to 1.0, and both 'probs' and 'obs' should be of the same length.
	No checks on these constraints are made.
	"""
	return fact(sum(obs))/listprod(list(map(fact, obs))) * listprod([pow(probs[i], obs[i]) for i in range(len(probs))])

def obs_to_probs(obs):
	"""Standardize a list of observations to be a sequence of probabilities that sum to 1.0."""
	lsum = float(sum(obs))
	return [ x/lsum for x in obs ]

class Multinom_Error(Exception):
	pass

def multinom_cases( caselist, counts, categories, items ):
	"""Add to the list of cases a new case consisting of the 'counts' list, after
	distributing 'items' items over 'categories' remaining categories.
	"""
	if categories == 1:
		# There's only one category left to be filled, so put all remaining items in it.
		counts.append(items)
		caselist.append(counts)
	elif items == 0:
		# There are no more items, so put 0 in all remaining categories
		for n in range(categories):
			counts.append(0)
		caselist.append(counts)
	else:
		for n in range(items+1):
			newcounts = copy.copy(counts)
			newcounts.append(n)
			multinom_cases( caselist, newcounts, categories-1, items-n )

def all_multinom_cases( categories, items ):
	"""Returns a list of all multinomial combinations (each a list) of 'items' items distributed
	in all possible ways over 'categories' categories."""
	if categories==0 and items>0:
		raise MultinomError("Can't distribute %d items over 0 cases." % items)
	all_cases = []
	multinom_cases(all_cases, [], categories, items)
	return all_cases

def fillzeroes( case, factor ):
	"""The 'case' argument is a list of counts; if there are any zero counts
	in the case, the non-zero counts are multiplied by 'factor' and the zero counts
	are replaced with 1.
	"""
	return case if min(case)>0 else [ c*factor if c>0 else 1 for c in case ]


class Multinom:
	"""Store two sets of (implictly ordered) multinomial data, perform an exact test,
	and store some or all of the test results.
	"""
	def __init__(self, ref_obs, site_obs):
		"""Arguments are lists.  Each list contains the number (count) of samples (stations)
		in different ordered cagegories (stages).  Stages should be listed from highest (III) to
		lowest (I), (or best to worst).  Both lists should be the same length, and corresponding elements
		in the two lists should represent the same stage.  The list for reference stations
		is first, followed by the list for site stations.
		"""
		self.ref_obs = ref_obs
		self.ref_probs = obs_to_probs(ref_obs)
		self.site_obs = site_obs
		self.save_cases = False
		self.__reset()
	def new_site_obs(site_obs):
		"""Use a different interpretation of site data with the same reference data.
		"""
		self.site_obs = site_obs
		self.__reset()
	def __reset(self):
		"""Perform internal housekeeping for object initialization.
		"""
		self.cases = None
		self.p_value = None
		self.n_extreme_cases = None
		self.__curr_mod_src = None
	def new_src_set(self, src_set):
		"""Internal callback routine.
		"""
		self.__curr_mod_src = [ self.site_obs[i] - src_set[i] for i in range(len(src_set)) ]
		n_to_move = sum(src_set)
		self.redistrib( n_to_move, self.site_obs[len(src_set): ], [] )
	def new_sink_set(self, sink_set):
		"""Internal callback routine.
		"""
		new_case = copy.copy(self.__curr_mod_src)
		new_case.extend(sink_set)
		self.p_value += multinom_p(self.ref_probs, new_case)
		self.n_extreme_cases += 1
		if self.save_cases:
			self.cases.append(new_case)
	def subset( self, sublist, accumlist ):
		"""Find all combinations of elements of 'sublist' (which is a simple list
		of integers [counts]).  Each combination has a length equal to 'sublist',
		and zeroes may appear in any position (category) except the last.  Execute
		'new_src_set()' with each combination.
		"""
		if len(sublist) == 0:
			self.new_src_set(accumlist)
		else:
			if sublist[0] == 0:
				newaccum = copy.copy(accumlist)
				newaccum.append(0)
				self.subset(sublist[1:], newaccum)
			else:
				if len(sublist) == 1:
					for i in range(sublist[0]):
						newelem = i+1
						newaccum = copy.copy(accumlist)
						newaccum.append(newelem)
						self.subset(sublist[1:], newaccum)
				else:
					for i in range(sublist[0]+1):
						newelem = i
						newaccum = copy.copy(accumlist)
						newaccum.append(newelem)
						self.subset(sublist[1:], newaccum)
	def redistrib( self, counts, sublist, accumlist ):
		"""Create lists of all combinations of 'counts' redistributed over 'sublist',
		which is a simple list of integers (counts).  Execute 'new_sink_set()'
		with each combination.
		"""
		if len(sublist) == 1:
			newaccum = copy.copy(accumlist)
			newaccum.append(counts + sublist[0])
			self.new_sink_set(newaccum)
		else:
			for i in range(counts+1):
				newaccum = copy.copy(accumlist)
				newaccum.append(i + sublist[0])
				self.redistrib(counts-i, sublist[1:], newaccum)
	def onesided_exact_test( self, fill_zeroes=False, fill_factor=10, save_cases=False ):
		"""Compute, return, and store the p value for a one-sided
		exact multinomial test of the distribution of site successional stages
		against reference successional stages.
		Arguments:
			fill_zeroes	:	Boolean indicating whether or not the elements of ref_obs should be
							all multiplied by fill_factor and then increased by 1 before
							probabilities are calculated.
			fill_factor	:	The factor to be used in conjunction with fill_zeroes.
			save_cases	:	Boolean indicating whether or not all cases tested should
							be saved in a 'cases' attribute.
		Value:
			The p value of the test.
		Side effects:
			The following attributes of the Multinom object may be modified.
				n_extreme_cases	:	The number of combinations of counts that are
									further skewed from the reference distribution
									than the given site observations.
				cases			:	All combinations of sample counts.  This is
									only set if 'save_cases' = True.
		"""
		self.save_cases = save_cases
		if save_cases:
			self.cases = []
		else:
			self.cases = None
		if fill_zeroes:
			self.ref_probs = obs_to_probs(fillzeroes(self.ref_obs, fill_factor))
		self.p_value = multinom_p(self.ref_probs, self.site_obs)
		self.n_extreme_cases = 0
		ncats = len(self.site_obs) - 1
		for i in range(ncats):
			if self.site_obs[i] > 0:
				orig_src = self.site_obs[:i+1]
				self.subset( orig_src, [])
		if fill_zeroes:
			# Restore unfilled probabilities
			self.ref_probs = obs_to_probs(self.ref_obs)
		return self.p_value
	def twosided_exact_test( self, fill_zeroes=False, fill_factor=10, save_cases=False ):
		"""Compute, return, and store the p value for a two-sided multinomial exact
		test of the distribution of site against reference.
		Arguments:
			fill_zeroes	:	Boolean indicating whether or not the elements of ref_obs should be
							all multiplied by fill_factor and then increased by 1 before
							probabilities are calculated.
			fill_factor	:	The factor to be used in conjunction with fill_zeroes.
			save_cases	:	Boolean indicating whether or not all cases tested should
							be saved in a 'cases' attribute.
		Value:
			The p value of the test.
		Side effects:
			The following attributes of the Multinom object may be modified.
				n_extreme_cases	:	The number of combinations of counts that are
									further skewed from the reference distribution
									than the given site observations.
				cases			:	All combinations of sample counts.  This is
									only set if 'save_cases' = True.
		"""
		self.save_cases = save_cases
		if fill_zeroes:
			self.ref_probs = obs_to_probs(fillzeroes(self.ref_obs, fill_factor))
		site_p = multinom_p(self.ref_probs, self.site_obs)
		self.p_value = site_p
		self.n_extreme_cases = 0
		cases = all_multinom_cases(len(self.site_obs), sum(self.site_obs))
		if save_cases:
			self.cases = cases
		else:
			self.cases = None
		for case in cases:
			p = multinom_p(self.ref_probs, case)
			if p < site_p:
				self.p_value += p
				self.n_extreme_cases += 1
				if self.save_cases:
					self.cases.append(case)
		if fill_zeroes:
			# Restore unfilled probabilities
			self.ref_probs = obs_to_probs(self.ref_obs)
		return self.p_value


def onesided_exact_likelihood( ref_obs, site_obs, ref_samples=None, fill_zeroes=False, fill_factor=10 ):
	"""Calculate the distribution of one-sided exact test probabilities based on the likelihood
	of observing the given distribution of reference area samples.
	Arguments:
		ref_obs		:	A list of reference category counts.
		site_obs	:	A list of site category counts.
		ref_samples	:	The number of reference area samples to use to compute the cases
						that will be used for the multinomial probabilities that will be
						used to calculate likelihood of different reference probabilities.
						If None, sum(ref_obs) will be used instead.
		fill_zeroes	:	Boolean indicating whether or not the elements of ref_obs should be
						all multiplied by fill_factor and then increased by 1 before
						probabilities are calculated.
		fill_factor	:	The factor to be used in conjunction with fill_zeroes.
	Value:
		A list of tuples,where each tuple represents one of the possible distributions of
		reference area samples, consisting of the following elements:
			0) The distribution of reference area samples being evaluated;
			1) The likelihood of that being the true reference area distribution given
				the observed distribution of reference area data;
			2) the raw p value for the one-sided exact test for that reference area
				distribution;
			3) the normalized likelihood for the case (normalized to the maximum
				likelihood); and
			4) the final p value for the one-sided exact test, after the raw p value
				has been multiplied by the normalized likelihood.
	"""
	caselist = []
	max_like = 0
	if ref_samples == None:
		refcases = all_multinom_cases(len(ref_obs), sum(ref_obs))
	else:
		refcases = all_multinom_cases(len(ref_obs), ref_samples)
	if fill_zeroes:
		refcases = [ fillzeroes(case, fill_factor) for case in refcases ]
	for case in refcases:
		raw_like = multinom_p(obs_to_probs(case), ref_obs)
		if raw_like > max_like:
			max_like = raw_like
		m = Multinom(case, site_obs)
		# 'onesided_exact_test()' should not fill zero probabilites for reference cases because that is already done.
		raw_p = m.onesided_exact_test(fill_zeroes=False)
		caselist.append([ case, raw_like, raw_p])
	norm_like = [ case[1]/max_like for case in caselist ]
	adj_p = [ case[2] * case[1]/max_like for case in caselist ]
	map(lambda l1,l2: l1.append(l2), caselist, norm_like)
	map(lambda l1,l2: l1.append(l2), caselist, adj_p)
	return caselist

def twosided_exact_likelihood( ref_obs, site_obs, ref_samples=None, fill_zeroes=False, fill_factor=10 ):
	"""Calculate the distribution of two-sided exact test probabilities based on the likelihood
	of observing the given distribution of reference area samples.
	Arguments:
		ref_obs		:	A list of reference category counts.
		site_obs	:	A list of site category counts.
		ref_samples	:	The number of reference area samples to use to compute the cases
						that will be used for the multinomial probabilities that will be
						used to calculate likelihood of different reference probabilities.
						If None, sum(ref_obs) will be used instead.
		fill_zeroes	:	Boolean indicating whether or not the elements of ref_obs should be
						all multiplied by fill_factor and then increased by 1 before
						probabilities are calculated.
		fill_factor	:	The factor to be used in conjunction with fill_zeroes.
	Value:
		A list of tuples,where each tuple represents one of the possible distributions of
		reference area samples, consisting of the following elements:
			0) The distribution of reference area samples being evaluated;
			1) The likelihood of that being the true reference area distribution given
				the observed distribution of reference area data;
			2) the raw p value for the one-sided exact test for that reference area
				distribution;
			3) the normalized likelihood for the case (normalized to the maximum
				likelihood); and
			4) the final p value for the one-sided exact test, after the raw p value
				has been multiplied by the normalized likelihood.
	"""
	caselist = []
	max_like = 0
	if ref_samples == None:
		refcases = all_multinom_cases(len(ref_obs), sum(ref_obs))
	else:
		refcases = all_multinom_cases(len(ref_obs), ref_samples)
	if fill_zeroes:
		refcases = [ fillzeroes(case, fill_factor) for case in refcases ]
	for case in refcases:
		raw_like = multinom_p(obs_to_probs(case), ref_obs)
		if raw_like > max_like:
			max_like = raw_like
		m = Multinom(case, site_obs)
		# 'twosided_exact_test()' should not fill zero probabilites for reference cases because that is already done.
		raw_p = m.twosided_exact_test(fill_zeroes=False)
		caselist.append([ case, raw_like, raw_p])
	norm_like = [ case[1]/max_like for case in caselist ]
	adj_p = [ case[2] * case[1]/max_like for case in caselist ]
	map(lambda l1,l2: l1.append(l2), caselist, norm_like)
	map(lambda l1,l2: l1.append(l2), caselist, adj_p)
	return caselist
