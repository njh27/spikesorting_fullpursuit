import numpy as np
from scipy.stats import norm
from spikesorting_fullpursuit.consolidate import optimal_align_templates
from spikesorting_fullpursuit.analyze_spike_timing import find_overlapping_spike_bool
from spikesorting_fullpursuit.parallel.segment_parallel import time_window_to_samples

"""CHECKED"""


def find_decision_boundary_equal_var(mu_1, mu_2, var, p_1=0.5):
    """ Helper function to return the decision boundary between two 1D
    Gaussians with equal variance and Gaussian with mean=mu_1 has prior
    probability of p_1.
    This is the solution to:
        p_1 * N(mu_1, sqrt(var)) = p_2 * N(mu_2, sqrt(var))
    where N(mu, sigma) is the normal distribution PDF with mean=mu and
    standard deviation=sigma."""
    if p_1 < 0 or p_1 > 1:
        raise ValueError("Prior probably must lie in the interval [0, 1]")
    if var <= 0:
        raise ValueError("Variance var must be a positive number")
    if mu_1 == mu_2:
        raise ValueError("Means of the Gaussians cannot be equal")
    p_2 = 1 - p_1
    numerator = mu_1 ** 2 - mu_2 ** 2
    denominator = 2 * (mu_1 - mu_2)
    numerator_ll = 2 * var * np.log(p_1/p_2)
    decision_boundary = numerator / denominator - numerator_ll / denominator

    return decision_boundary


def find_decision_boundary(mu_1, mu_2, var_1, var_2):
    """ Helper function to return the decision boundary between 1D Gaussians with
    unequal variance but equal prior probabilities.
    This is the solution to:
        N(mu_1, sqrt(var_1)) = N(mu_2, sqrt(var_2))
    where N(mu, sigma) is the normal distribution PDF with mean=mu and
    standard deviation=sigma."""
    if var_1 <= 0 or var_2 <= 0:
        raise ValueError("Variances must be a positive numbers")
    if mu_1 == mu_2:
        raise ValueError("Means of the Gaussians cannot be equal")
    # coefficients of quadratic equation ax^2 + bx + c = 0
    a = var_1 - var_2
    b = 2 * (mu_1 * var_2 - mu_2 * var_1)
    c = mu_2 ** 2.0 * var_1 - mu_1 ** 2.0 * var_2 - 2 * var_1 * var_2 * np.log(np.sqrt(var_1)/np.sqrt(var_2))
    x1 = (-b + np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    x2 = (-b - np.sqrt(b**2.0 - 4.0 * a * c)) / (2.0 * a)
    # Choose the solution where the PDFs are maximized
    if norm.pdf(x1, mu_1, np.sqrt(var_1)) > norm.pdf(x2, mu_1, np.sqrt(var_1)):
        decision_boundary = x1
    else:
        decision_boundary = x2

    return decision_boundary


def check_template_pair(template_1, template_2, chan_covariance_mats, sort_info):
    """
    Intended for testing whether a sum of templates is equal to a given
    template. Templates are assumed to be aligned with one another as no
    shifting is performed. Probability of confusiong the templates is
    returned. This confusion is symmetric, i.e. p_confusion template_1 assigned
    to template_2 equals p_confusion template_2 assigned to template_1. """
    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']

    # Compute separability given V = template_1.
    E_L_t1 = 0.5 * np.dot(template_1, template_1)
    E_L_t2 = np.dot(template_1, template_2) - 0.5 * np.dot(template_2, template_2)
    var_diff = 0
    for chan in range(0, n_chans):
        t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
        diff_template = template_1[t_win[0]:t_win[1]] - template_2[t_win[0]:t_win[1]]
        var_diff += (diff_template[None, :]
                     @ chan_covariance_mats[chan]
                     @ diff_template[:, None])

    # Expected difference between t1 and t2 likelihood functions
    E_diff_t1_t2 = E_L_t1 - E_L_t2
    if var_diff > 0:
        # Probability likelihood nt - t2 < 0
        p_confusion = norm.cdf(0, E_diff_t1_t2, np.sqrt(var_diff))
    else:
        p_confusion = 1.

    return p_confusion


def compute_separability_metrics(templates, channel_covariance_mats,
                                 sort_info):
    """ Calculate the various variance and template metrics needed to compute
    separability_metrics between units and the delta likelihood function for
    binary pursuit."""

    # Ease of use variables
    n_chans = sort_info['n_channels']
    n_templates = len(templates)
    template_samples_per_chan = sort_info['n_samples_per_chan']

    separability_metrics = {}
    separability_metrics['templates'] = np.vstack(templates)
    # Compute our template sum squared error (see note below).
    separability_metrics['template_SS'] = np.sum(separability_metrics['templates'] ** 2, axis=1)
    separability_metrics['template_SS_by_chan'] = np.zeros((n_templates, n_chans))
    # Get channel covariance of appropriate size from the extra large covariance matrices
    separability_metrics['channel_covariance_mats'] = channel_covariance_mats
    separability_metrics['contamination'] = np.zeros(n_templates)
    separability_metrics['peak_channel'] = np.zeros(n_templates, dtype=np.int64)

    separability_metrics['channel_p_noise'] = np.zeros(n_chans)
    # Expected number of threshold crossing due to noise given sigma
    noise_crossings = 2 * norm.cdf(-sort_info['sigma'], 0, 1) * sort_info['n_samples']
    for chan in range(0, n_chans):
        non_noise_crossings = sort_info['n_threshold_crossings'][chan] - noise_crossings
        # Can't be less than zero
        non_noise_crossings = max(non_noise_crossings, 0.0)
        # separability_metrics['channel_p_noise'][chan] = 1.0 - (non_noise_crossings / sort_info['n_samples'])
        separability_metrics['channel_p_noise'][chan] = ((sort_info['n_samples'] -
                sort_info['n_threshold_crossings'][chan]) / sort_info['n_samples'])

    # Compute variance for each neuron and the boundaries with noise for expected
    # likelihood function distribution given this variance
    separability_metrics['neuron_variances'] = np.zeros(n_templates)
    separability_metrics['neuron_lower_thresholds'] = np.zeros(n_templates)
    for n in range(0, n_templates):
        expectation = 0.5 * separability_metrics['template_SS'][n]
        for chan in range(0, n_chans):
            t_win = [chan*template_samples_per_chan, (chan+1)*template_samples_per_chan]
            separability_metrics['template_SS_by_chan'][n, chan] = np.sum(
                    separability_metrics['templates'][n, t_win[0]:t_win[1]] ** 2)

            separability_metrics['neuron_variances'][n] += (separability_metrics['templates'][n, t_win[0]:t_win[1]][None, :]
                        @ separability_metrics['channel_covariance_mats'][chan]
                        @ separability_metrics['templates'][n, t_win[0]:t_win[1]][:, None])

        separability_metrics['neuron_lower_thresholds'][n] = (
                                expectation - sort_info['sigma_lower_bound']
                                * np.sqrt(separability_metrics['neuron_variances'][n]))

        # Determine peak channel for this unit
        separability_metrics['peak_channel'][n] = ( np.argmax(np.abs(
                                    separability_metrics['templates'][n, :]))
                                    // template_samples_per_chan )
        separability_metrics['contamination'][n] = norm.sf(
                            separability_metrics['neuron_lower_thresholds'][n],
                            -expectation,
                            np.sqrt(separability_metrics['neuron_variances'][n]))

    return separability_metrics


def find_noisy_templates(separability_metrics, sort_info):
    """ Identify templates as noisy if their lower confidene bound is less than
    or equal to the decision boundary at 0. """
    n_neurons = separability_metrics['templates'].shape[0]

    noisy_templates = np.zeros(n_neurons, dtype=np.bool)
    # Find neurons that are too close to noise and need to be deleted
    for neuron in range(0, n_neurons):
        if separability_metrics['neuron_lower_thresholds'][neuron] <= 0.0:
            print("Neuron", neuron, "is NOISY")
            noisy_templates[neuron] = True

    return noisy_templates


def delete_and_threshold_noise(separability_metrics, sort_info,
                                        noisy_templates):
    """ This function first decides whether the templates indicated as noise by
    noisy_templates will be useful for sorting other units. If it is, then it
    is kept. If its detectability is so small or dissimilar to other units,
    then it will be deleted permaneantly. """
    n_chans = sort_info['n_channels']
    max_shift = (sort_info['n_samples_per_chan'] // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    new_noisy_templates = np.copy(noisy_templates)
    for noise_n in range(0, n_neurons):
        # Find each noisy template that could be deleted
        if not noisy_templates[noise_n]:
            continue
        for n in range(0, n_neurons):
            # For each template that will NOT be deleted
            if noisy_templates[n]:
                continue
            # First get aligned templates
            shift_temp_n, shift_temp_noise_n, _, _ = optimal_align_templates(
                                separability_metrics['templates'][n, :],
                                separability_metrics['templates'][noise_n, :],
                                n_chans, max_shift=max_shift, align_abs=True)
            # Expected value of Likelihood function for good neuron, n, given
            # that the true voltage is the noise neuron, noise_n
            expectation_n_noise_n = np.dot(shift_temp_n, shift_temp_noise_n) \
                                    - 0.5 * separability_metrics['template_SS'][n]
            # Find the upper bound of the distribution of the likelihood
            # function for neuron n, given that voltage = noise_n
            noise_match_upper_bound = expectation_n_noise_n + sort_info['sigma_lower_bound'] \
                                * np.sqrt(separability_metrics['neuron_variances'][n])
            if noise_match_upper_bound > 0.0:
                # The likelihood function for good template n given the noise
                # template has enough probability of exceeding threshold to be
                # either incorrectly added to the good unit or improve its
                # sorting so rethreshold the noise unit and keep it
                new_noisy_templates[noise_n] = False
                # Rethreshold the noise unit so that spikes are only added to
                # it if they exceed the sigma upper bound of the noisy unit
                # given that the voltage is noise (i.e. = 0). Since distributions
                # are symmetric, this is the same as the negative of the currently
                # assigned lower threshold.
                separability_metrics['neuron_lower_thresholds'][noise_n] *= -1
                break
            # If we make it here for each neuron, then if we delete the noise
            # unit, spikes from that unit are unlikely to be added to any of
            # the good units and it is of low enough quality that it won't
            # help sorting

    return separability_metrics, new_noisy_templates


def delete_noise_units(separability_metrics, noisy_templates):
    """ Remove data associated with deleted noise templates from
    separability_metrics and reassign values IN PLACE to separability_metrics. """
    for key in separability_metrics.keys():
        if key in ['channel_covariance_mats', 'channel_p_noise']:
            continue
        elif key in ['templates', 'template_SS_by_chan']:
            separability_metrics[key] = separability_metrics[key][~noisy_templates, :]
        elif key in ['template_SS', 'neuron_variances',
                     'neuron_lower_thresholds', 'contamination', 'peak_channel']:
            separability_metrics[key] = separability_metrics[key][~noisy_templates]
        else:
            print("!!! Could not find a condition for metrics key !!!", key)

    return separability_metrics


def add_n_spikes(separability_metrics, neuron_labels):
    """ Adds number of spikes assigned to each unit in separabilit_metrics
    after binary pursuit is run. Spikes are added by modifying the input
    separability_metrics dictionary. Called in binary_pursuit_parallel.
    """
    separability_metrics['n_spikes'] = np.zeros(separability_metrics['templates'].shape[0])
    for n in range(0, separability_metrics['templates'].shape[0]):
        separability_metrics['n_spikes'][n] = np.count_nonzero(neuron_labels == n)

    return None


def pairwise_separability(separability_metrics, sort_info):
    """
    Uses separability metrics to determine the overlap between pairs of units
    and estimate the probability of confusing spikes between the units. Similar
    estimates are made for the probability of adding noise to a unit and missing
    spikes from a unit.
    """
    n_chans = sort_info['n_channels']
    template_samples_per_chan = sort_info['n_samples_per_chan']
    max_shift = (template_samples_per_chan // 4) - 1
    n_neurons = separability_metrics['templates'].shape[0]

    # Compute separability from noise for each neuron
    noise_misses = np.zeros(n_neurons)
    noise_contamination = np.zeros(n_neurons)
    # Compute separability measure for all pairs of neurons
    pair_separability_matrix = np.zeros((n_neurons, n_neurons))
    for n1 in range(0, n_neurons):
        E_L_n1 = 0.5 * separability_metrics['template_SS'][n1]
        noise_misses[n1] = norm.cdf(
                    separability_metrics['neuron_lower_thresholds'][n1], E_L_n1, np.sqrt(separability_metrics['neuron_variances'][n1]))

        E_L_n1_noise = -0.5 * separability_metrics['template_SS'][n1]
        p_spike_added_given_noise = norm.sf(
                    separability_metrics['neuron_lower_thresholds'][n1],
                    E_L_n1_noise, np.sqrt(separability_metrics['neuron_variances'][n1]))
        p_spike_added = separability_metrics['n_spikes'][n1] / sort_info['n_samples']
        # Probability of adding a spike must be at >= conditional probability
        p_spike_added = max(p_spike_added, p_spike_added_given_noise)
        noise_contamination[n1] = (p_spike_added_given_noise *
                separability_metrics['channel_p_noise'][separability_metrics['peak_channel'][n1]]
                / p_spike_added)

        for n2 in range(0, separability_metrics['templates'].shape[0]):
            # Need to optimally align n2 template with n1 template for computing
            # the covariance/difference of their Likelihood functions
            shift_temp1, shift_temp2, optimal_shift, shift_samples_per_chan = optimal_align_templates(
                                separability_metrics['templates'][n1, :],
                                separability_metrics['templates'][n2, :],
                                n_chans, max_shift=max_shift, align_abs=True)
            var_diff = 0
            for chan in range(0, n_chans):
                s_win = [chan*shift_samples_per_chan, (chan+1)*shift_samples_per_chan]
                # Adjust covariance matrix to new templates, shifting according to template 1
                if optimal_shift >= 0:
                    s_chan_cov = separability_metrics['channel_covariance_mats'][chan][optimal_shift:, optimal_shift:]
                else:
                    s_chan_cov = separability_metrics['channel_covariance_mats'][chan][0:optimal_shift, 0:optimal_shift]
                # Multiply template with covariance then template transpose to
                # get variance of the likelihood function (or the difference)
                diff_template = shift_temp1[s_win[0]:s_win[1]] - shift_temp2[s_win[0]:s_win[1]]
                var_diff += (diff_template[None, :]
                             @ s_chan_cov
                             @ diff_template[:, None])

            n_1_n2_SS = np.dot(shift_temp1, shift_temp2)
            # Assuming full template 2 data here as this will be counted in
            # computation of likelihood function
            E_L_n2 = n_1_n2_SS - 0.5 * separability_metrics['template_SS'][n2]

            # Expected difference between n1 and n2 likelihood functions
            E_diff_n1_n2 = E_L_n1 - E_L_n2
            if var_diff > 0:
                pair_separability_matrix[n1, n2] = norm.cdf(0, E_diff_n1_n2, np.sqrt(var_diff))
            else:
                # No variance (should be the same unit)
                pair_separability_matrix[n1, n2] = 0

    return pair_separability_matrix, noise_contamination, noise_misses
